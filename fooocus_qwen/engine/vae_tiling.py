"""Тайловое декодирование VAE, не подмешивающее края плиток.

Зачем это здесь вместо готового `diffusers`. Его `tiled_decode` смешивает
соседние плитки линейно по всему перекрытию, а значит **включает в кадр
край плитки** — те столбцы, где декодер работает с дополнением и врёт.
Вранье измерено (`tools/experiments/vae_tile_edge.py`): 7.1 уровня из 255 в
последнем столбце плитки и 9.1 в предпоследней строке против 0.2 в её
середине. Левый и верхний края чисты — свёртки каузальные, дополнение
асимметрично, поэтому портится только «дальний» край.

Вес края в линейной смеси мал, но не равен нулю: за десяток пикселей до
границы он ещё около четырёх процентов. Четыре процента от семи уровней —
это те самые тонкие цветные линии на `stride·k − 10`, которые заказчик
видит на гладких градиентах.

Отсюда решение: край плитки не смешивать, а **отбрасывать**. Каждая плитка
теряет ``trim`` пикселей справа и снизу — кроме тех, чей край и есть край
кадра: там дополнение соответствует настоящей границе изображения, врать
декодеру не на чем, а обрезать нечем закрыть.

Оставшиеся части складываются с весами (накопитель и сумма весов), а не
нарезаются встык: деление на сумму весов даёт разложение единицы при любой
геометрии, включая неполные плитки по краям кадра, и переход между
плитками получается плавным, без излома производной на стыке.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch

LOGGER = logging.getLogger(__name__)

# Сколько пикселей отбрасывать с правого и нижнего краёв плитки. Замер даёт
# заметное вранье на 23 столбцах и до 48 строк; 64 — ближайшая круглая
# величина с запасом, кратная коэффициенту сжатия VAE (16).
TRIM = 64


def decode_tiled(
    latent: torch.Tensor,
    decode: Callable[[torch.Tensor], torch.Tensor],
    *,
    tile_latent: int,
    stride_latent: int,
    ratio: int,
    trim: int = TRIM,
) -> torch.Tensor:
    """Декодирует латент плитками, отбрасывая испорченные края.

    ``latent`` — тензор ``(B, C, F, H, W)``; ``decode`` получает срез латента
    и возвращает декодированную плитку в тех же измерениях. Возвращается
    кадр ``(B, C', F, H·ratio, W·ratio)``.
    """
    height, width = latent.shape[-2:]
    overlap = (tile_latent - stride_latent) * ratio
    if trim > overlap:
        raise ValueError(
            f"обрезка {trim} шире перекрытия плиток {overlap}: между плитками остались бы "
            "незакрытые полосы. Уменьшите обрезку или шаг плитки."
        )

    # Кадр мельче плитки режут только чтобы испортить: одна плитка — это и
    # есть цельное декодирование, и краёв внутри кадра у неё нет.
    if height <= tile_latent and width <= tile_latent:
        return decode(latent)

    accumulator: torch.Tensor | None = None
    weights: torch.Tensor | None = None

    for top in tile_starts(height, tile=tile_latent, stride=stride_latent):
        for left in tile_starts(width, tile=tile_latent, stride=stride_latent):
            piece = latent[:, :, :, top : top + tile_latent, left : left + tile_latent]
            decoded = decode(piece)

            # Край кадра обрезать нечем и незачем: соседа за ним нет.
            at_right = left + piece.shape[-1] >= width
            at_bottom = top + piece.shape[-2] >= height
            if not at_right:
                decoded = decoded[..., :, : decoded.shape[-1] - trim]
            if not at_bottom:
                decoded = decoded[..., : decoded.shape[-2] - trim, :]

            if accumulator is None:
                shape = list(decoded.shape)
                shape[-2] = height * ratio
                shape[-1] = width * ratio
                accumulator = torch.zeros(shape, dtype=torch.float32, device=decoded.device)
                weights = torch.zeros(
                    [1] * (len(shape) - 2) + shape[-2:], dtype=torch.float32, device=decoded.device
                )

            window = _window(
                decoded.shape[-2],
                decoded.shape[-1],
                ramp=overlap - trim,
                ramp_top=top > 0,
                ramp_left=left > 0,
                ramp_bottom=not at_bottom,
                ramp_right=not at_right,
                device=decoded.device,
            )
            y0, x0 = top * ratio, left * ratio
            y1, x1 = y0 + decoded.shape[-2], x0 + decoded.shape[-1]
            accumulator[..., y0:y1, x0:x1] += decoded.to(torch.float32) * window
            weights[..., y0:y1, x0:x1] += window

    if accumulator is None or weights is None:  # pragma: no cover — цикл всегда даёт хоть плитку
        raise RuntimeError("латент не дал ни одной плитки")

    uncovered = int((weights <= 0).sum())
    if uncovered:  # pragma: no cover — геометрия проверена тестами на всех размерах
        raise RuntimeError(f"плитки не закрыли {uncovered} пикселей кадра")
    return (accumulator / weights).to(latent.dtype)


def tile_starts(size: int, *, tile: int, stride: int) -> list[int]:
    """Начала плиток по одной оси. Последняя сдвинута назад до полной.

    Наивный шаг ``range(0, size, stride)`` оставляет в конце огрызок: при
    латенте 118 и плитке 32 последняя плитка выходит в шесть клеток. Беда не
    в размере, а в контексте — декодер видит полоску в 96 пикселей вместо
    512 и работает по ней заметно иначе. Дальше эта плитка подмешивается к
    области, которую предыдущая уже закрыла целиком, и на месте, где
    начинается её вес, встаёт видимая полоса: замер даёт до 5.9 уровня из
    255 на ``x ≈ 1800`` кадра шириной 1888, на трёх кадрах с разными
    промтами в одном и том же месте.

    Поэтому последняя плитка не обрезается по остатку, а начинается там,
    где ей хватает места на полный размер. Перекрытие с предыдущей от этого
    растёт — это дешевле, чем неполный контекст, и качество по кадру
    остаётся однородным.
    """
    if size <= tile:
        return [0]
    last = size - tile
    starts = list(range(0, last, stride))
    starts.append(last)
    return starts


def _window(
    height: int,
    width: int,
    *,
    ramp: int,
    ramp_top: bool,
    ramp_left: bool,
    ramp_bottom: bool,
    ramp_right: bool,
    device: torch.device,
) -> torch.Tensor:
    """Вес плитки: единица в середине, плавный спад к смешиваемым краям.

    К краю, за которым соседа нет (граница кадра), спада нет — иначе
    единственный источник этих пикселей входил бы в сумму с малым весом и
    вытягивался бы из шума.
    """
    vertical = _ramp(height, ramp, ramp_top, ramp_bottom, device)
    horizontal = _ramp(width, ramp, ramp_left, ramp_right, device)
    return vertical[:, None] * horizontal[None, :]


def _ramp(length: int, ramp: int, at_start: bool, at_end: bool, device: torch.device) -> torch.Tensor:
    """Одномерное окно. Спад не доходит до нуля: нулевой вес — потерянный пиксель."""
    line = torch.ones(length, dtype=torch.float32, device=device)
    size = min(ramp, length // 2)
    if size <= 0:
        return line
    slope = torch.linspace(1.0 / (size + 1), 1.0, size, dtype=torch.float32, device=device)
    if at_start:
        line[:size] = slope
    if at_end:
        line[length - size :] = slope.flip(0)
    return line


class SeamlessTiledVae:
    """Примесь к VAE, заменяющая тайловый декодер `diffusers` на здешний.

    Класс подменяется у готового объекта (``vae.__class__ = ...``), а не
    создаётся заново: веса уже прочитаны пайплайном, и перезагружать их ради
    одного метода было бы расточительством. Полей примесь не добавляет —
    подмена класса от этого безопасна.
    """

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
        from diffusers.models.autoencoders.vae import DecoderOutput

        ratio = self.spatial_compression_ratio
        if self.tile_sample_min_height != self.tile_sample_min_width:
            raise ValueError("здешний декодер рассчитан на квадратную плитку")
        if self.tile_sample_stride_height != self.tile_sample_stride_width:
            raise ValueError("здешний декодер рассчитан на одинаковый шаг по осям")

        def decode_piece(piece: torch.Tensor) -> torch.Tensor:
            """Повторяет нетайловый путь ``_decode`` для одного куска латента."""
            self.clear_cache()
            x = self.post_quant_conv(piece)
            out = None
            for frame in range(x.shape[2]):
                self._conv_idx = [0]
                decoded = self.decoder(
                    x[:, :, frame : frame + 1],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=(frame == 0),
                )
                out = decoded if out is None else torch.cat([out, decoded], dim=2)
            return out

        decoded = decode_tiled(
            z,
            decode_piece,
            tile_latent=self.tile_sample_min_height // ratio,
            stride_latent=self.tile_sample_stride_height // ratio,
            ratio=ratio,
        )
        self.clear_cache()

        if self.config.patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import _unpatchify

            decoded = _unpatchify(decoded, patch_size=self.config.patch_size)

        # Нетайловый путь клампит результат, и тайловый обязан делать то же:
        # иначе два пути дают разные значения там, где декодер вышел за предел.
        decoded = torch.clamp(decoded, min=-1.0, max=1.0)

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)


def uninstall(vae) -> None:
    """Возвращает штатный декодер `diffusers`.

    Нужна опытам: загрузчик ставит здешний декодер сразу при загрузке, и без
    этой функции сравнение «штатный против здешнего» незаметно превращается
    в сравнение здешнего с самим собой — ровно та ошибка, на которой уже
    один раз сгорело исследование полос.
    """
    for base in type(vae).__bases__:
        if not issubclass(base, SeamlessTiledVae):
            vae.__class__ = base
            return


def install(vae) -> None:
    """Ставит здешний тайловый декодер на уже загруженный VAE."""
    if isinstance(vae, SeamlessTiledVae):
        return
    vae.__class__ = type(f"Seamless{type(vae).__name__}", (SeamlessTiledVae, type(vae)), {})
    LOGGER.debug("Тайловый декодер VAE заменён на бесшовный")

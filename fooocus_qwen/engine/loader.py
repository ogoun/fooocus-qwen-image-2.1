"""Сборка пайплайна: загрузка весов, размещение, вспомогательные режимы."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from . import attention, vae_tiling
from .embeds_cache import EmbedsCache
from .pipeline import QwenImage21StudioPipeline, assert_contract
from .residency import ResidencyManager

LOGGER = logging.getLogger(__name__)



# Плитка и шаг для тайлинга VAE. Размер плитки задаёт память, шаг — как
# часто встречается шов. Полосы убирает не размер, а сам декодер
# (``engine/vae_tiling.py``): измерено на одном латенте, кадр 1280x1888,
# цветные линии считаются по хроматической мерке —
#
#   вариант                линий на решётке плиток   с      ГиБ
#   цельное декодирование                        0   1.3   15.77
#   512/256, штатный tiled_decode                5   5.7    2.37
#   512/256, здешний декодер                     0   4.3    2.36
#
# То есть здешний тайлинг неотличим от цельного декодирования по швам и
# стоит при этом в семь раз меньше памяти
# (docs/research/2026-09-22-shvy-ot-kraev-plitok-vae.md).
VAE_TILE = 512
VAE_TILE_STRIDE = 256


def _configure_vae_tiling(pipe) -> None:
    """Включает тайлинг VAE и заменяет его декодер на бесшовный.

    Сам тайлинг нужен: цельное декодирование кадра 1280x1888 требует 15.8
    ГиБ поверх резидентного трансформера в 13.3 — вместе это больше карты,
    и она уходит в вытеснение (23 с на декодирование вместо двух).

    Шаг задаётся полем напрямую: ``enable_tiling`` его не принимает.

    Размер плитки полос не убирает — он их только разрежает. Убирает их
    замена самого тайлового декодера: штатный смешивает плитки по всему
    перекрытию и потому вносит в кадр их края, где декодер врёт в тридцать
    пять раз сильнее, чем в середине. Здешний края отбрасывает
    (``engine/vae_tiling.py``).
    """
    pipe.vae.enable_tiling(
        tile_sample_min_height=VAE_TILE,
        tile_sample_min_width=VAE_TILE,
    )
    pipe.vae.tile_sample_stride_height = VAE_TILE_STRIDE
    pipe.vae.tile_sample_stride_width = VAE_TILE_STRIDE
    vae_tiling.install(pipe.vae)


def load_int8_transformer(model_dir: Path, int8_file: Path):
    """Трансформер с INT8-весами Unsloth, в режиме «только веса».

    Файл — сериализация torchao: 224 линейных слоя в ``Int8Tensor``, 73
    тензора (нормы, смещения, вложения) в bf16. Модель строится по
    конфигурации основной модели без выделения памяти под веса
    (``init_empty_weights``; буферы — настоящие: частоты временного вложения
    в файле не хранятся) и получает тензоры из файла как есть.

    Квантование активаций, с которым Unsloth отдаёт веса (W8A8), снимается.
    На RTX 3090 перемножение в INT8 не быстрее bf16, а квантование
    активаций на каждом слое добавляет 60 % ко времени шага: 33.9 с против
    21.6 с на кадре 1024² (bf16 — 20.5 с). Веса при этом остаются INT8 —
    видеопамять та же, 6.8 ГиБ вместо 13.3.
    """
    from accelerate import init_empty_weights
    from diffusers import QwenImage21Transformer2DModel
    from safetensors import safe_open
    from torchao.prototype.safetensors.safetensors_support import unflatten_tensor_state_dict

    config = QwenImage21Transformer2DModel.load_config(str(Path(model_dir) / "transformer"))
    with init_empty_weights(include_buffers=False):
        transformer = QwenImage21Transformer2DModel.from_config(config)

    with safe_open(str(int8_file), framework="pt") as handle:
        metadata = handle.metadata()
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    state, leftover = unflatten_tensor_state_dict(tensors, metadata)
    result = transformer.load_state_dict(state, strict=False, assign=True)
    problems = [*result.missing_keys, *result.unexpected_keys, *leftover]
    if problems:
        raise RuntimeError(
            f"INT8-веса не подходят к трансформеру: {len(problems)} несовпадений, первое — {problems[0]}. "
            "Файл повреждён или от другой версии модели; удалите его, и он скачается заново."
        )

    quantized = 0
    for module in transformer.modules():
        weight = getattr(module, "weight", None)
        if weight is not None and hasattr(weight, "act_quant_kwargs"):
            weight.act_quant_kwargs = None
            quantized += 1
    LOGGER.info("INT8-трансформер: %d слоёв в INT8 (только веса)", quantized)
    return transformer.eval()


def load(
    model_dir: Path,
    device: str = "cuda",
    pin_memory: bool = True,
    cache_capacity: int = 4,
    int8_file: Path | None = None,
    sage_attention: bool = False,
) -> tuple[QwenImage21StudioPipeline, ResidencyManager, EmbedsCache]:
    """Загружает модель и раскладывает её по памяти.

    Веса читаются на хост: размещением дальше управляет ResidencyManager, и
    позволить diffusers самому что-то перенести значило бы получить два хозяина
    у одной видеопамяти.

    ``int8_file`` — трансформер из INT8-весов вместо bf16 (шарды bf16 тогда
    не нужны вовсе); ``sage_attention`` — внимание SageAttention, если пакет
    установлен (``engine/attention.py``).
    """
    assert_contract()

    started = time.perf_counter()
    LOGGER.info("Загружаю модель из %s%s", model_dir, " (трансформер INT8)" if int8_file else "")
    extra = {"transformer": load_int8_transformer(model_dir, int8_file)} if int8_file else {}
    pipe = QwenImage21StudioPipeline.from_pretrained(str(model_dir), dtype=torch.bfloat16, **extra)
    attention.apply(pipe.transformer, sage_attention)

    _configure_vae_tiling(pipe)

    residency = ResidencyManager(pipe, device=device, pin_memory=pin_memory)
    residency.start()

    cache = EmbedsCache(capacity=cache_capacity)
    pipe.attach(residency, cache, torch.device(device))

    LOGGER.info("Модель готова за %.1f с", time.perf_counter() - started)
    return pipe, residency, cache

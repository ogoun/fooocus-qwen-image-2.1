"""Тайловое декодирование, которое не подмешивает края плиток.

Декодер VAE врёт на правом и нижнем краях плитки: измерено — 7.1 уровня из
255 в последнем столбце и 9.1 в предпоследней строке против 0.2 в середине,
то есть в тридцать пять и пятьдесят шесть раз больше
(`tools/experiments/vae_tile_edge.py`). Левый и верхний края при этом чисты:
свёртки каузальные, дополнение асимметрично.

Линейное смешивание `diffusers` даёт этому краю ненулевой вес — до четырёх
процентов там, где вранья ещё несколько уровней, — и он проступает в кадре
тонкой цветной линией. Линии в кадрах заказчика стоят на `256k − 10`, ровно
там, куда попадает максимум произведения «вранье × вес».

Лечение единственно возможное: край плитки не использовать вовсе. Здесь это
и проверяется — не на глаз и не на живой модели, а подставным декодером,
который врёт на краю заведомо и грубо.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fooocus_qwen.engine import vae_tiling

RATIO = 16
TILE_LATENT = 32
STRIDE_LATENT = 16
TRIM = 64


def _coordinates(height: int, width: int) -> torch.Tensor:
    """Латент, в котором значение равно номеру своей клетки."""
    values = torch.arange(height * width, dtype=torch.float32).reshape(1, 1, 1, height, width)
    return values


def _honest_decode(tile: torch.Tensor) -> torch.Tensor:
    """Честный декодер: растягивает клетку латента в квадрат ratio×ratio."""
    return tile.repeat_interleave(RATIO, dim=-2).repeat_interleave(RATIO, dim=-1)


def _lying_decode_for(height: int, width: int):
    """Декодер, врущий на правом и нижнем краях плитки — но не кадра.

    Именно так ведёт себя настоящий: дополнение на границе кадра отвечает
    настоящему краю изображения, и врать декодеру там не на чем — замер
    показывает вранье только на внутренних краях плитки.

    Позиция плитки восстанавливается из самого латента: значение клетки
    равно её номеру, то есть ``y * width + x``.

    Величина вранья заведомо больше любого содержимого: просочись оно в
    результат хоть частью, это будет видно сразу.
    """

    def decode(tile: torch.Tensor) -> torch.Tensor:
        decoded = _honest_decode(tile).clone()
        last = int(tile[0, 0, 0, -1, -1].item())
        at_right = last % width == width - 1
        at_bottom = last // width == height - 1
        if not at_bottom:
            decoded[..., -TRIM:, :] += 10_000.0
        if not at_right:
            decoded[..., :, -TRIM:] += 10_000.0
        return decoded

    return decode


def _run(latent: torch.Tensor, decode) -> torch.Tensor:
    return vae_tiling.decode_tiled(
        latent,
        decode,
        tile_latent=TILE_LATENT,
        stride_latent=STRIDE_LATENT,
        ratio=RATIO,
        trim=TRIM,
    )


def test_geometry_is_exact():
    latent = _coordinates(80, 118)
    result = _run(latent, _honest_decode)
    assert result.shape[-2:] == (80 * RATIO, 118 * RATIO)


def test_an_honest_decoder_is_reproduced_exactly():
    """Смешивание само по себе ничего не искажает."""
    latent = _coordinates(80, 118)
    expected = _honest_decode(latent)
    result = _run(latent, _honest_decode)
    assert torch.allclose(result, expected, atol=1e-3), (result - expected).abs().max()


def test_the_lying_edge_never_reaches_the_result():
    """Главное свойство: край плитки в кадр не попадает."""
    latent = _coordinates(80, 118)
    expected = _honest_decode(latent)
    result = _run(latent, _lying_decode_for(80, 118))

    error = (result - expected).abs()
    assert error.max() < 1.0, f"край просочился: максимум ошибки {error.max():.1f}"


def test_the_frame_edge_is_kept_whole():
    """У последней плитки правый край — это край кадра, и он настоящий.

    Обрезать его нельзя: там нет соседа, который бы это место закрыл, и
    обрезка стала бы дырой либо растянутым последним столбцом.
    """
    latent = _coordinates(80, 118)
    expected = _honest_decode(latent)
    result = _run(latent, _honest_decode)

    assert torch.allclose(result[..., :, -1], expected[..., :, -1], atol=1e-3)
    assert torch.allclose(result[..., -1, :], expected[..., -1, :], atol=1e-3)


def test_a_latent_smaller_than_one_tile_is_decoded_in_one_piece():
    latent = _coordinates(20, 20)
    calls = []

    def counting(tile):
        calls.append(tuple(tile.shape[-2:]))
        return _honest_decode(tile)

    result = _run(latent, counting)
    assert calls == [(20, 20)], f"мелкий кадр не надо резать: {calls}"
    assert torch.allclose(result, _honest_decode(latent), atol=1e-3)


def test_every_pixel_is_covered_by_at_least_one_tile():
    """Обрезка не должна оставить дыр: это проверяется на всех размерах."""
    for height, width in [(80, 118), (64, 64), (33, 97), (100, 40)]:
        latent = _coordinates(height, width)
        result = _run(latent, _honest_decode)
        expected = _honest_decode(latent)
        error = (result - expected).abs().max()
        assert error < 1e-2, f"{height}x{width}: расхождение {error}"


def test_trim_that_would_leave_holes_is_refused():
    """Обрезка шире перекрытия оставила бы незакрытые полосы — это ошибка."""
    latent = _coordinates(80, 80)
    with pytest.raises(ValueError, match="обрез"):
        vae_tiling.decode_tiled(
            latent,
            _honest_decode,
            tile_latent=TILE_LATENT,
            stride_latent=STRIDE_LATENT,
            ratio=RATIO,
            trim=(TILE_LATENT - STRIDE_LATENT) * RATIO + 1,
        )


def test_the_loader_installs_the_seamless_decoder():
    """Без этой проверки правку можно снять, и ни один тест не заметит.

    Сам декодер проверяется выше подставным декодером, но проверять надо и
    то, что он вообще попадает в работу: полосы вернулись бы молча.
    """
    from fooocus_qwen.engine import loader

    class _Vae:
        spatial_compression_ratio = 16
        tile_sample_min_height = tile_sample_min_width = 0
        tile_sample_stride_height = tile_sample_stride_width = 0

        def enable_tiling(self, *, tile_sample_min_height, tile_sample_min_width):
            self.tile_sample_min_height = tile_sample_min_height
            self.tile_sample_min_width = tile_sample_min_width

    class _Pipe:
        vae = _Vae()

    pipe = _Pipe()
    loader._configure_vae_tiling(pipe)

    assert isinstance(pipe.vae, vae_tiling.SeamlessTiledVae), "тайловый декодер остался штатным"
    assert pipe.vae.tile_sample_min_height == loader.VAE_TILE
    assert pipe.vae.tile_sample_stride_height == loader.VAE_TILE_STRIDE


def test_installing_twice_changes_nothing():
    """Загрузчик зовут и повторно — цепочка классов расти от этого не должна."""

    class _Vae:
        pass

    vae = _Vae()
    vae_tiling.install(vae)
    first = type(vae)
    vae_tiling.install(vae)
    assert type(vae) is first

"""Тайлинг VAE не должен оставлять швов на гладких градиентах.

Значения по умолчанию у diffusers — плитка 256 при шаге 192, то есть на
шов приходится 64 пикселя растушёвки. На гладком градиенте это видно
глазом: заказчик нашёл светлые вертикальные полосы на кадре с красным
небом раньше, чем их нашли измерения.

Замер на одном и том же латенте (docs/research/
2026-09-22-polosy-ot-tajlinga-vae.md) показал, что плитка 512 при шаге 256
неотличима от полного отсутствия тайлинга — 0.283 против 0.281 по СКО и
шесть светлых столбцов против шести, — а стоит 2.35 ГиБ против 15.77.

Здесь проверяется, что настройка действительно применяется к VAE: вызвать
``enable_tiling()`` без аргументов легко и обратно, а полосы вернутся
молча.
"""

from __future__ import annotations

import pytest

from fooocus_qwen.engine import loader


class _Vae:
    """Столько от VAE, сколько трогает настройка."""

    def __init__(self) -> None:
        self.use_tiling = False
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

    def enable_tiling(self, tile_sample_min_height=None, tile_sample_min_width=None):
        self.use_tiling = True
        if tile_sample_min_height:
            self.tile_sample_min_height = tile_sample_min_height
        if tile_sample_min_width:
            self.tile_sample_min_width = tile_sample_min_width


class _Pipe:
    def __init__(self) -> None:
        self.vae = _Vae()


def test_tiling_stays_on():
    """Выключать нельзя: цельное декодирование не помещается в карту.

    Кадр 1280x1888 требует 15.8 ГиБ поверх резидентного трансформера в
    13.3 — вместе больше двадцати четырёх, и карта уходит в вытеснение.
    """
    pipe = _Pipe()
    loader._configure_vae_tiling(pipe)
    assert pipe.vae.use_tiling is True


def test_the_tile_is_large_enough_to_hide_the_seam():
    pipe = _Pipe()
    loader._configure_vae_tiling(pipe)
    assert pipe.vae.tile_sample_min_height == loader.VAE_TILE
    assert pipe.vae.tile_sample_min_width == loader.VAE_TILE
    assert loader.VAE_TILE >= 512, (
        "плитка меньше 512 измеренно даёт вдвое больше полос"
    )


def test_the_stride_is_set_too_and_leaves_a_wide_blend():
    """Шаг — половина дела, и ``enable_tiling`` его не принимает.

    Именно шаг задаёт ширину перекрытия, по которому соседние плитки
    смешиваются. Оставить его прежним (192) при плитке 512 значило бы
    получить перекрытие в 320 пикселей — не беда, но проверять надо то,
    что задано осознанно, а не то, что осталось от умолчания.
    """
    pipe = _Pipe()
    loader._configure_vae_tiling(pipe)
    assert pipe.vae.tile_sample_stride_height == loader.VAE_TILE_STRIDE
    assert pipe.vae.tile_sample_stride_width == loader.VAE_TILE_STRIDE
    blend = loader.VAE_TILE - loader.VAE_TILE_STRIDE
    assert blend >= 256, f"растушёвка шва всего {blend} px — измеренно этого мало"


def test_the_loader_actually_calls_it():
    """Настройка бесполезна, если её забыли позвать из загрузчика."""
    import inspect

    source = inspect.getsource(loader.load)
    assert "_configure_vae_tiling(pipe)" in source

"""Сборка пайплайна: загрузка весов, размещение, вспомогательные режимы."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from . import vae_tiling
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


def load(
    model_dir: Path,
    device: str = "cuda",
    pin_memory: bool = True,
    cache_capacity: int = 4,
) -> tuple[QwenImage21StudioPipeline, ResidencyManager, EmbedsCache]:
    """Загружает модель и раскладывает её по памяти.

    Веса читаются на хост: размещением дальше управляет ResidencyManager, и
    позволить diffusers самому что-то перенести значило бы получить два хозяина
    у одной видеопамяти.
    """
    assert_contract()

    started = time.perf_counter()
    LOGGER.info("Загружаю модель из %s", model_dir)
    pipe = QwenImage21StudioPipeline.from_pretrained(str(model_dir), dtype=torch.bfloat16)

    _configure_vae_tiling(pipe)

    residency = ResidencyManager(pipe, device=device, pin_memory=pin_memory)
    residency.start()

    cache = EmbedsCache(capacity=cache_capacity)
    pipe.attach(residency, cache, torch.device(device))

    LOGGER.info("Модель готова за %.1f с", time.perf_counter() - started)
    return pipe, residency, cache

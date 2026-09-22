"""Сборка пайплайна: загрузка весов, размещение, вспомогательные режимы."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from .embeds_cache import EmbedsCache
from .pipeline import QwenImage21StudioPipeline, assert_contract
from .residency import ResidencyManager

LOGGER = logging.getLogger(__name__)



# Плитка и шаг для тайлинга VAE. Значения по умолчанию (256 и 192) дают на
# гладких градиентах отчётливые светлые вертикальные полосы: на шов
# приходится всего 64 пикселя растушёвки, и он виден. Заказчик увидел это
# первым, на кадре с красным небом.
#
# Числа выбраны измерением, а не на глаз (docs/research/
# 2026-09-22-polosy-ot-tajlinga-vae.md). На одном и том же латенте:
#
#   вариант          полосы СКО   светлых столбцов   память декодирования
#   без тайлинга          0.281                  6            15.77 ГиБ
#   512 / 256             0.283                  6             2.35 ГиБ
#   256 / 192 (было)      0.527                 13             1.09 ГиБ
#
# То есть 512/256 неотличимо от полного отсутствия тайлинга и стоит при
# этом в семь раз меньше памяти, чем цельное декодирование. Лишние 1.3 ГиБ
# против прежних значений — цена, которую платить стоит.
VAE_TILE = 512
VAE_TILE_STRIDE = 256


def _configure_vae_tiling(pipe) -> None:
    """Включает тайлинг VAE с плиткой, которая не оставляет швов.

    Сам тайлинг нужен: цельное декодирование кадра 1280x1888 требует 15.8
    ГиБ поверх резидентного трансформера в 13.3 — вместе это больше карты,
    и она уходит в вытеснение (23 с на декодирование вместо двух).

    Шаг задаётся полем напрямую: ``enable_tiling`` его не принимает, хотя
    именно он и определяет ширину растушёвки шва.
    """
    pipe.vae.enable_tiling(
        tile_sample_min_height=VAE_TILE,
        tile_sample_min_width=VAE_TILE,
    )
    pipe.vae.tile_sample_stride_height = VAE_TILE_STRIDE
    pipe.vae.tile_sample_stride_width = VAE_TILE_STRIDE


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

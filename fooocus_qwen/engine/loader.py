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

    # На 2K скрытое представление декодируется целиком и занимает заметно больше
    # памяти, чем сами веса VAE; тайлинг снимает этот пик.
    pipe.vae.enable_tiling()

    residency = ResidencyManager(pipe, device=device, pin_memory=pin_memory)
    residency.start()

    cache = EmbedsCache(capacity=cache_capacity)
    pipe.attach(residency, cache, torch.device(device))

    LOGGER.info("Модель готова за %.1f с", time.perf_counter() - started)
    return pipe, residency, cache

"""Пресеты качества: разрешение и число шагов.

Значения рассчитаны из пропускной способности 3090 и уточняются
``tools/benchmark.py`` — таблица в спецификации ведётся по измерениям, а не по
оценкам. Число шагов — самый честный рычаг ускорения: оно линейно управляет
временем и не трогает веса.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityPreset:
    """Один набор параметров качества."""

    name: str
    output_resolution: int
    num_inference_steps: int
    # Дистиллят turbo вместо полной модели (``engine/turbo.py``): свои шаги,
    # узлы и планировщик. ``num_inference_steps`` у такого пресета — это те
    # же шесть шагов, для прогресса и метаданных.
    turbo: bool = False


PRESETS: dict[str, QualityPreset] = {
    "LowQuality": QualityPreset("LowQuality", output_resolution=1024, num_inference_steps=16),
    "MiddleQuality": QualityPreset("MiddleQuality", output_resolution=1536, num_inference_steps=28),
    # Значения из карточки модели: 2K и сорок шагов.
    "MaxQuality": QualityPreset("MaxQuality", output_resolution=2048, num_inference_steps=40),
    # Дистиллят на 6 шагов, площадь 1024²: кадр 1248x832 за ~12 с против ~21
    # у LowQuality. Выше 1024² дистиллят даёт сетку с периодом 8 px — рисунок
    # на уровне токенов латента; её энергия в спектре 1.2 на 1024² (как у
    # полной модели), 2.1 на 1888x1280 и 2.8 на 2528x1696, одинаково на bf16,
    # INT8 и с SageAttention и в эталонном коде автора без нашего движка
    # (docs/research/2026-09-24-uskorenie-turbo-sage-int8.md). Доводка
    # полной моделью сетку ослабляет, но не убирает.
    "Turbo": QualityPreset("Turbo", output_resolution=1024, num_inference_steps=6, turbo=True),
}

NAMES: tuple[str, ...] = ("LowQuality", "MiddleQuality", "MaxQuality", "Turbo")
DEFAULT = "MiddleQuality"


def get(name: str) -> QualityPreset:
    return PRESETS[name]

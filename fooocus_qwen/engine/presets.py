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
    # Дистиллят на 6 шагов при 1536 px: 24.8 с против 94.7 у MiddleQuality
    # при сопоставимом качестве (tools/experiments/turbo.py). Правку автор
    # учил на 1024² и 1536², генерацию — на 1024² и 2048²; 1536 проверено
    # замером и выглядит не хуже.
    "Turbo": QualityPreset("Turbo", output_resolution=1536, num_inference_steps=6, turbo=True),
}

NAMES: tuple[str, ...] = ("LowQuality", "MiddleQuality", "MaxQuality", "Turbo")
DEFAULT = "MiddleQuality"


def get(name: str) -> QualityPreset:
    return PRESETS[name]

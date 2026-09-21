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


PRESETS: dict[str, QualityPreset] = {
    "LowQuality": QualityPreset("LowQuality", output_resolution=1024, num_inference_steps=16),
    "MiddleQuality": QualityPreset("MiddleQuality", output_resolution=1536, num_inference_steps=28),
    # Значения из карточки модели: 2K и сорок шагов.
    "MaxQuality": QualityPreset("MaxQuality", output_resolution=2048, num_inference_steps=40),
}

NAMES: tuple[str, ...] = ("LowQuality", "MiddleQuality", "MaxQuality")
DEFAULT = "MiddleQuality"


def get(name: str) -> QualityPreset:
    return PRESETS[name]

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
    # Отдельный трансформер вместо основного (``TURBO4`` — 4-шаговый
    # дистиллят Viggle, ``TURBO_OFFICIAL`` — Qwen-Image-2.1-Turbo;
    # ``engine/turbo.py::SwapTransformer``). Пустая строка — основной.
    transformer: str = ""
    # Свои узлы расписания — без CFG и негатива (``turbo.call_arguments``);
    # пустой кортеж — штатное расписание полной модели.
    sigmas: tuple[float, ...] = ()


TURBO4 = "turbo4"
TURBO_OFFICIAL = "turbo_official"
# Узлы — здесь, а не в engine/turbo.py: пресеты читает интерфейс, и тянуть
# ради кортежа модуль движка незачем.
SIGMAS4: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
# Официальный Turbo: узлы из его ``model_index.json`` (``sample_sigmas``).
SIGMAS_OFFICIAL: tuple[float, ...] = (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568)


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
    # Черновик: тот же дистиллят на площади 768² — перебрать промты и сиды,
    # а понравившийся кадр повторить на Turbo тем же сидом. 5.4 с против 9.8
    # на 1024² (×1.8, RTX 3090, INT8 + SageAttention), на фрагментах 1:1
    # чисто (tools/experiments/lora_overhead.py, совет из сообщества). 896²
    # почти ничего не даёт — 8.3 с, поэтому именно 768.
    "TurboDraft": QualityPreset("TurboDraft", output_resolution=768, num_inference_steps=6, turbo=True),
    # 4-шаговый дистиллят Viggle (v0.1) от Abiray, GGUF Q4_K_M, отдельный
    # трансформер: 19.3 с на кадр 1024² на RTX 4060 Laptop против 28.7 у
    # Turbo. Надписи держит, правку — хуже Turbo: кадр уходит от исходника
    # (tools/experiments/lowvram_turbo4.py). Первый выбор и возврат с него —
    # подмена трансформера, на 8 ГБ — несколько секунд чтения с диска.
    "Turbo4": QualityPreset("Turbo4", output_resolution=1024, num_inference_steps=4, transformer=TURBO4,
                            sigmas=SIGMAS4),
    # Официальный Qwen-Image-2.1-Turbo (Qwen, 2026-10-09): 8 шагов, CFG 1,
    # свой трансформер, энкодер и VAE — основной модели. В отличие от лоры
    # Viggle чист на любом размере: сетка 8 px на 1536 и 2048 на уровне
    # полной модели. RTX 3090, bf16 + SageAttention: 1024 — 10–13 с, 1536 —
    # ~25 с (Medium — 77), 2048 — ~49 с (Max — ~283)
    # (tools/experiments/turbo_official.py, docs/research/2026-10-10-qwen-turbo.md).
    "QwenTurbo": QualityPreset("QwenTurbo", output_resolution=1024, num_inference_steps=8,
                               transformer=TURBO_OFFICIAL, sigmas=SIGMAS_OFFICIAL),
    "QwenTurboHD": QualityPreset("QwenTurboHD", output_resolution=1536, num_inference_steps=8,
                                 transformer=TURBO_OFFICIAL, sigmas=SIGMAS_OFFICIAL),
    "QwenTurbo2K": QualityPreset("QwenTurbo2K", output_resolution=2048, num_inference_steps=8,
                                 transformer=TURBO_OFFICIAL, sigmas=SIGMAS_OFFICIAL),
}

NAMES: tuple[str, ...] = (
    "LowQuality", "MiddleQuality", "MaxQuality", "Turbo", "TurboDraft", "Turbo4",
    "QwenTurbo", "QwenTurboHD", "QwenTurbo2K",
)
DEFAULT = "MiddleQuality"


def get(name: str) -> QualityPreset:
    return PRESETS[name]

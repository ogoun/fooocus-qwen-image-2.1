r"""Опыт: откуда светлые вертикальные полосы на гладких градиентах.

Заказчик увидел их на кадре с красным небом (`user/outputs/2026-09-22/
05-59-29.png`). Полосы слабые — единицы уровней из 255, — идут во всю
высоту и заметны только там, где градиент гладкий. В официальных примерах
Qwen такого нет.

Наша сборка отличается от стоковой тремя вещами, и только одна из них
трогает декодирование: ``vae.enable_tiling()`` в загрузчике. Тайлинг
декодирует латент перекрывающимися плитками 256 пикселей с шагом 192 и
сшивает их с растушёвкой; на стыках остаётся расхождение. Включено оно
было ради пика памяти при 2K, и цена решения не измерялась.

Второй подозреваемый — точность декодера. Веса грузятся в bfloat16, у
которого восемь бит мантиссы; на гладком градиенте накопленная ошибка
декодера как раз и проявляется полосами, а на текстуре тонет.

Опыт разводит эти два предположения: один и тот же кадр (тот же сид, тот
же промт, тот же размер) считается в четырёх сочетаниях. Мерка численная —
высокочастотный остаток профиля столбцов по гладкой области неба.

Запуск (из корня проекта):
    .venv\Scripts\python tools\experiments\vae_stripes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fooocus_qwen.logging_setup import use_utf8_console

use_utf8_console()

import argparse
import json
import time

import numpy as np
import torch
from PIL import Image

from fooocus_qwen import config, logging_setup
from fooocus_qwen.engine import loader

OUT = config.LOG_DIR / "vae-stripes"

# Параметры взяты из метаданных того самого кадра, чтобы сравнивать с ним же.
SEED = 327573650
STEPS = 28
WIDTH, HEIGHT = 1280, 1888
RESOLUTION = 1536
PROMPT = (
    "The image is a vertical cinematic photograph of a solitary leafless white tree "
    "standing in a desert landscape, set against a dramatic deep crimson sky and a "
    "ground of black sand. The background is dominated by the intense red sky, which "
    "transitions from a lighter dusty rose near the horizon to a darker maroon at the "
    "top, creating a surreal and ominous atmosphere. In the center of the frame, the "
    "stark white tree rises vertically, its smooth, pale bark contrasting sharply with "
    "the surrounding darkness."
)


def stripe_strength(image: Image.Image) -> dict:
    """Высокочастотный остаток профиля столбцов по верхней трети кадра.

    Верхняя треть — небо: гладкий градиент, на котором полосы и видны.
    Плавный ход градиента вычитается скользящим средним, потому что мерить
    надо полосы, а не сам градиент.
    """
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    sky = array[: array.shape[0] // 3].mean(axis=2)
    column = sky.mean(axis=0)

    window = 41
    padded = np.pad(column, window // 2, mode="edge")
    smooth = np.convolve(padded, np.ones(window) / window, mode="valid")
    residual = column - smooth

    return {
        "stripe_sigma": round(float(residual.std()), 3),
        "stripe_peak": round(float(residual.max()), 2),
        "bright_columns": int((residual > residual.mean() + 4 * residual.std()).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Тайлинг и точность VAE против полос")
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    pipe, residency, _cache = loader.load(config.MODEL_DIR)
    original_dtype = next(pipe.vae.parameters()).dtype
    print(f"VAE загружен в {original_dtype}\n", flush=True)

    rows: dict[str, dict] = {}
    for tiling in (True, False):
        for dtype in (original_dtype, torch.float32):
            key = f"{'тайлинг' if tiling else 'целиком'}+{str(dtype).split('.')[-1]}"
            if tiling:
                pipe.vae.enable_tiling()
            else:
                pipe.vae.disable_tiling()
            pipe.vae.to(dtype)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                image = pipe(
                    prompt=PROMPT,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=args.steps,
                    output_resolution=RESOLUTION,
                    generator=torch.Generator(device=pipe._execution_device).manual_seed(SEED),
                ).images[0]
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                rows[key] = {"ok": False, "reason": "нехватка видеопамяти", "detail": str(error)[:80]}
                print(f"{key}: {rows[key]}", flush=True)
                continue

            image.save(OUT / f"{key}.png")
            row = {"ok": True, **stripe_strength(image)}
            row["seconds"] = round(time.perf_counter() - started, 1)
            row["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            rows[key] = row
            print(f"{key}: {row}", flush=True)
            (OUT / "scores.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    pipe.vae.to(original_dtype)

    print("\n=== сводка (чем меньше полос, тем лучше) ===")
    print(f"{'вариант':>22} | {'полосы, СКО':>11} | {'пик':>5} | {'столбцов':>8} | "
          f"{'с':>6} | {'ГиБ':>5}")
    for key, row in sorted(rows.items(), key=lambda item: item[1].get("stripe_sigma", 99)):
        if not row.get("ok"):
            print(f"{key:>22} | {row['reason']}")
            continue
        print(f"{key:>22} | {row['stripe_sigma']:>11} | {row['stripe_peak']:>5} | "
              f"{row['bright_columns']:>8} | {row['seconds']:>6} | {row['peak_vram_gib']:>5}")
    print(f"\nперестановок энкодера: {int(residency.stats()['swaps'])}")
    print(f"кадры: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

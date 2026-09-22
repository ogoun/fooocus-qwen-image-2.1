r"""Опыт: наша ли вина светлые вертикальные полосы.

Опыт `vae_stripes.py` снял два подозрения сразу: полосы не зависят от
точности декодера (bfloat16 против float32 — разница 0.2 %) и остаются без
тайлинга. Значит они в самом латенте, а не в декодировании.

Наша сборка отличается от стоковой тремя вещами, и все три трогают именно
то, что происходит до декодера: перестановка весов между хостом и картой
(``ResidencyManager``), кэш эмбеддингов в переопределённом
``_get_qwen_prompt_embeds`` и тайлинг VAE. Планировщик, сигмы и сдвиг
расписания мы не трогаем вовсе — пайплайн считает их сам.

Здесь один и тот же кадр (тот же сид, промт, размер, число шагов) считается
двумя пайплайнами:

  ``stock``  — ``QwenImage21Pipeline`` как в официальном примере, с
               ``enable_model_cpu_offload`` вместо нашей резидентности,
               без тайлинга и без кэша;
  ``studio`` — наш ``loader.load()`` со всем, что мы добавили.

Если полосы одинаковы — они свойство модели, и лечить их надо настройками
генерации, а не кодом. Если у стокового их нет — виновата наша сборка, и
опыт сузит поиск до трёх мест.

Плечи запускаются **разными процессами**: два пайплайна в одном не
помещаются, а остатки пула аллокатора от первого искажают второй (ровно это
уже испортило замер масштаба референсов).

Запуск (из корня проекта):
    .venv\Scripts\python tools\experiments\stock_vs_studio.py --arm stock
    .venv\Scripts\python tools\experiments\stock_vs_studio.py --arm studio
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

OUT = config.LOG_DIR / "stock-vs-studio"

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
    """Высокочастотный остаток профиля столбцов по гладкому небу."""
    array = np.asarray(image.convert("RGB")).astype(np.float32)
    sky = array[: array.shape[0] // 3].mean(axis=2)
    column = sky.mean(axis=0)
    window = 41
    padded = np.pad(column, window // 2, mode="edge")
    residual = column - np.convolve(padded, np.ones(window) / window, mode="valid")
    return {
        "stripe_sigma": round(float(residual.std()), 3),
        "stripe_peak": round(float(residual.max()), 2),
        "bright_columns": int((residual > residual.mean() + 4 * residual.std()).sum()),
    }


def build(arm: str):
    if arm == "studio":
        from fooocus_qwen.engine import loader

        pipe, _residency, _cache = loader.load(config.MODEL_DIR)
        return pipe

    # Стоковый путь: ровно как в примере из докстроки пайплайна, только с
    # выгрузкой на хост — 33 ГБ весов в 24 ГБ карты иначе не помещаются, и
    # это штатный для diffusers способ, а не наша самодеятельность.
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline.from_pretrained(str(config.MODEL_DIR), dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    return pipe


def main() -> int:
    parser = argparse.ArgumentParser(description="Стоковый пайплайн против нашего")
    parser.add_argument("--arm", choices=("stock", "studio"), required=True)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    pipe = build(args.arm)
    device = getattr(pipe, "_execution_device", torch.device("cuda"))

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    image = pipe(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=args.steps,
        output_resolution=RESOLUTION,
        generator=torch.Generator(device=device).manual_seed(SEED),
    ).images[0]
    elapsed = time.perf_counter() - started

    image.save(OUT / f"{args.arm}.png")
    row = {"ok": True, **stripe_strength(image)}
    row["seconds"] = round(elapsed, 1)
    row["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    row["size"] = f"{image.size[0]}x{image.size[1]}"

    path = OUT / "scores.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rows[args.arm] = row
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{args.arm}: {row}")
    if len(rows) == 2:
        a, b = rows.get("stock", {}), rows.get("studio", {})
        print("\n=== сводка ===")
        for name, r in (("stock", a), ("studio", b)):
            print(f"{name:>8} | полосы СКО {r.get('stripe_sigma')} | пик {r.get('stripe_peak')} "
                  f"| столбцов {r.get('bright_columns')} | {r.get('seconds')} с")
        if a.get("stripe_sigma") and b.get("stripe_sigma"):
            ratio = b["stripe_sigma"] / a["stripe_sigma"]
            print(f"\nнаш/стоковый: {ratio:.2f} — "
                  + ("полосы свойство модели, не наша вина" if 0.8 <= ratio <= 1.25
                     else "расхождение существенное, виновата сборка"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

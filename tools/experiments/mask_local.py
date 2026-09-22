r"""Опыт второго круга: локальная правка, где маска действительно решает.

Первый круг (`mask_protocol.py`) просил «сделать волосы белыми» и
тем себя обесценил: волосы выходят за маску, и перекрасить их целиком —
правильный ответ модели, а не промах. Отношение «внутри к снаружи» для такой
задачи ничего не измеряет.

Здесь сюжет локальный по существу: добавить предмет, которого в кадре нет, в
явно очерченной области. Всё, что изменилось вне этой области, — уже промах,
и его можно считать честно.

Главное же — здесь проверяется другая **кодировка** маски. Первый круг
показал, что объяснять маску словами не просто бесполезно, а вредно:
формулировка «чёрное сохранить в точности как в <image1>» заставила модель
вырезать фон в прозрачность вместо того, чтобы его сохранить. Модель мыслит
альфа-каналом — она его умеет и строит охотно.

Значит маску логично подавать не вторым чёрно-белым изображением, а **дыркой
в альфа-канале самого исходника**: одно условное изображение RGBA, где
область правки прозрачна. Такая кодировка не требует ни второго изображения,
ни тегов (при одном изображении теги официально запрещены) и говорит модели
ровно то, что нужно, на её собственном языке. Сравниваем три кодировки:

  ``pair``  — как сейчас: исходник плюс чёрно-белая маска вторым изображением;
  ``hole``  — исходник RGBA с прозрачной дыркой на месте правки;
  ``none``  — без маски вовсе, опорная точка.

Запуск:
    .venv\Scripts\python tools\experiments\mask_local.py
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
from PIL import Image, ImageDraw

from fooocus_qwen import config, logging_setup
from fooocus_qwen.engine import loader
from fooocus_qwen.imaging import masking

OUT = config.LOG_DIR / "mask-local"
TASK = "a small red rose tucked into her hair"

# Формулировок две, а не три: «называющая маску» из первого круга выбыла —
# она порождает прозрачность, а не локальность, и повторять её не за чем.
PROMPTS: dict[str, str] = {
    "bare": TASK,
    "fill": (
        f"Fill in the missing part of the picture: {TASK}. Match the existing "
        f"lighting, texture and perspective. The result must be a complete opaque "
        f"photograph with no transparent areas."
    ),
}


def hard(mask: Image.Image) -> Image.Image:
    array = np.asarray(mask.convert("L"))
    return Image.fromarray(np.where(array >= 128, 255, 0).astype(np.uint8), mode="L")


def scores(source: Image.Image, produced: Image.Image, binary: np.ndarray) -> dict:
    a = np.asarray(source.convert("RGB")).astype(np.float32)
    scaled = produced.resize(source.size, Image.LANCZOS)
    b = np.asarray(scaled.convert("RGB")).astype(np.float32)
    delta = np.abs(a - b).mean(axis=2)
    inside, outside = delta[binary], delta[~binary]

    opaque = 100.0
    if scaled.mode == "RGBA":
        alpha = np.asarray(scaled)[..., 3]
        opaque = round(float((alpha >= 250).mean() * 100), 1)

    return {
        "inside_mean": round(float(inside.mean()), 2),
        "outside_mean": round(float(outside.mean()), 2),
        "ratio": round(float(inside.mean() / max(outside.mean(), 1e-6)), 2),
        "outside_changed_pct": round(float((outside > 16).mean() * 100), 1),
        "opaque_pct": opaque,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Локальная правка: уважает ли модель маску")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    source = Image.open(config.LOG_DIR / "smoke" / "source.png").convert("RGB")
    width, height = source.size

    # Пятно сбоку у виска: место, где цветок уместен, и заведомо небольшое.
    raw = Image.new("L", source.size, 0)
    ImageDraw.Draw(raw).ellipse(
        (int(width * 0.60), int(height * 0.22), int(width * 0.76), int(height * 0.38)), fill=255
    )
    raw.save(OUT / "mask.png")

    sharp = hard(masking.refine(raw, grow=8, feather=0))
    binary = np.asarray(sharp).astype(np.float32) >= 128

    pipe, residency, _cache = loader.load(config.MODEL_DIR)

    # Дырка в альфе: исходник RGBA, где область правки полностью прозрачна.
    holed = source.convert("RGBA")
    alpha = np.asarray(sharp).copy()
    holed.putalpha(Image.fromarray((255 - alpha).astype(np.uint8), mode="L"))
    holed.save(OUT / "input-hole.png")

    encodings = {
        "pair": [source, masking.as_condition(sharp)],
        "hole": [holed],
        "none": [source],
    }

    rows: dict[str, dict] = {}
    for mask_kind, condition in encodings.items():
        for prompt_kind, prompt in PROMPTS.items():
            key = f"{mask_kind}+{prompt_kind}"
            started = time.perf_counter()
            produced = pipe(
                prompt=prompt,
                image=condition,
                num_inference_steps=args.steps,
                output_resolution=args.resolution,
                generator=torch.Generator(device=pipe._execution_device).manual_seed(66),
            ).images[0]
            produced.save(OUT / f"{key}.png")
            row = scores(source, produced, binary)
            row["seconds"] = round(time.perf_counter() - started, 1)
            rows[key] = row
            print(f"{key}: {row}", flush=True)
            (OUT / "scores.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    print("\n=== сводка (чем меньше «снаружи», тем точнее модель попала в маску) ===")
    print(f"{'вариант':>16} | {'внутри':>7} | {'снаружи':>8} | {'снаружи изм.':>13} | {'непрозр.':>9}")
    for key, row in sorted(rows.items(), key=lambda item: item[1]["outside_mean"]):
        print(f"{key:>16} | {row['inside_mean']:>7} | {row['outside_mean']:>8} | "
              f"{row['outside_changed_pct']:>11} % | {row['opaque_pct']:>8} %")
    print(f"\nперестановок энкодера: {int(residency.stats()['swaps'])}")
    print(f"кадры: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r"""Опыт: во что обходится работа вне видеокарты, когда исходник большой.

Граф прохождения строился на исходнике 1024x1024, и всё, что не трансформер
и не VAE, заняло там меньше двух процентов: склейка 102 мс, доля
обрезанного 59 мс, запись PNG 125 мс, отпечаток для кэша 4 мс. На таком
фоне оптимизировать их незачем.

Но все они линейны по числу пикселей, а пользователь правит фотографии с
телефона и камеры. Фотография 6000x4000 — это в двадцать три раза больше
пикселей, и те же операции превращаются в секунды. Здесь это меряется, а не
предполагается.

Видеокарта не нужна: меряются numpy, PIL и hashlib.

Запуск (из корня проекта):
    .venv\Scripts\python tools\experiments\cpu_side.py
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
from PIL import Image, ImageDraw

from fooocus_qwen import config
from fooocus_qwen.engine.embeds_cache import fingerprint
from fooocus_qwen.imaging import masking, metadata

OUT = config.LOG_DIR / "cpu-side"


def timed(label: str, function, repeats: int = 3) -> tuple[str, float]:
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - started)
    return label, best


def main() -> int:
    parser = argparse.ArgumentParser(description="Стоимость работы вне видеокарты")
    parser.add_argument("--sizes", type=int, nargs="*", default=[1024, 2048, 4096],
                        help="сторона квадратного кадра")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}

    for side in args.sizes:
        size = (side, side)
        rng = np.random.default_rng(7)
        source = Image.fromarray(rng.integers(0, 255, (side, side, 3), dtype=np.uint8), "RGB")
        produced = Image.fromarray(rng.integers(0, 255, (side, side, 3), dtype=np.uint8), "RGB")
        raw = Image.new("L", size, 0)
        ImageDraw.Draw(raw).ellipse((side // 4, side // 4, side * 3 // 4, side * 3 // 4), fill=255)
        soft = masking.refine(raw, grow=8, feather=12)

        measured = dict([
            timed("отпечаток для кэша", lambda: fingerprint(source), args.repeats),
            timed("refine маски", lambda: masking.refine(raw, grow=8, feather=12), args.repeats),
            timed("маска в условное", lambda: masking.as_condition(soft), args.repeats),
            timed("склейка blend", lambda: masking.blend(source, produced, soft), args.repeats),
            timed("доля обрезанного", lambda: masking.clipped_share(source, produced, soft),
                  args.repeats),
            timed("запись PNG", lambda: metadata.save_png(
                produced, OUT / f"probe-{side}.png", {"prompt": "x"}), args.repeats),
        ])
        rows[str(side)] = {key: round(value, 3) for key, value in measured.items()}
        rows[str(side)]["ИТОГО"] = round(sum(measured.values()), 3)
        print(f"{side}x{side}: {rows[str(side)]}", flush=True)

    (OUT / "scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    print("\n=== секунды по размеру кадра ===")
    labels = [key for key in rows[str(args.sizes[0])] if key != "ИТОГО"] + ["ИТОГО"]
    header = " | ".join(f"{str(side) + 'px':>9}" for side in args.sizes)
    print(f"{'операция':>22} | {header}")
    for label in labels:
        cells = " | ".join(f"{rows[str(side)][label]:>9}" for side in args.sizes)
        print(f"{label:>22} | {cells}")
    print(f"\njson: {OUT / 'scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

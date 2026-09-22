r"""Опыт: что именно считает видеокарта внутри шага денойзинга.

Граф прохождения (`tools/profile_pipeline.py`) довёл до узла «шаг
трансформера» — 82.6 % всего времени, — и дальше упёрся: наш профилировщик
видит вызов целиком, а не то, из чего он состоит. Здесь берётся
``torch.profiler``: он показывает время каждого ядра CUDA и даёт разложить
эти секунды по операциям.

Мерится установившийся шаг, а не первый: первый идёт по ветке prefill, где
считается вся совместная последовательность, а последующие — по ветке
decode, где заново считаются только запросы целевого изображения. Работу
делают именно последующие.

Запуск (из корня проекта):
    .venv\Scripts\python tools\experiments\kernel_profile.py
    .venv\Scripts\python tools\experiments\kernel_profile.py --with-source
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fooocus_qwen.logging_setup import use_utf8_console

use_utf8_console()

import argparse
import json

import torch
from PIL import Image
from torch.profiler import ProfilerActivity, profile

from fooocus_qwen import config, logging_setup
from fooocus_qwen.engine import loader

OUT = config.LOG_DIR / "kernel-profile"
PROMPT = "a wooden desk with a brass lamp and a cup of tea, warm afternoon light"


def main() -> int:
    parser = argparse.ArgumentParser(description="Разложение шага денойзинга по ядрам CUDA")
    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument("--steps", type=int, default=4,
                        help="первый шаг — prefill, он из разложения исключается")
    parser.add_argument("--with-source", action="store_true")
    parser.add_argument("--top", type=int, default=22)
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    pipe, _residency, _cache = loader.load(config.MODEL_DIR)
    device = pipe._execution_device

    condition = None
    if args.with_source:
        condition = [Image.open(config.LOG_DIR / "smoke" / "source.png").convert("RGB")]

    def run(steps: int):
        return pipe(
            prompt=PROMPT,
            image=condition,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=steps,
            output_resolution=1024 if condition else args.resolution,
            output_type="latent",
            generator=torch.Generator(device=device).manual_seed(7),
        )

    print("прогрев…", flush=True)
    run(2)
    torch.cuda.synchronize()

    print("профилирую…", flush=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        run(args.steps)
        torch.cuda.synchronize()

    events = prof.key_averages()
    total_cuda = sum(event.self_device_time_total for event in events)
    rows = []
    for event in sorted(events, key=lambda e: -e.self_device_time_total):
        if event.self_device_time_total <= 0:
            continue
        rows.append({
            "op": event.key,
            "self_cuda_ms": round(event.self_device_time_total / 1000, 1),
            "share_pct": round(100 * event.self_device_time_total / total_cuda, 1),
            "calls": event.count,
        })

    name = f"{'edit' if condition else 't2i'}-{args.resolution}"
    (OUT / f"{name}.json").write_text(
        json.dumps({"total_cuda_ms": round(total_cuda / 1000, 1), "ops": rows[:60]},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {name}: {args.steps} шагов, всего на карте {total_cuda / 1000:.0f} мс ===")
    print(f"{'операция':>46} | {'своё, мс':>9} | {'доля':>6} | {'вызовов':>8}")
    for row in rows[: args.top]:
        print(f"{row['op']:>46} | {row['self_cuda_ms']:>9} | {row['share_pct']:>5}% | "
              f"{row['calls']:>8}")
    print(f"\njson: {OUT / (name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

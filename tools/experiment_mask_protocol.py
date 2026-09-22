r"""Опыт: как заставить модель уважать маску, а не только нашу склейку.

Визуальный разбор дымового прогона показал, что при правке по маске модель
меняет кадр целиком (перекрашивает все волосы), а маска работает только в
нашей финальной склейке — оттуда и резкий эллиптический шов. Официальная
рекомендация Qwen гласит: маска должна быть с резкими краями, чисто белое —
править, чисто чёрное — сохранить, «серые и размытые переходы сбивают модель
с толку». Мы же подаём ей растушёванную маску: растушёвка нужна склейке, но
модели она вредит.

Здесь проверяются два независимых предположения:
  1. резкость маски как условного изображения;
  2. упоминание маски в самом промте (без него модель не знает, что
     ``<image2>`` — маска, а не ещё одна картинка для композиции).

Мерим не на глаз. Пайплайн перерисовывает кадр целиком, поэтому вне маски
совпадения пиксель в пиксель не будет никогда — важно отношение: насколько
сильно изменилось внутри маски по сравнению с тем, насколько изменилось
снаружи. Чем оно выше, тем точнее модель попала в заданную область.

Запуск:
    .venv\Scripts\python tools\experiment_mask_protocol.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from fooocus_qwen.imaging import masking

OUT = config.LOG_DIR / "mask-protocol"
TASK = "make her hair white"

# Формулировки промта. «Голая» — то, что оболочка отправляет сейчас.
PROMPTS: dict[str, str] = {
    "bare": TASK,
    "named": (
        f"<image1> is the photograph. <image2> is a mask: the white area marks the only "
        f"region that may change, the black area must stay exactly as in <image1>. "
        f"Inside the white area of <image2>: {TASK}."
    ),
    "inpaint": (
        f"Edit only the region marked white in <image2>. Keep every other pixel of "
        f"<image1> unchanged. {TASK.capitalize()}, matching the existing lighting, "
        f"texture and perspective at the edges of the region."
    ),
}


def hard(mask: Image.Image) -> Image.Image:
    """Резкая маска: только 0 и 255, без промежуточных значений."""
    array = np.asarray(mask.convert("L"))
    return Image.fromarray(np.where(array >= 128, 255, 0).astype(np.uint8), mode="L")


def scores(source: Image.Image, produced: Image.Image, binary: np.ndarray) -> dict:
    """Насколько изменилось внутри маски против снаружи — и не стала ли маска альфой.

    Второй вопрос здесь не менее важен первого. Разбор дорисовки полей показал,
    что модель делает белую область маски **прозрачной**: везде, где альфа
    неполная, под ней лежит пурпур — цвет, которым декодер заполняет то, чего
    «нет». Похоже, при отсутствии инструкций второе условное изображение
    читается как альфа-матовка, а не как указание области правки. Поэтому доля
    непрозрачных пикселей внутри маски — прямая проверка этой гипотезы: если
    формулировка промта сработала, она обязана быть около ста процентов.
    """
    a = np.asarray(source.convert("RGB")).astype(np.float32)
    scaled = produced.resize(source.size, Image.LANCZOS)
    b = np.asarray(scaled.convert("RGB")).astype(np.float32)
    delta = np.abs(a - b).mean(axis=2)
    inside, outside = delta[binary], delta[~binary]

    if scaled.mode == "RGBA":
        alpha = np.asarray(scaled)[..., 3]
        opaque_inside = round(float((alpha[binary] >= 250).mean() * 100), 1)
    else:
        opaque_inside = 100.0

    return {
        "inside_mean": round(float(inside.mean()), 2),
        "outside_mean": round(float(outside.mean()), 2),
        "ratio": round(float(inside.mean() / max(outside.mean(), 1e-6)), 2),
        "outside_changed_pct": round(float((outside > 16).mean() * 100), 1),
        "opaque_inside_pct": opaque_inside,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Подбор протокола передачи маски модели")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    source = Image.open(config.LOG_DIR / "smoke" / "source.png").convert("RGB")
    raw = Image.open(config.LOG_DIR / "smoke" / "edit-mask-input.png").convert("L")
    soft = masking.refine(raw, grow=8, feather=12)   # как сейчас
    sharp = hard(masking.refine(raw, grow=8, feather=0))
    binary = np.asarray(sharp).astype(np.float32) >= 128

    pipe, residency, _cache = loader.load(config.MODEL_DIR)

    rows: dict[str, dict] = {}
    for mask_kind, mask in (("soft", soft), ("sharp", sharp)):
        for prompt_kind, prompt in PROMPTS.items():
            key = f"{mask_kind}+{prompt_kind}"
            started = time.perf_counter()
            result = pipe(
                prompt=prompt,
                image=[source, masking.as_condition(mask)],
                num_inference_steps=args.steps,
                output_resolution=args.resolution,
                generator=torch.Generator(device=pipe._execution_device).manual_seed(66),
            )
            produced = result.images[0]
            produced.save(OUT / f"{key}.png")
            row = scores(source, produced, binary)
            row["seconds"] = round(time.perf_counter() - started, 1)
            rows[key] = row
            print(f"{key}: {row}", flush=True)
            (OUT / "scores.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    print("\n=== сводка (чем выше отношение, тем точнее попадание в маску) ===")
    print(f"{'вариант':>16} | {'внутри':>7} | {'снаружи':>8} | {'отнош.':>7} | "
          f"{'снаружи изм.':>13} | {'непрозр. внутри':>16}")
    for key, row in sorted(rows.items(), key=lambda item: -item[1]["ratio"]):
        print(f"{key:>16} | {row['inside_mean']:>7} | {row['outside_mean']:>8} | "
              f"{row['ratio']:>7} | {row['outside_changed_pct']:>12} % | "
              f"{row['opaque_inside_pct']:>15} %")
    print(f"\nперестановок энкодера: {int(residency.stats()['swaps'])}")
    print(f"кадры: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

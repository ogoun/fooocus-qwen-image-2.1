r"""Опыт: чем чинить дорисовку полей.

Дымовой прогон дал в новой площади прозрачность вместо продолжения сцены
(разбор — docs/research/2026-09-22-maska-kak-alfa.md). Опыты с локальной
правкой показали два виновника: растушёванная маска, которую модель читает
как альфа-матовку, и любое упоминание прозрачности в промте, которое её
включает даже в отрицательной форме («без прозрачных областей»).

Здесь три кодировки на том же сюжете, что и в прогоне:

  ``soft``  — как сейчас: бордюр продолжением краевых пикселей, маска
              растушёвана на 12 пикселей;
  ``sharp`` — то же, но маска резкая, чисто чёрно-белая;
  ``hole``  — одно изображение RGBA: бордюр прозрачен, второй картинки нет.

Первый заход провалили все три кодировки разом — прозрачным вышел **весь**
кадр, а не только новая площадь. Значит виновата не кодировка. Снимок `hole`
объяснил, что произошло: модель вырезала человека наклейкой. Прозрачный
бордюр в обучающих данных так и выглядит, а промт «continue the scene
naturally» описывает **операцию**, а не сцену, и ничем модель не удерживает.
Официальная рекомендация Qwen про правку ровно об этом: описывать свет,
перспективу и стиль существующей сцены, а не действие.

Поэтому второй заход проверяет два удерживающих рычага:

  ``prompt``   — описание сцены вместо описания операции;
  ``negative`` — отрицательный промт против прозрачности при true_cfg_scale.

Второй рычаг не противоречит находке первого круга («слово „прозрачность“ в
промте её включает»): там оно стояло в положительном промте, где модель
слышит слово и не слышит отрицания при нём. В отрицательном промте механизм
обратный — он вычитается из предсказания, а не прибавляется к нему.

Мерка прямая: доля непрозрачных пикселей в новой площади. Всё, что ниже ста
процентов, — уже брак, потому что исходник непрозрачен.

Запуск:
    .venv\Scripts\python tools\experiment_outpaint.py
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
from fooocus_qwen.imaging import masking, outpaint

OUT = config.LOG_DIR / "outpaint-fix"
VAGUE = "continue the scene naturally"
SCENE = (
    "a full photograph of a woman with long dark hair, bare shoulders, looking at "
    "the camera, standing against a plain light grey studio backdrop, soft even "
    "studio lighting, the whole frame filled with the studio wall"
)
NO_CUTOUT = "transparent background, cut-out, sticker, alpha channel, checkerboard"


def main() -> int:
    parser = argparse.ArgumentParser(description="Кодировка маски при дорисовке полей")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--amount", type=float, default=0.35)
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    source = Image.open(config.LOG_DIR / "smoke" / "source.png").convert("RGB")
    plan = outpaint.plan(source.size, ["left", "bottom"], args.amount)
    canvas, mask = outpaint.expand(source, plan)
    new_area = np.asarray(mask) > 127

    soft = masking.refine(mask, grow=8, feather=12)
    sharp_array = np.where(np.asarray(masking.refine(mask, grow=8, feather=0)) >= 128, 255, 0)
    sharp = Image.fromarray(sharp_array.astype(np.uint8), mode="L")

    # Дырка в альфе: тот же холст, но новая площадь прозрачна, а не заполнена
    # продолжением края. Второе изображение не нужно вовсе.
    holed = canvas.copy()
    holed.putalpha(Image.fromarray(np.where(new_area, 0, 255).astype(np.uint8), mode="L"))
    holed.save(OUT / "input-hole.png")

    pair_soft = [canvas, masking.as_condition(soft)]
    pair_sharp = [canvas, masking.as_condition(sharp)]

    # Каждый прогон — кодировка, промт и (не)отрицательный промт. Порядок
    # такой, чтобы дешёвые прогоны без true_cfg_scale шли первыми.
    trials = {
        "sharp+сцена": (pair_sharp, SCENE, None),
        "hole+сцена": ([holed], SCENE, None),
        "soft+сцена": (pair_soft, SCENE, None),
        "sharp+сцена+отрицание": (pair_sharp, SCENE, NO_CUTOUT),
        "sharp+операция+отрицание": (pair_sharp, VAGUE, NO_CUTOUT),
    }

    pipe, residency, _cache = loader.load(config.MODEL_DIR)

    rows: dict[str, dict] = {}
    for key, (condition, prompt, negative) in trials.items():
        started = time.perf_counter()
        produced = pipe(
            prompt=prompt,
            negative_prompt=negative,
            true_cfg_scale=4.0 if negative else 1.0,
            image=condition,
            # Размер кадра НЕ задаётся: так делает и оболочка при
            # aspect.FOLLOW_REFERENCE, и воспроизвести надо именно тот режим,
            # в котором дефект наблюдался. Явные 1408x1408 при двух условных
            # изображениях уводят карту в вытеснение — проверено.
            num_inference_steps=args.steps,
            output_resolution=args.resolution,
            generator=torch.Generator(device=pipe._execution_device).manual_seed(66),
        ).images[0]
        produced.save(OUT / f"{key}.png")

        scaled = produced.resize(plan.canvas_size, Image.LANCZOS)
        if scaled.mode == "RGBA":
            alpha = np.asarray(scaled)[..., 3]
            opaque = round(float((alpha[new_area] >= 250).mean() * 100), 1)
        else:
            opaque = 100.0
        rows[key] = {
            "opaque_in_new_area_pct": opaque,
            "frame": f"{produced.size[0]}x{produced.size[1]}",
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(f"{key}: {rows[key]}", flush=True)
        (OUT / "scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    print("\n=== сводка (сто процентов и есть годный результат) ===")
    for key, row in sorted(rows.items(), key=lambda item: -item[1]["opaque_in_new_area_pct"]):
        print(f"{key:>6} | непрозрачно в новой площади {row['opaque_in_new_area_pct']:>6} % "
              f"| кадр {row['frame']}")
    print(f"\nперестановок энкодера: {int(residency.stats()['swaps'])}")
    print(f"кадры: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

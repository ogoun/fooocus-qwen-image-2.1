r"""Опыт: правка тем же сидом, которым кадр сгенерирован, — портится ли она.

Повод (2026-10-28): Kokon (docs/engineering-notes.md, stable-diffusion.cpp):
правка тем же сидом в том же рабочем размере стартует с шума, из которого
кадр родился. Результат пересвечен и часто игнорирует промт. Видели на
2368×1760 и 2048², на ~1 Мп — нет. Kokon подменяет такой сид.

У нас другой движок (diffusers), шум — ``torch.Generator(seed)`` формы
латента. При правке в том же размере форма та же, значит шум тот же — но
важен ли он, решит только опыт.

Рабочий путь ``Generator.generate``: генерация сидом S, затем правка без
маски тем же S и двумя другими. Задачи такие, что послушание мерится числом:
- ``bw`` — «чёрно-белая фотография»: средняя насыщенность (HSV S, 0..255)
  должна упасть к нулю;
- ``night`` — «ночь»: средняя яркость должна упасть.

Ещё две меры: ``change`` — средняя разница с исходником (правка, которую
игнорировали, мала); ``sharp`` — дисперсия лапласиана к исходнику (больше
единицы — резче исходника, «пересвечено» по Kokon).

Кадры — в ``tmp/edit_same_seed/``.

Запуск (нужна видеокарта):
    .venv\Scripts\python tools\experiments\edit_same_seed.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import generator as gen  # noqa: E402
from fooocus_qwen.engine import presets  # noqa: E402
from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402

WORK = ROOT / "tmp" / "edit_same_seed"
SCENES = {
    "market": "a photograph of a busy outdoor fruit market at noon, colourful stalls with oranges, lemons and "
              "green apples, people in bright clothes, blue sky",
    "garden": "a photograph of a cottage garden in full bloom, red and purple flowers along a stone path, a "
              "wooden bench, warm afternoon sun",
}
TASKS = {
    "bw": "turn it into a black and white photograph",
    "night": "make it night, dark sky, the scene lit only by a few warm lamps",
}
OTHER_SEEDS = (1000, 2000)


def measures(source, image) -> dict:
    src = np.asarray(source.convert("RGB"))
    out = np.asarray(image.convert("RGB").resize(source.size))
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)

    def lap(a):
        return float(cv2.Laplacian(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

    return {
        "saturation": round(float(hsv[..., 1].mean()), 1),
        "brightness": round(float(hsv[..., 2].mean()), 1),
        "change": round(float(np.abs(src.astype(np.float32) - out.astype(np.float32)).mean()), 1),
        "sharp": round(lap(out) / max(lap(src), 1e-6), 2),
    }


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(description="Editing with the seed that made the image")
    parser.add_argument("--presets", nargs="*", default=["QwenTurbo", "QwenTurbo2K"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--precision", default="bf16")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR = WORK / "outputs"
    config.SETTINGS_FILE = WORK / "settings.json"
    config.SETTINGS_FILE.write_text(json.dumps({"precision": args.precision, "sage_attention": True}), encoding="utf-8")

    from fooocus_qwen.ui.state import Studio

    generator = Studio(config.AppConfig(lang="en", preload=False)).generator
    rows = []
    for preset_name in args.presets:
        preset = presets.get(preset_name)
        for scene, prompt in SCENES.items():
            source = generator.generate(gen.GenerationRequest(
                prompt=prompt, preset=preset, aspect="1:1", seed=args.seed))[0].image.convert("RGB")
            source.save(WORK / f"{preset_name}_{scene}_source.png")
            base = measures(source, source)
            for task, instruction in TASKS.items():
                for offset in (0, *OTHER_SEEDS):
                    seed = args.seed + offset
                    started = time.perf_counter()
                    edited = generator.generate(gen.GenerationRequest(
                        prompt=instruction, preset=preset, seed=seed, source=source,
                        mask_mode=gen.MASK_NONE))[0].image
                    seconds = time.perf_counter() - started
                    edited.save(WORK / f"{preset_name}_{scene}_{task}_seed{seed}.png")
                    row = {"preset": preset_name, "scene": scene, "task": task,
                           "seed": "same" if offset == 0 else f"other{offset}", "size": list(edited.size),
                           "seconds": round(seconds, 1), "source_saturation": base["saturation"],
                           "source_brightness": base["brightness"], **measures(source, edited)}
                    rows.append(row)
                    print(row, flush=True)
                    (WORK / "scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== same seed against other seeds ===")
    print(f"{'preset':12} {'scene':7} {'task':6} {'seed':9} {'sat':>6} {'bright':>6} {'change':>6} {'sharp':>6}")
    for row in rows:
        print(f"{row['preset']:12} {row['scene']:7} {row['task']:6} {row['seed']:9} {row['saturation']:6} "
              f"{row['brightness']:6} {row['change']:6} {row['sharp']:6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

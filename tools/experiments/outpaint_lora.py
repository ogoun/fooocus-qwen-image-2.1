r"""Опыт: расширение кадра лорой outpaint вместо маски — держится ли кадр на месте.

Повод (2026-10-02): прежнее расширение рисовало маску на новой площади, и
правка шла вторым шагом. Пользователь хочет «сразу расширять», как сценарий
ComfyUI с лорой ``ausboss/Qwen-Image-2.1-Outpaint-LoRA``: серый холст
целиком — исходник правки без маски, лора держит кадр на месте, оригинал
вклеивается обратно (``imaging/outpaint.py``).

Мера, как в карточке лоры: PSNR сохранённой части — нарисованное там, где
лежит оригинал, против самого оригинала в масштабе генерации (дБ, выше —
лучше; ниже ~30 — кадр сдвинут или перерисован, вклейка даст шов). И доля
серого, оставшегося незаполненным. Сравниваются: с лорой и без, Turbo и
полная модель, расширение вбок и до 16:9.

Запуск (нужны веса модели, Turbo и лоры outpaint — качается сама):
    .venv\Scripts\python tools\experiments\outpaint_lora.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import presets  # noqa: E402
from fooocus_qwen.engine.generator import GenerationRequest  # noqa: E402
from fooocus_qwen.imaging import outpaint  # noqa: E402
from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402

WORK = ROOT / "tmp" / "outpaint_lora"
SOURCES = ROOT / "tmp" / "docs_screenshots"


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main() -> int:
    use_utf8_console()
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Outpaint LoRA: does the picture stay in place")
    parser.add_argument("--precision", default="", help="override the precision (temporary settings file)")
    parser.add_argument("--profile", default="", help="override the memory profile: high or low")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR = WORK / "outputs"
    if args.precision or args.profile:
        from fooocus_qwen import settings

        chosen = settings.load()
        config.SETTINGS_FILE = WORK / "settings.json"
        config.SETTINGS_FILE.write_text(json.dumps({
            "precision": args.precision or chosen.precision, "sage_attention": chosen.sage_attention,
            "memory_profile": args.profile or chosen.memory_profile,
        }), encoding="utf-8")
    from fooocus_qwen.ui.state import Studio

    studio = Studio(config.AppConfig(lang="en", preload=False))
    generator = studio.generator
    studio.ensure_turbo_weights()  # в профиле «low» — свой, облегчённый адаптер
    outpaint_lora, failure = studio.outpaint_lora("en")
    if failure:
        print(failure)
        return 1

    cases = [
        ("mug", "showcase-mug.png", ("sides", ["left", "right"], 0.35)),
        ("fox", "showcase-fox.png", ("ratio", "16:9")),
        ("storefront", "showcase-storefront.png", ("sides", ["top"], 0.5)),
    ]
    print(f"{'case':12} {'preset':10} {'lora':5} {'kept PSNR':>9} {'gray left':>9} {'seconds':>7}")
    for name, file, how in cases:
        source = Image.open(SOURCES / file).convert("RGB")
        plan = outpaint.plan(source.size, how[1], how[2]) if how[0] == "sides" else outpaint.plan_for_ratio(source.size, how[1])
        for preset in ("Turbo", "LowQuality"):
            job = outpaint.prepare(source, plan, outpaint.generation_area(presets.get(preset).output_resolution))
            for with_lora in (True, False):
                started = time.perf_counter()
                produced = generator.generate(GenerationRequest(
                    prompt=outpaint.prompt(), preset=presets.get(preset), source=job.canvas, seed=42,
                    size=job.size, reference_scale=job.resolution,
                    loras=(outpaint_lora,) if with_lora else (),
                ))[0].image.convert("RGB")
                seconds = time.perf_counter() - started
                left, top, right, bottom = plan.paste_box
                box = tuple(round(v * s) for v, s in zip(
                    (left, top, right, bottom),
                    (job.size[0] / plan.canvas_size[0], job.size[1] / plan.canvas_size[1]) * 2))
                drawn = np.asarray(produced, dtype=np.float32)[box[1]:box[3], box[0]:box[2]]
                truth = np.asarray(job.canvas, dtype=np.float32)[box[1]:box[3], box[0]:box[2]]
                outside = np.asarray(produced, dtype=np.float32).copy()
                outside[box[1]:box[3], box[0]:box[2]] = np.nan
                flat = outside[~np.isnan(outside).any(axis=2)]
                gray = float((np.abs(flat - 128).max(axis=1) < 6).mean() * 100) if flat.size else 0.0
                tag = f"{name}_{preset}_{'lora' if with_lora else 'plain'}"
                produced.save(WORK / f"{tag}_raw.png")
                outpaint.stitch(source, produced, job).save(WORK / f"{tag}.png")
                print(f"{name:12} {preset:10} {str(with_lora):5} {psnr(drawn, truth):9.1f} {gray:8.1f}% {seconds:7.1f}",
                      flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

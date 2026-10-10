r"""Опыт: официальный Qwen-Image-2.1-Turbo против Turbo Viggle и полной модели.

Повод (2026-10-10): Qwen выпустили свой ускоренный чекпоинт — 8 шагов, CFG 1,
расписание в чекпоинте, показан на 2K. У нынешнего Turbo (лора Viggle, 6
шагов) выше 1 Мп появляется сетка с периодом 8 px
(``docs/research/2026-09-24-uskorenie-turbo-sage-int8.md``). Вопросы:

1. сколько стоит кадр официального Turbo на 1024, 1536 и 2048;
2. есть ли у него сетка 8 px на крупных кадрах (мера — остаток высоких
   частот, свёрнутый по фазе 8×8: размах средней клетки в уровнях 0..255 к
   её шуму; у полной модели — фон);
3. как это смотрится рядом с Medium (28 шагов, 1536) и Viggle Turbo (1024).

Настройки — во временном файле (точность ``--precision``, по умолчанию bf16):
выбор пользователя не трогается. Картинки — в ``tmp/turbo_official/``.

Запуск (нужны веса модели, Viggle Turbo и официального Turbo):
    .venv\Scripts\python tools\experiments\turbo_official.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import presets  # noqa: E402
from fooocus_qwen.engine.generator import GenerationRequest  # noqa: E402
from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402

WORK = ROOT / "tmp" / "turbo_official"
PROMPTS = {
    "portrait": "a close-up photograph of an elderly woman with deep wrinkles and silver hair, soft window light, "
                "shallow depth of field, detailed skin texture",
    "poster": 'a minimalist travel poster with the large title "KYOTO" and the line "Autumn 2026", red maple '
              "leaves, a pagoda silhouette, flat colours, clean typography",
    "sky": "a calm lake at dawn under a smooth pastel sky, mist over the water, a lone rowing boat",
}


def grid_score(image: Image.Image) -> tuple[float, float]:
    """Размах средней клетки 8×8 высокочастотного остатка и её отношение к шуму."""
    gray = image.convert("L")
    blur = np.asarray(gray.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    residual = np.asarray(gray, dtype=np.float32) - blur
    h8, w8 = residual.shape[0] // 8 * 8, residual.shape[1] // 8 * 8
    cells = residual[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8)
    mean_cell = cells.mean(axis=(0, 2))
    amplitude = float(mean_cell.max() - mean_cell.min())
    noise = float(residual.std() / np.sqrt(cells.shape[0] * cells.shape[2]))
    return amplitude, amplitude / max(noise, 1e-6)


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(description="Official Qwen-Image-2.1-Turbo against Viggle Turbo and the full model")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR = WORK / "outputs"
    config.SETTINGS_FILE = WORK / "settings.json"
    config.SETTINGS_FILE.write_text(json.dumps({"precision": args.precision, "sage_attention": True}), encoding="utf-8")

    from fooocus_qwen.ui.state import Studio

    studio = Studio(config.AppConfig(lang="en", preload=False))
    generator = studio.generator

    def official(resolution: int) -> presets.QualityPreset:
        return presets.QualityPreset(f"Official{resolution}", resolution, 8, transformer=presets.TURBO_OFFICIAL,
                                     sigmas=presets.SIGMAS_OFFICIAL)

    variants = [
        ("official-1024", official(1024)), ("official-1536", official(1536)), ("official-2048", official(2048)),
        ("viggle-1024", presets.get("Turbo")), ("medium-1536", presets.get("MiddleQuality")),
    ]
    print(f"{'variant':15} {'prompt':9} {'size':>11} {'seconds':>7} {'grid amp':>8} {'grid/noise':>10}")
    for name, preset in variants:
        for tag, prompt in PROMPTS.items():
            # Первый кадр варианта включает подмену трансформера — его время
            # отдельно: в таблице — второй.
            for _attempt in range(2 if tag == "portrait" else 1):
                started = time.perf_counter()
                image = generator.generate(GenerationRequest(
                    prompt=prompt, preset=preset, aspect="1:1", seed=args.seed))[0].image
                seconds = time.perf_counter() - started
            image.save(WORK / f"{name}_{tag}.png")
            amplitude, ratio = grid_score(image)
            print(f"{name:15} {tag:9} {str(image.size):>11} {seconds:7.1f} {amplitude:8.2f} {ratio:10.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

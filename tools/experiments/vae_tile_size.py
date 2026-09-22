r"""Опыт: сколько памяти стоит цельное декодирование без тайлинга.

Светлые вертикальные полосы на гладких градиентах порождает тайлинг VAE, и
только он: на самой гладкой полосе кадра без тайлинга их ноль, с тайлингом
— девять столбцов с пиком в семь уровней. Точность декодера ни при чём
(bfloat16 против float32 — разница 0.5 %), и стоковый пайплайн diffusers,
который тайлинг не включает, даёт ровно ноль.

Тайлинг был включён ради пика памяти, и цена решения не измерялась. Здесь
она измеряется: сколько стоит **само декодирование** без тайлинга, отдельно
от денойзинга. От этого числа зависит, можно ли тайлинг просто выключить
или придётся освобождать место, выселяя трансформер на время декодирования.

Денойзинг считается один раз (``output_type="latent"``), дальше меряются
только декодирования — сравнение чистое.

Запуск (из корня проекта):
    .venv\Scripts\python tools\experiments\vae_tile_size.py
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

OUT = config.LOG_DIR / "vae-tile-size"

SEED = 327573650
STEPS = 28
WIDTH, HEIGHT = 1280, 1888
RESOLUTION = 1536
PROMPT = (
    "The image is a vertical cinematic photograph of a solitary leafless white tree "
    "standing in a desert landscape, set against a dramatic deep crimson sky and a "
    "ground of black sand. The background is dominated by the intense red sky, which "
    "transitions from a lighter dusty rose near the horizon to a darker maroon at the "
    "top, creating a surreal and ominous atmosphere."
)


def stripe_strength(image: Image.Image) -> dict:
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


@torch.no_grad()
def decode(pipe, latents, height: int, width: int) -> Image.Image:
    """Повторяет хвост пайплайна (строки 828-842) для готового латента.

    ``no_grad`` обязателен: сам пайплайн выполняется под ним, а здесь мы
    вызываем декодер напрямую, и без него тензор приходит с графом
    вычислений — ``postprocess`` падает на ``numpy()``.
    """
    unpacked = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    unpacked = unpacked.to(pipe.vae.dtype)
    shape = (1, pipe.vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(shape).to(unpacked.device, unpacked.dtype)
    std = torch.tensor(pipe.vae.config.latents_std).view(shape).to(unpacked.device, unpacked.dtype)
    decoded = pipe.vae.decode(unpacked * std + mean, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Размер плитки VAE против полос")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--size", type=int, nargs=2, default=None,
                        help="кадр, по умолчанию 1280 1888")
    parser.add_argument("--evict", action="store_true",
                        help="выселить трансформер на хост перед декодированием")
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    width, height = args.size if args.size else (WIDTH, HEIGHT)
    pipe, residency, _cache = loader.load(config.MODEL_DIR)

    print(f"кадр {width}x{height}; считаю латент один раз…", flush=True)
    latents = pipe(
        prompt=PROMPT,
        height=height,
        width=width,
        num_inference_steps=args.steps,
        output_resolution=max(width, height),
        output_type="latent",
        generator=torch.Generator(device=pipe._execution_device).manual_seed(SEED),
    ).images

    # У тайлинга два параметра, и второй важнее первого. Размер плитки
    # задаёт память, а **шаг** — ширину перекрытия, по которому соседние
    # плитки смешиваются: чем меньше шаг относительно плитки, тем шире
    # растушёвка шва. По умолчанию 256/192, то есть на шов приходится всего
    # 64 пикселя, и он виден. ``enable_tiling`` шаг не принимает, но поле
    # у VAE открытое.
    variants: list[tuple[str, int, int]] = [
        ("без тайлинга", 0, 0),
        ("256/192 (сейчас)", 256, 192),
        ("256/128", 256, 128),
        ("512/256", 512, 256),
        ("768/384", 768, 384),
        ("1024/512", 1024, 512),
    ]

    rows: dict[str, dict] = {}
    for key, tile, stride in variants:
        if tile == 0:
            pipe.vae.disable_tiling()
        else:
            pipe.vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile)
            pipe.vae.tile_sample_stride_height = stride
            pipe.vae.tile_sample_stride_width = stride

        if args.evict:
            # Во время декодирования трансформер не нужен: 13.3 ГБ, которые
            # он занимает, и есть то место, которого не хватает цельному
            # декодированию.
            residency._transformer.to_host()  # опыт лезет во внутренности намеренно
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            image = decode(pipe, latents, height, width)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rows[key] = {"ok": False, "reason": "нехватка видеопамяти"}
            print(f"{key}: {rows[key]}", flush=True)
            continue

        safe = key.replace(" ", "-").replace("/", "x").replace("(", "").replace(")", "")
        image.save(OUT / f"{safe}.png")
        row = {"ok": True, **stripe_strength(image)}
        row["decode_seconds"] = round(time.perf_counter() - started, 1)
        row["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        rows[key] = row
        print(f"{key}: {row}", flush=True)
        (OUT / "scores.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("\n=== сводка ===")
    print(f"{'вариант':>16} | {'полосы СКО':>10} | {'пик':>5} | {'столбцов':>8} | "
          f"{'с':>5} | {'ГиБ':>6}")
    for key, row in rows.items():
        if not row.get("ok"):
            print(f"{key:>16} | {row['reason']}")
            continue
        print(f"{key:>16} | {row['stripe_sigma']:>10} | {row['stripe_peak']:>5} | "
              f"{row['bright_columns']:>8} | {row['decode_seconds']:>5} | {row['peak_vram_gib']:>6}")
    print(f"\nкадры: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

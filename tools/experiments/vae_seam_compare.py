r"""Опыт: остаются ли швы после замены тайлового декодера.

Мерка независима от содержимого кадра — сравнивается не «красивее ли стало»,
а расхождение с **цельным декодированием** того же латента. Цельное
декодирование швов не имеет по построению: плиток в нём нет.

Вдобавок считается хроматическая мерка, которой швы и были найдены в кадрах
заказчика: зелёный против среднего красного и синего, высокочастотный
остаток профиля столбцов. Полосы от плиток — цветные, и в этой мерке они
видны там, где в яркостной тонут.

Запуск (из корня проекта):
    .venv\Scripts\python tools\experiments\vae_seam_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fooocus_qwen.logging_setup import use_utf8_console

use_utf8_console()

import argparse
import time

import numpy as np
import torch
from PIL import Image

from fooocus_qwen import config
from fooocus_qwen.engine import vae_tiling

SOURCE = config.OUTPUT_DIR / "2026-09-22" / "19-35-06.png"


def chroma_lines(array: np.ndarray, stride: int) -> dict:
    """Ищет цветные вертикальные линии и проверяет, стоят ли они по решётке плиток."""
    chroma = array[:, :, 1] - 0.5 * (array[:, :, 0] + array[:, :, 2])
    column = chroma.mean(axis=0)
    window = 21
    trend = np.convolve(np.pad(column, window // 2, mode="edge"), np.ones(window) / window, mode="valid")
    residual = column - trend
    sigma = float(residual.std())
    peaks = [i for i in range(len(residual)) if abs(residual[i]) > 4 * sigma]
    # Линии, попавшие в окрестность границы плиток, — это швы, остальное содержимое.
    seams = [i for i in peaks if min((i + 1) % stride, stride - (i + 1) % stride) <= 20]
    return {
        "sigma": round(sigma, 4),
        "peak": round(float(np.abs(residual).max()), 3),
        "lines": len(peaks),
        "on_grid": len(seams),
        "positions": seams[:10],
    }


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="Швы тайлового декодера VAE")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    args = parser.parse_args()

    from diffusers import AutoencoderKLQwenImage21

    print("гружу VAE…", flush=True)
    vae = AutoencoderKLQwenImage21.from_pretrained(
        config.MODEL_DIR, subfolder="vae", torch_dtype=torch.bfloat16
    ).to("cuda")
    vae.eval()

    image = Image.open(args.source).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    opaque = np.ones_like(array[..., :1])
    x = torch.from_numpy(np.concatenate([array, opaque], axis=2))
    x = x.permute(2, 0, 1)[None, :, None].to("cuda", torch.bfloat16)

    print(f"кодирую {image.width}x{image.height} без тайлинга…", flush=True)
    vae.disable_tiling()
    latent = vae.encode(x).latent_dist.mode()

    results: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    peaks: dict[str, float] = {}

    def run(name: str) -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        decoded = vae.decode(latent, return_dict=False)[0][:, :, 0]
        torch.cuda.synchronize()
        timings[name] = time.perf_counter() - started
        peaks[name] = torch.cuda.max_memory_allocated() / 2**30
        out = decoded.float().clamp(-1, 1)[0].permute(1, 2, 0).cpu().numpy()[..., :3]
        results[name] = (out + 1.0) * 127.5

    print("декодирую целиком (эталон)…", flush=True)
    run("целиком")

    print("декодирую штатным тайлингом diffusers…", flush=True)
    vae.enable_tiling(tile_sample_min_height=args.tile, tile_sample_min_width=args.tile)
    vae.tile_sample_stride_height = args.stride
    vae.tile_sample_stride_width = args.stride
    run("diffusers")

    print("декодирую здешним тайлингом…", flush=True)
    vae_tiling.install(vae)
    run("здешний")

    reference = results["целиком"]
    print(f"\n{'вариант':12} {'расхожд. с целым':>17} {'макс.':>7} {'швов':>5} {'линий':>6} {'с':>6} {'ГиБ':>6}")
    for name in ("целиком", "diffusers", "здешний"):
        delta = np.abs(results[name] - reference)
        stats = chroma_lines(results[name], args.stride)
        print(
            f"{name:12} {delta.mean():17.4f} {delta.max():7.2f} "
            f"{stats['on_grid']:5d} {stats['lines']:6d} {timings[name]:6.2f} {peaks[name]:6.2f}"
        )
        if stats["on_grid"]:
            print(f"             швы на x = {stats['positions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

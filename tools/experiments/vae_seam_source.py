r"""Опыт: полосы рождаются в декодере или приходят уже в латенте?

Прошлый заход сравнивал варианты декодирования латента, полученного
**кодированием кадра, в котором полосы уже были**. Такой латент несёт их как
содержимое, и любая мерка находит их во всех вариантах одинаково — опыт в
принципе не мог отличить источник от переносчика.

Здесь латент считается денойзингом, как при настоящей генерации: тем же
сидом, промтом и размером, что у кадра заказчика. Дальше он декодируется
тремя путями, и каждый кадр меряется одной меркой. Если линии есть и при
цельном декодировании — плитки ни при чём, и искать надо выше.

Латент сохраняется на диск: он стоит полторы минуты, а нужен будет ещё.

Запуск (из корня проекта):
    .venv\Scripts\python tools\experiments\vae_seam_source.py
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

from fooocus_qwen import config, logging_setup
from fooocus_qwen.engine import loader, vae_tiling

OUT = config.LOG_DIR / "vae-seam"

# Кадр заказчика 19-35-06.png: гладкий, ослепительно светлый — худший случай
# для швов, на нём они и видны.
SEED = 1255241689
STEPS = 28
WIDTH, HEIGHT = 1280, 1888
RESOLUTION = 1536
PROMPT = (
    "The image is a vertical dreamlike digital illustration of a surreal landscape, the "
    "background and its palette dominated by blinding white light and soft pastel hues. The "
    "sky occupies the upper half of the frame, rendered in a gradient that shifts from pale "
    "cream near the horizon to a washed-out, almost invisible white at the top edge, "
    "suggesting an intense, sourceless sun. In the center, a large, diffuse orb of pure white "
    "light hovers, its edges bleeding into the surrounding atmosphere, creating a halo effect "
    "that obscures any distinct features. Below this celestial element, the ground appears as "
    "a flat, reflective surface, likely water or polished stone, mirroring the sky in muted "
    "tones of lavender and pale blue."
)


def chroma_lines(array: np.ndarray, stride: int, axis: int = 0) -> dict:
    """Цветные линии и их положение относительно решётки плиток.

    ``axis=0`` — вертикальные линии (профиль по столбцам), ``axis=1`` —
    горизонтальные. Мерить надо обе оси: раскладка плиток по ширине и по
    высоте независима, и огрызок в конце одной из них даёт полосу только
    вдоль неё.
    """
    chroma = array[:, :, 1] - 0.5 * (array[:, :, 0] + array[:, :, 2])
    column = chroma.mean(axis=axis)
    window = 21
    trend = np.convolve(np.pad(column, window // 2, mode="edge"), np.ones(window) / window, mode="valid")
    residual = column - trend
    sigma = float(residual.std())
    peaks = [i for i in range(len(residual)) if abs(residual[i]) > 4 * sigma]
    on_grid = [i for i in peaks if min((i + 1) % stride, stride - (i + 1) % stride) <= 20]
    return {"sigma": round(sigma, 4), "lines": len(peaks), "on_grid": len(on_grid), "where": on_grid[:8]}


def latent_periodicity(latent: torch.Tensor, stride_latent: int) -> str:
    """Есть ли в самом латенте периодичность с шагом плиток."""
    array = latent.float()[0].mean(dim=0)[0].cpu().numpy()
    column = array.mean(axis=0)
    residual = column - np.convolve(np.pad(column, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    sigma = float(residual.std())
    marks = [i for i in range(len(residual)) if abs(residual[i]) > 3 * sigma]
    on_grid = [i for i in marks if (i + 1) % stride_latent <= 1 or (i + 1) % stride_latent >= stride_latent - 1]
    return f"СКО {sigma:.4f}, выбросов {len(marks)}, из них на решётке плиток {len(on_grid)}"


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description="Источник полос: декодер или латент")
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--latent", type=Path, default=OUT / "latent.pt")
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    pipe, residency, _cache = loader.load(config.MODEL_DIR, pin_memory=False)

    if args.latent.is_file():
        print(f"беру готовый латент {args.latent}", flush=True)
        latents = torch.load(args.latent).to(pipe._execution_device)
    else:
        print(f"считаю латент {WIDTH}x{HEIGHT}, {STEPS} шагов…", flush=True)
        latents = pipe(
            prompt=PROMPT,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            output_resolution=RESOLUTION,
            output_type="latent",
            generator=torch.Generator(device=pipe._execution_device).manual_seed(SEED),
        ).images
        torch.save(latents.cpu(), args.latent)

    unpacked = pipe._unpack_latents(latents, HEIGHT, WIDTH, pipe.vae_scale_factor).to(pipe.vae.dtype)
    shape = (1, pipe.vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(shape).to(unpacked.device, unpacked.dtype)
    std = torch.tensor(pipe.vae.config.latents_std).view(shape).to(unpacked.device, unpacked.dtype)
    z = unpacked * std + mean

    ratio = pipe.vae.spatial_compression_ratio
    print("\nсам латент:", latent_periodicity(z, args.stride // ratio))

    # Трансформер на время декодирования не нужен, а место нужно: цельное
    # декодирование просит около шестнадцати гигабайт.
    residency._transformer.to_host()  # опыт лезет во внутренности намеренно

    results = {}
    for name in ("целиком", "diffusers", "здешний"):
        if name == "целиком":
            pipe.vae.disable_tiling()
        else:
            pipe.vae.enable_tiling(tile_sample_min_height=args.tile, tile_sample_min_width=args.tile)
            pipe.vae.tile_sample_stride_height = args.stride
            pipe.vae.tile_sample_stride_width = args.stride
            if name == "здешний":
                vae_tiling.install(pipe.vae)
            else:
                # Загрузчик ставит здешний декодер сразу при загрузке: без
                # этой строки «штатный» вариант был бы им же — ровно та
                # ошибка, на которой уже один раз сгорело это исследование.
                vae_tiling.uninstall(pipe.vae)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        decoded = pipe.vae.decode(z, return_dict=False)[0][:, :, 0]
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() / 2**30

        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        image.save(OUT / f"{name}.png")
        results[name] = (np.asarray(image.convert("RGB"), dtype=np.float32), seconds, peak)

    reference = results["целиком"][0]
    print(f"\n{'вариант':12} {'верт.':>6} {'гориз.':>7} {'макс. откл.':>12} {'расх. с целым':>14} {'с':>6} {'ГиБ':>6}")
    for name, (array, seconds, peak) in results.items():
        vertical = chroma_lines(array, args.stride, axis=0)
        horizontal = chroma_lines(array, args.stride, axis=1)
        delta = float(np.abs(array - reference).mean())
        worst = float(np.abs(array - reference).max())
        print(
            f"{name:12} {vertical['lines']:6d} {horizontal['lines']:7d} {worst:12.2f} "
            f"{delta:14.4f} {seconds:6.2f} {peak:6.2f}"
        )
        if vertical["where"]:
            print(f"             вертикальные x = {vertical['where']}")
        if horizontal["where"]:
            print(f"             горизонтальные y = {horizontal['where']}")
    print(f"\nкадры сохранены в {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

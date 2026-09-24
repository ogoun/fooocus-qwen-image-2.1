r"""Опыт: SageAttention и INT8-веса трансформера — скорость, видеопамять, качество.

Профиль шага при 1536² (`docs/research/2026-09-22-profil-i-optimizacii.md`):
линейные слои 1.78 с, внимание 0.97 с, прочее 0.28 с; внимание идёт по
memory-efficient ядру на 40 TFLOPS против 71 у GEMM. Тогда оба быстрых
механизма упирались в Triton, которого на Windows не было. Теперь есть
``triton-windows`` и сборки SageAttention 2 под Windows.

Микротест на размерах трансформера (1536², 24 головы по 128) показал:

* SageAttention 2 — внимание в 2.3 раза быстрее SDPA, косинус с эталоном 0.99993;
* ``torch._int_mm`` — INT8 GEMM **не быстрее** bf16 на RTX 3090 (59–65 TOPS
  против 71 TFLOPS). Значит INT8 здесь — рычаг видеопамяти, а не скорости.

Опыт проверяет это на полной генерации. Варианты:

  ``base``      — как в проекте сейчас;
  ``sage``      — внимание SageAttention (``set_attention_backend("sage")``);
  ``sage+int8`` — плюс веса трансформера в INT8 (torchao, только веса).

Меры: секунды на шаг, пик видеопамяти, LPIPS к картинке ``base`` с тем же
сидом (0 — неотличимо; ориентир Unsloth для их INT8 — 0.064 в среднем).
Каждый случай — один прогревочный прогон и один мерный.

Картинки ложатся в ``logs/sage-int8/`` для осмотра глазами.

Запуск:
    .venv\Scripts\python tools\experiments\sage_int8.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Веса AlexNet для LPIPS — в каталог проекта, а не в профиль пользователя.
os.environ.setdefault("TORCH_HOME", str(ROOT / "tmp" / "torch_home"))

from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402

use_utf8_console()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from fooocus_qwen import config, logging_setup  # noqa: E402
from fooocus_qwen.engine import loader  # noqa: E402

OUT = config.LOG_DIR / "sage-int8"

CASES = [
    ("t2i-portrait", 'a close-up portrait of an elderly fisherman with a knitted cap, harbour at dawn, '
                     'a wooden sign behind him that reads "NORTH PIER"', None),
    ("t2i-still", "a still life of lemons, a glass carafe and a linen napkin on a marble table, "
                  "soft window light", None),
    ("edit", "make the mug glossy cobalt blue", ROOT / "tmp" / "docs_screenshots" / "showcase-mug.png"),
]


def allow_sage_with_full_masks() -> None:
    """SageAttention не принимает ``attn_mask``, а трансформер на каждом шаге
    передаёт маску «настоящих» ключей — против дополнения промта справа. При
    одном промте без дополнения она целиком из True и ничего не отсекает:
    такая маска опускается. Маска с отсечением уходит штатному ядру."""
    from diffusers.models.transformers import transformer_qwenimage21 as module

    original = module.dispatch_attention_fn

    def dispatch(query, key, value, attn_mask=None, backend=None, **kwargs):
        # backend=None значит «выбранный глобально», то есть тоже sage: так
        # вызывает внимание первый проход по промту — с причинной маской.
        if attn_mask is not None:
            if bool(attn_mask.all()):
                attn_mask = None
            else:
                backend = "native"
        return original(query, key, value, attn_mask=attn_mask, backend=backend, **kwargs)

    module.dispatch_attention_fn = dispatch


def lpips_model():
    import lpips

    return lpips.LPIPS(net="alex", verbose=False).cuda().eval()


def to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1)[None].cuda()


def main() -> int:
    parser = argparse.ArgumentParser(description="SageAttention и INT8: скорость, память, качество")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--resolution", type=int, default=1536)
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)
    pipe, _residency, _cache = loader.load(config.MODEL_DIR)
    device = pipe._execution_device
    metric = lpips_model()
    allow_sage_with_full_masks()
    done = json.loads((OUT / "scores.json").read_text(encoding="utf-8")) if (OUT / "scores.json").exists() else {}

    def run(prompt: str, source: Path | None) -> tuple[Image.Image, float]:
        condition = [Image.open(source).convert("RGB")] if source else None
        torch.cuda.synchronize()
        started = time.perf_counter()
        image = pipe(
            prompt=prompt,
            image=condition,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.steps,
            output_resolution=1024 if condition else args.resolution,
            generator=torch.Generator(device=device).manual_seed(11),
        ).images[0]
        torch.cuda.synchronize()
        return image, time.perf_counter() - started

    def set_sage(on: bool) -> None:
        pipe.transformer.set_attention_backend("sage" if on else "native")

    def int8(_on: bool) -> None:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_

        before = torch.cuda.memory_allocated()
        quantize_(pipe.transformer, Int8WeightOnlyConfig())
        torch.cuda.empty_cache()
        print(f"  INT8: занято {before / 2**30:.2f} → {torch.cuda.memory_allocated() / 2**30:.2f} ГиБ")

    variants = [
        ("base", lambda: set_sage(False)),
        ("sage", lambda: set_sage(True)),
        ("sage+int8", lambda: (int8(True), set_sage(True))),  # последним: квантование необратимо
    ]
    # Уже измеренное не повторяется: базовые прогоны занимают по полторы минуты.
    rows: dict[str, dict] = dict(done)
    base_images: dict[str, Image.Image] = {
        case: Image.open(OUT / f"{case}-base.png") for case, _p, _s in CASES
        if f"{case} / base" in done and (OUT / f"{case}-base.png").exists()
    }
    for variant, apply in variants:
        if all(f"{case} / {variant}" in rows for case, _p, _s in CASES):
            continue
        apply()
        for case, prompt, source in CASES:
            run(prompt, source)  # прогрев: компиляция ядер, кэш промтов
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            image, spent = run(prompt, source)
            image.save(OUT / f"{case}-{variant}.png")
            row = {
                "секунд": round(spent, 2),
                "пик_гиб": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
            if variant == "base":
                base_images[case] = image
            else:
                with torch.no_grad():
                    row["lpips"] = round(float(metric(to_tensor(image), to_tensor(base_images[case]))), 4)
            rows[f"{case} / {variant}"] = row
            print(f"{case} / {variant}: {row}", flush=True)
            (OUT / "scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== сводка ===")
    print(f"{'случай / вариант':>26} | {'сек':>7} | {'выигрыш':>8} | {'пик ГиБ':>7} | {'LPIPS':>6}")
    for key, row in rows.items():
        case = key.split(" / ")[0]
        base = rows[f"{case} / base"]["секунд"]
        gain = 100 * (base - row["секунд"]) / base
        print(f"{key:>26} | {row['секунд']:>7} | {gain:>7.1f}% | {row['пик_гиб']:>7} | {row.get('lpips', '—'):>6}")
    print(f"\njson: {OUT / 'scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

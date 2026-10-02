r"""Опыт: пользовательские LoRA на Qwen-Image-2.1 — форматы, эффект и цена.

Повод (2026-10-02) — поддержка LoRA, как в Fooocus и InvokeAI. Вопросы:

1. подключаются ли реальные LoRA 2.1 всех встречающихся форматов (peft,
   DiffSynth ``.default.``, ai-toolkit/ComfyUI со слитым ``gate_up``) и дают
   ли эффект — средняя разница пикселей с тем же сидом и промтом без LoRA;
2. что встроенный в diffusers путь (``pipe.load_lora_weights`` с файлом как
   есть) делает со слитым ``gate_up`` — сравнение с нашим переводом
   (``engine/lora.py``);
3. сколько стоит неслитая LoRA на шаге: Turbo с одной, двумя LoRA и без;
   отдельно — подключение (перестановка трансформера).

LoRA — в ``tmp/loras_test/`` (скачать: см. docs/research/2026-10-02-lora.md),
картинки — в ``tmp/lora_user/``.

Запуск (нужны веса модели и Turbo):
    .venv\Scripts\python tools\experiments\lora_user.py --runs 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import lora, presets  # noqa: E402
from fooocus_qwen.engine.generator import GenerationRequest  # noqa: E402
from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402

LORAS = ROOT / "tmp" / "loras_test"
WORK = ROOT / "tmp" / "lora_user"
PROMPT = "zenlesszonezero, a girl with short black hair standing on a city street at night, neon signs, full body"


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(description="User LoRA on Qwen-Image-2.1: formats, effect, cost")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR = WORK / "outputs"

    from fooocus_qwen.ui.state import Studio

    studio = Studio(config.AppConfig(lang="en", preload=False))
    generator = studio.generator
    pipe = generator.pipe

    def frame(tag: str, choices, seed: int = 11) -> tuple[np.ndarray, float]:
        resolved, problems = lora.resolve(LORAS, [lora.LoraChoice(name, weight) for name, weight in choices])
        if problems:
            raise SystemExit(f"LoRA problems: {problems}")
        started = time.perf_counter()
        image = generator.generate(GenerationRequest(
            prompt=PROMPT, preset=presets.get("Turbo"), aspect="1:1", seed=seed, loras=tuple(resolved),
        ))[0].image
        seconds = time.perf_counter() - started
        image.save(WORK / f"{tag}.png")
        return np.asarray(image.convert("RGB"), dtype=np.float32), seconds

    # 1. Эффект каждого формата.
    base, _ = frame("base", [])
    print("\nformat effect (mean |Δ| of 0..255 against the same frame without LoRA):")
    for name in lora.list_names(LORAS):
        info = lora.inspect(lora.path_for(LORAS, name), name)
        if not info.usable:
            print(f"  {name:20} refused: {info.problem} {info.family}")
            continue
        image, _ = frame(f"fmt_{name}", [(name, 1.0)])
        print(f"  {name:20} rank {info.rank:>3}, {info.layers:>3} layers: {float(np.abs(image - base).mean()):5.1f}")

    # 2. Встроенный путь diffusers на слитом gate_up.
    ours, _ = frame("zzz_ours", [("zzz-style", 1.0)])
    generator.generate(GenerationRequest(prompt=PROMPT, preset=presets.get("Turbo"), seed=11))  # выгрузить свои
    stock_name = "stock_zzz"

    def attach(_transformer):
        pipe.load_lora_weights(str(LORAS / "zzz-style.safetensors"), adapter_name=stock_name)

    generator._residency.restage_transformer(attach)
    loaded = getattr(pipe.transformer, "peft_config", {})[stock_name]
    targets = sorted(loaded.target_modules) if hasattr(loaded, "target_modules") else []
    pipe.enable_lora()
    pipe.set_adapters(["turbo", stock_name], [1.0, 1.0])
    generator._turbo.activate(True)
    import torch

    stock = generator._pipe(
        prompt=PROMPT, image=None, height=1024, width=1024, num_inference_steps=6, sigmas=[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25],
        true_cfg_scale=1.0, generator=torch.Generator(device="cuda").manual_seed(11), output_resolution=1024,
    ).images[0]
    stock.save(WORK / "zzz_stock.png")
    stock_array = np.asarray(stock.convert("RGB"), dtype=np.float32)
    mlp = [name for name in targets if "mlp" in name]
    print(f"\nstock diffusers loader on the fused gate_up file: target modules {targets[:8]}… mlp ones: {mlp}")
    print(f"  ours vs stock: {float(np.abs(ours - stock_array).mean()):.1f}; stock vs base: {float(np.abs(stock_array - base).mean()):.1f}")
    generator._residency.restage_transformer(lambda _t: pipe.delete_adapters([stock_name]))

    # 3. Цена на шаге: подключение отдельно (первый кадр), затем установившиеся.
    print("\ncost (Turbo 1024², seconds per frame):")
    for label, choices in (("no LoRA", []), ("1 LoRA", [("zzz-style", 1.0)]),
                           ("2 LoRA", [("zzz-style", 0.8), ("photo-aesthetics", 0.5)])):
        _, first = frame(f"cost_{label}", choices, seed=3)
        times = [frame(f"cost_{label}", choices, seed=3)[1] for _ in range(args.runs)]
        print(f"  {label:8} first {first:5.1f} s (incl. attaching), then median {statistics.median(times):5.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

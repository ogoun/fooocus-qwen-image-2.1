r"""Опыт: дистиллированный Qwen-Image-2.1-viggle-turbo — 6 шагов вместо 16–40.

Viggle выпустили LoRA-адаптер к нашей же модели, обученный дистилляцией
(DMD): генерация и правка с 1–3 референсами за 6 проходов трансформера без
CFG. Правила автора: ``num_inference_steps=6``,
``sigmas=[1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]``, планировщик с
``shift_terminal=None`` (штатное 0.02 портит последний шаг), LoRA не
сливается с весами. Правка по маске и RGBA автором не проверялись — для
нас это главный вопрос: маска у нас — второе условное изображение.

Опыт идёт через ``Generator`` — ту же дорогу, что интерфейс, включая
вклейку по маске, — и сравнивает на одинаковых сидах:

  * генерацию с надписью (LowQuality, 16 шагов, против turbo);
  * генерацию 1536² (MiddleQuality, 28 шагов, против turbo);
  * правку по маске (LowQuality против turbo): держится ли вклейка и что внутри;
  * правку без маски с одним исходником.

Мерится время генерации и пик видеопамяти; качество — глазами по парам
«база | turbo» в ``logs/turbo/`` (LPIPS тут не мера: дистиллят и не
обязан повторять учителя пиксель в пиксель).

Адаптер и конфигурация планировщика ожидаются в ``tmp/viggle-turbo/``
(скачиваются с Hugging Face: Viggle/Qwen-Image-2.1-viggle-turbo).

Запуск:
    .venv\Scripts\python tools\experiments\turbo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402

use_utf8_console()

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from fooocus_qwen import config, logging_setup  # noqa: E402
from fooocus_qwen.engine import loader, presets  # noqa: E402
from fooocus_qwen.engine.generator import MASK_MASK, GenerationRequest, Generator  # noqa: E402
from fooocus_qwen.engine.residency import StagedModule  # noqa: E402
from fooocus_qwen.imaging import aspect  # noqa: E402
from fooocus_qwen.prompting.styles import load_styles  # noqa: E402

OUT = config.LOG_DIR / "turbo"
TURBO = ROOT / "tmp" / "viggle-turbo"
LORA = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
SIGMAS = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25]
MUG = ROOT / "tmp" / "docs_screenshots" / "showcase-mug.png"


class TurboPipe:
    """Обёртка пайплайна для генератора: 6 шагов, свои sigmas, без CFG.

    Всё, кроме вызова, передаётся настоящему пайплайну как есть.
    """

    def __init__(self, pipe) -> None:
        self._pipe = pipe

    def __getattr__(self, name):
        return getattr(self._pipe, name)

    def __call__(self, *args, **kwargs):
        kwargs["num_inference_steps"] = len(SIGMAS)
        kwargs["sigmas"] = SIGMAS
        kwargs["true_cfg_scale"] = 1.0
        kwargs.pop("negative_prompt", None)
        return self._pipe(*args, **kwargs)


class Turbo:
    """Включает и выключает turbo у генератора: адаптер, планировщик, вызов."""

    def __init__(self, engine, residency) -> None:
        self.engine = engine
        self.pipe = engine._pipe
        self.base_scheduler = self.pipe.scheduler
        self.turbo_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(TURBO, subfolder="scheduler")
        # Менеджер размещения переносит трансформер между хостом и картой по
        # списку параметров, снятому при загрузке; LoRA добавляет новые
        # (``…base_layer``, ``…lora_A``). Поэтому адаптер подключается к
        # трансформеру на хосте, а затем трансформер регистрируется заново.
        residency._transformer.to_host()
        self.pipe.load_lora_weights(str(TURBO), weight_name=LORA, adapter_name="turbo")
        residency._transformer = StagedModule(self.pipe.transformer, residency._device, residency._pin_memory)
        residency._transformer.to_device()
        self.on(False)

    def on(self, enabled: bool) -> None:
        if enabled:
            self.pipe.enable_lora()
            self.pipe.scheduler = self.turbo_scheduler
            self.engine._pipe = TurboPipe(self.pipe)
        else:
            self.pipe.disable_lora()
            self.pipe.scheduler = self.base_scheduler
            self.engine._pipe = self.pipe


def mask_for(size: tuple[int, int]) -> Image.Image:
    """Пустое место на столе справа от кружки — то же, что на снимке README."""
    mask = Image.new("L", size, 0)
    width, height = size
    ImageDraw.Draw(mask).rectangle(
        (int(width * 0.68), int(height * 0.64), int(width * 0.92), int(height * 0.86)), fill=255
    )
    return mask


def outside_changed(source: Image.Image, result: Image.Image, mask: Image.Image) -> float:
    """Доля пикселей вне маски, отличающихся от исходника (должна быть 0)."""
    a = np.asarray(source.convert("RGB"), dtype=np.int16)
    b = np.asarray(result.convert("RGB").resize(source.size), dtype=np.int16)
    outside = np.asarray(mask) == 0
    return float((np.abs(a - b).max(axis=2)[outside] > 0).mean())


def main() -> int:
    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)
    pipe, residency, cache = loader.load(config.MODEL_DIR)
    engine = Generator(pipe, residency, cache, load_styles(config.STYLES_DIR))
    turbo = Turbo(engine, residency)
    mug = Image.open(MUG).convert("RGBA")

    cases = [
        ("t2i-text", dict(prompt='a hand-painted wooden sign on a bakery wall that reads "FRESH BREAD DAILY", '
                                 "morning light, photorealistic", preset=presets.get("LowQuality"), seed=5)),
        ("t2i-1536", dict(prompt="an alpine lake at sunrise, a red canoe on the shore, mist over the water, "
                                 "pine forest, photorealistic", preset=presets.get("MiddleQuality"), seed=9)),
        ("edit-mask", dict(prompt="a small potted succulent in a terracotta pot", preset=presets.get("LowQuality"),
                           seed=3, source=mug, mask=mask_for(mug.size), mask_mode=MASK_MASK,
                           aspect=aspect.FOLLOW_REFERENCE)),
        ("edit-whole", dict(prompt="make the mug glossy cobalt blue", preset=presets.get("LowQuality"), seed=3,
                            source=mug, aspect=aspect.FOLLOW_REFERENCE)),
    ]
    rows: dict[str, dict] = {}
    for case, kwargs in cases:
        images = {}
        for variant in ("base", "turbo"):
            turbo.on(variant == "turbo")
            request = GenerationRequest(prompt_original=kwargs["prompt"], **kwargs)
            engine.generate(request)  # прогрев: кэш промтов, ядра
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            image = engine.generate(request)[0].image
            torch.cuda.synchronize()
            spent = time.perf_counter() - started
            image.save(OUT / f"{case}-{variant}.png")
            images[variant] = image
            row = {"секунд": round(spent, 2), "пик_гиб": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            if "mask" in kwargs:
                row["вне_маски_изменено"] = outside_changed(mug, image, kwargs["mask"])
            rows[f"{case} / {variant}"] = row
            print(f"{case} / {variant}: {row}", flush=True)
            (OUT / "scores.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        pair = [images["base"].convert("RGB"), images["turbo"].convert("RGB").resize(images["base"].size)]
        canvas = Image.new("RGB", (pair[0].width * 2 + 16, pair[0].height), "white")
        canvas.paste(pair[0], (0, 0))
        canvas.paste(pair[1], (pair[0].width + 16, 0))
        canvas.save(OUT / f"{case}-pair.jpg", quality=90)

    print("\n=== сводка ===")
    for case, _ in cases:
        base, fast = rows[f"{case} / base"], rows[f"{case} / turbo"]
        print(f"{case:>11}: база {base['секунд']:>6} с → turbo {fast['секунд']:>6} с "
              f"(x{base['секунд'] / fast['секунд']:.1f}); пик {base['пик_гиб']} → {fast['пик_гиб']} ГиБ"
              + (f"; вне маски изменено: {fast.get('вне_маски_изменено')}" if "вне_маски_изменено" in fast else ""))
    print(f"\nпары «база | turbo»: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

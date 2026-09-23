r"""Опыт: во что обходится правка, и почему она вешает карту.

Заказчик запустил правку и получил двадцать минут без единого шага
прогресса: карта забита под потолок, сорок гигабайт в оперативной памяти —
вытеснение, а не зависание.

Предположение. При правке исходник идёт **условным изображением**, и его
латенты попадают в ту же последовательность, что и целевые: на кадре 2048
это 16384 токена цели плюс столько же условия. Внимание квадратично, и
удвоение последовательности стоит вчетверо. Бенчмарк мерил генерацию без
условий (MaxQuality — 16.2 ГиБ) и референсы на среднем пресете (один
референс — 21.0 ГиБ), а путь правки не мерил ни разу.

Защита по памяти (``resolve_reference_scale``) смотрит только на число
референсов, и правку я из неё исключил намеренно — «у правки детальность
исходника решает качество». Опыт проверяет, чего это стоило.

Мерится пик памяти и время на шаг; шагов немного, потому что пик
приходится на первый.

**Один запуск — один режим.** Пул аллокатора, выросший на предыдущем
прогоне, не возвращается по ``empty_cache()`` целиком, и у потолка в 24 ГБ
этого хватает, чтобы свалить следующий режим в вытеснение: первая редакция
этого опыта мерила все пресеты подряд и дала правке на среднем пресете 60
секунд на шаг вместо настоящих. Список пресетов в одном запуске годится
для разведки, итоговое число — отдельным процессом.

Запуск (из корня проекта):
    .venv\Scripts\python tools\experiments\edit_cost.py
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
from dataclasses import replace as _replace

import torch
from PIL import Image

from fooocus_qwen import config, logging_setup
from fooocus_qwen.engine import loader, presets
from fooocus_qwen.engine.generator import (
    MASK_MASK,
    MASK_NONE,
    GenerationRequest,
    Generator,
)
from fooocus_qwen.imaging import aspect
from fooocus_qwen.prompting.styles import load_styles

OUT = config.LOG_DIR / "edit-cost"
PROMPT = "make the background a quiet autumn park"


def main() -> int:
    parser = argparse.ArgumentParser(description="Стоимость правки по пресетам")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--source", type=str, default=None,
                        help="путь к исходнику; по умолчанию кадр из дымового прогона")
    parser.add_argument("--presets", nargs="*", default=list(presets.NAMES))
    parser.add_argument("--budget-seconds", type=float, default=300.0)
    parser.add_argument("--mask", action="store_true",
                        help="правка по маске: условных изображений два, а не одно")
    parser.add_argument("--scales", type=int, nargs="*", default=None,
                        help="перебрать масштаб условных изображений вместо сравнения с генерацией")
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    source_path = Path(args.source) if args.source else config.LOG_DIR / "smoke" / "source.png"
    source = Image.open(source_path).convert("RGB")
    print(f"исходник {source_path.name} {source.size}\n", flush=True)

    pipe, residency, cache = loader.load(config.MODEL_DIR)
    engine = Generator(pipe, residency, cache, load_styles(config.STYLES_DIR))

    rows: dict[str, dict] = {}
    for name in args.presets:
        preset = presets.get(name)
        # Правка без маски: ровно то, что делал заказчик (маска была пуста).
        mask = None
        if args.mask:
            from PIL import ImageDraw

            mask = Image.new("L", source.size, 0)
            width, height = source.size
            ImageDraw.Draw(mask).ellipse(
                (width // 4, height // 4, width * 3 // 4, height * 3 // 4), fill=255
            )

        request = GenerationRequest(
            prompt=PROMPT,
            preset=preset,
            aspect=aspect.FOLLOW_REFERENCE,
            source=source,
            mask=mask,
            mask_mode=MASK_MASK if mask is not None else MASK_NONE,
            seed=7,
        )
        # Для сравнения — та же генерация без исходника вовсе.
        plain = GenerationRequest(prompt=PROMPT, preset=preset, seed=7)

        if args.scales:
            # Перебор масштаба условных изображений: кадр при этом обязан
            # остаться пресетным, ради чего рычаг и разделяли.
            pairs = [(f"масштаб {s}", _replace(request, reference_scale=s)) for s in args.scales]
        else:
            pairs = [("правка", request), ("без исходника", plain)]

        for kind, req in pairs:
            key = f"{name}/{kind}"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            try:
                produced = engine.generate(
                    _with_steps(req, args.steps),
                    progress=_watchdog(started, args.budget_seconds),
                )
            except (torch.cuda.OutOfMemoryError, TimeoutError) as error:
                torch.cuda.empty_cache()
                rows[key] = {
                    "ok": False,
                    "reason": type(error).__name__,
                    "detail": str(error)[:90],
                    "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                }
                print(f"{key}: {rows[key]}", flush=True)
                _save(rows)
                continue

            elapsed = time.perf_counter() - started
            image = produced[0].image if produced else None
            rows[key] = {
                "ok": image is not None,
                "seconds": round(elapsed, 1),
                "seconds_per_step": round(elapsed / args.steps, 1),
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "frame": f"{image.size[0]}x{image.size[1]}" if image else "—",
            }
            print(f"{key}: {rows[key]}", flush=True)
            _save(rows)

    print("\n=== сводка ===")
    print(f"{'режим':>28} | {'кадр':>11} | {'с/шаг':>6} | {'пик, ГиБ':>8}")
    for key, row in rows.items():
        if not row.get("ok"):
            print(f"{key:>28} | {row['reason']} | пик {row.get('peak_vram_gib')}")
            continue
        print(f"{key:>28} | {row['frame']:>11} | {row['seconds_per_step']:>6} | "
              f"{row['peak_vram_gib']:>8}")
    print(f"\nкадры: {OUT}")
    return 0


def _with_steps(request: GenerationRequest, steps: int) -> GenerationRequest:
    from dataclasses import replace

    from fooocus_qwen.engine.presets import QualityPreset

    short = QualityPreset(request.preset.name, request.preset.output_resolution, steps)
    return replace(request, preset=short)


def _watchdog(started: float, budget: float):
    """Прерывает ползущий режим: без него опыт сам зависнет на двадцать минут."""

    def report(_index: int, _step: int, _total: int) -> None:
        if time.perf_counter() - started > budget:
            raise TimeoutError(f"дольше {budget:.0f} с — режим непрактичен")

    return report


def _save(rows: dict) -> None:
    (OUT / "scores.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())

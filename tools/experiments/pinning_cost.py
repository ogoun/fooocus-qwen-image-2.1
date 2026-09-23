r"""Опыт: стоит ли закрепление памяти тех секунд, что оно отнимает у запуска.

В журнале заказчика видно: «Готовлю копии весов на хосте» — двадцать четыре
секунды из сорока девяти, то есть половина запуска. Это ``pin_memory()`` на
29.6 ГБ (трансформер 13.3 и энкодер 16.3): веса после ``from_pretrained``
уже лежат в оперативной памяти, и закрепление делает их **полную копию** в
странице, которую нельзя выгрузить.

Взамен закрепление ускоряет пересылку на карту: DMA идёт напрямую, без
промежуточного буфера. Пересылка случается при каждом промахе кэша
эмбеддингов — дважды на промах (туда и обратно, для обеих моделей).

Опыт считает обе стороны сделки: сколько секунд закрепление отнимает у
запуска и сколько возвращает на каждой перестановке. Точка безубыточности —
частное этих чисел, то есть число промахов кэша, после которого закрепление
окупается.

Запуск (из корня проекта):
    .venv\Scripts\python tools\experiments\pinning_cost.py --pin
    .venv\Scripts\python tools\experiments\pinning_cost.py --no-pin

Разными процессами: держать в одном два набора копий весов по 29.6 ГБ
негде.
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

import torch

from fooocus_qwen import config, logging_setup

OUT = config.LOG_DIR / "pinning"


def main() -> int:
    parser = argparse.ArgumentParser(description="Цена и польза закрепления памяти")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pin", dest="pin", action="store_true")
    group.add_argument("--no-pin", dest="pin", action="store_false")
    parser.add_argument("--swaps", type=int, default=3, help="сколько перестановок померить")
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    from diffusers import QwenImage21Pipeline  # noqa: F401 — прогрев импортов

    from fooocus_qwen.engine.pipeline import QwenImage21StudioPipeline, assert_contract
    from fooocus_qwen.engine.residency import ResidencyManager

    assert_contract()

    started = time.perf_counter()
    pipe = QwenImage21StudioPipeline.from_pretrained(str(config.MODEL_DIR), dtype=torch.bfloat16)
    read_seconds = time.perf_counter() - started

    residency = ResidencyManager(pipe, device="cuda", pin_memory=args.pin)
    started = time.perf_counter()
    residency.start()
    stage_seconds = time.perf_counter() - started

    # Перестановка: ровно то, что делает промах кэша эмбеддингов.
    swap_times = []
    for _ in range(args.swaps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with residency.text_encoder_resident():
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        swap_times.append(time.perf_counter() - started)

    row = {
        "закрепление": args.pin,
        "чтение_весов_с": round(read_seconds, 1),
        "подготовка_копий_с": round(stage_seconds, 1),
        "перестановка_с": round(min(swap_times), 2),
        "перестановки": [round(t, 2) for t in swap_times],
    }

    path = OUT / "scores.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    rows["с закреплением" if args.pin else "без закрепления"] = row
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{row}")

    if len(rows) == 2:
        pinned, plain = rows["с закреплением"], rows["без закрепления"]
        saved = plain["подготовка_копий_с"] - pinned["подготовка_копий_с"]  # < 0: закрепление дороже
        gain = plain["перестановка_с"] - pinned["перестановка_с"]
        print("\n=== сделка ===")
        print(f"запуск: {pinned['подготовка_копий_с']} с с закреплением против "
              f"{plain['подготовка_копий_с']} с без")
        print(f"перестановка: {pinned['перестановка_с']} с против {plain['перестановка_с']} с")
        if gain > 0:
            print(f"окупается после {abs(saved) / gain:.0f} промахов кэша")
        else:
            print("закрепление не ускоряет перестановку — окупаться нечему")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

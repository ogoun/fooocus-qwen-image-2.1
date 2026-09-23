r"""Проверка рычага: отвязать величину референсов от размера кадра.

В пайплайне ``output_resolution`` делает две разные вещи. Если ``height`` и
``width`` не заданы, он выводит из них размер кадра (строка 623:
``height = height or calculated_height``). Но условные изображения
масштабируются к площади ``output_resolution²`` **безусловно** (строки
655-661). Значит при явно заданном размере кадра этот параметр управляет
только величиной референсов.

Если так, то пять референсов, не помещающиеся в память при равном масштабе,
поместятся при уменьшенном, а кадр не потеряет ни пикселя. Скрипт это
проверяет: кадр держится 1536x1536, масштаб референсов меняется.

Один процесс — один масштаб, если нужны точные числа. Замер показал, что
порядок влияет на результат: масштаб 1024 после 512 и 768 в том же процессе
завис на первом же шаге, а в чистом процессе прошёл за 36 с на шаг. Пул
аллокатора, выросший на прошлых прогонах, не возвращается по
``empty_cache()`` полностью, и у потолка в 24 ГБ этого хватает, чтобы
свалиться в вытеснение. Список масштабов в одном запуске годится для грубой
разведки; итоговое число по режиму берётся отдельным запуском.

Порядок масштабов — по возрастанию, и это существенно. Первая версия шла по
убыванию и начинала с 1536 — случая, про который из бенчмарка уже известно,
что он деградирует: тринадцать минут у потолка памяти, и только потом очередь
доходила до режимов, ради которых всё затевалось. Дешёвые случаи идут первыми,
дорогой — последним, когда ответ уже получен.

Сторожей два, и это не избыточность. Внутренний проверяет бюджет на каждом
шаге шумоподавления и прерывает ползущий режим, отдав секунды на шаг. Но он
бессилен, когда не завершается сам шаг: при вытеснении в оперативную память
один шаг уходит за пределы бюджета целиком, управление в Python не
возвращается, и проверить нечего — ``pipe.interrupt`` пайплайн смотрит тоже
только между шагами. Поэтому есть и внешний: отдельный поток, который по
истечении бюджета дописывает строку «завис» и жёстко завершает процесс.
Промежуточный json пишется после каждого масштаба, поэтому уже измеренное
при таком завершении не теряется.

Запуск:
    .venv\Scripts\python tools\measure_reference_scale.py
    .venv\Scripts\python tools\measure_reference_scale.py --count 5 --scales 512 768 1024
"""

from __future__ import annotations

import sys
from pathlib import Path

# Инструменты запускают по пути (`python tools\x.py`), и тогда в sys.path
# попадает каталог скрипта, а не корень проекта: без этой строки любой из них
# падает с ModuleNotFoundError ещё до первой полезной работы.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fooocus_qwen.logging_setup import use_utf8_console

use_utf8_console()  # русская справка --help не должна падать на cp1252

import argparse
import json
import os
import threading
import time

import torch
from PIL import Image

from fooocus_qwen import config, logging_setup
from fooocus_qwen.engine import loader

FRAME = 1536
PROMPT = "arrange these objects into one quiet still life on a wooden table"

class Overrun(RuntimeError):
    """Режим не уложился в бюджет времени и прерван на полушаге."""

    def __init__(self, steps_done: int, elapsed: float) -> None:
        super().__init__(f"прервано на шаге {steps_done} через {elapsed:.0f} с")
        self.steps_done = steps_done
        self.elapsed = elapsed


def references(count: int) -> list[Image.Image]:
    """Разноцветные квадраты: важен объём токенов, а не содержание."""
    return [
        Image.new("RGB", (768, 768), ((47 * index) % 256, (90 + 30 * index) % 256, 160))
        for index in range(count)
    ]


def attempt(pipe, images: list[Image.Image], scale: int, steps: int, budget: float) -> dict:
    """Одна генерация при заданном масштабе референсов.

    Возвращает в том числе ``seconds_per_step`` — величина осмысленна и для
    прерванного прогона, а именно она и показывает, начался ли своп.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    marks: list[float] = []

    def watchdog(_pipe, index: int, _timestep, kwargs: dict) -> dict:
        marks.append(time.perf_counter() - started)
        if marks[-1] > budget:
            raise Overrun(index + 1, marks[-1])
        return kwargs

    def peak() -> float:
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)

    try:
        result = pipe(
            prompt=PROMPT,
            image=images,
            height=FRAME,
            width=FRAME,
            num_inference_steps=steps,  # мерим память и масштаб, а не качество
            output_resolution=scale,
            generator=torch.Generator(device=pipe._execution_device).manual_seed(7),
            callback_on_step_end=watchdog,
        )
    except Overrun as stop:
        pipe._current_timestep = None  # пайплайн не дошёл до своей уборки
        torch.cuda.empty_cache()
        return {
            "ok": False,
            "reason": "дольше бюджета",
            "seconds_per_step": round(stop.elapsed / max(stop.steps_done, 1), 1),
            "steps_done": stop.steps_done,
            "peak_vram_gib": peak(),
        }
    except torch.cuda.OutOfMemoryError as error:
        pipe._current_timestep = None
        torch.cuda.empty_cache()
        return {"ok": False, "reason": "нехватка видеопамяти", "detail": str(error)[:80]}

    elapsed = time.perf_counter() - started
    image = result.images[0]
    return {
        "ok": True,
        "seconds": round(elapsed, 1),
        "seconds_per_step": round(elapsed / steps, 1),
        "peak_vram_gib": peak(),
        "frame": f"{image.size[0]}x{image.size[1]}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Влияние output_resolution на память при референсах")
    parser.add_argument("--count", type=int, default=5, help="сколько референсов подавать")
    parser.add_argument("--scales", type=int, nargs="*", default=[512, 768, 1024, 1536],
                        help="по возрастанию: дешёвые режимы должны отвечать первыми")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--budget-seconds", type=float, default=240.0,
                        help="дольше — режим непрактичен, прерываем на текущем шаге")
    parser.add_argument("--fresh", action="store_true",
                        help="начать таблицу заново, не подхватывая прежние измерения")
    args = parser.parse_args()

    def save(rows: dict) -> None:
        (config.LOG_DIR / "reference-scale.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logging_setup.setup_logging(False)
    pipe, residency, _cache = loader.load(config.MODEL_DIR)
    print(f"кадр держим {FRAME}x{FRAME}, референсов {args.count}, шагов {args.steps}\n", flush=True)

    # Уже измеренное не теряется при запуске одного режима отдельно: числа по
    # каждому масштабу получают в отдельном процессе (см. докстроку), и без
    # слияния каждый такой запуск затирал бы всю накопленную таблицу.
    rows: dict[str, dict] = {}
    previous = config.LOG_DIR / "reference-scale.json"
    if not args.fresh and previous.exists():
        rows.update(json.loads(previous.read_text(encoding="utf-8")))

    for scale in args.scales:
        print(f"-- масштаб референсов {scale} --", flush=True)

        def stuck(_scale=scale) -> None:
            """Шаг не вернул управление: записать это и выйти.

            Мягко выйти нельзя — поток застрял в вызове CUDA, а не в Python.
            Данные при этом целы: файл дописывается после каждого масштаба.
            """
            rows[str(_scale)] = {
                "ok": False,
                "reason": "шаг не завершился в пределах бюджета",
                "budget_seconds": args.budget_seconds,
                "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
            save(rows)
            print(f"   завис на одном шаге дольше {args.budget_seconds:.0f} с — выходим",
                  flush=True)
            os._exit(3)

        guard = threading.Timer(args.budget_seconds * 1.5, stuck)
        guard.daemon = True
        guard.start()
        try:
            outcome = attempt(pipe, references(args.count), scale, args.steps, args.budget_seconds)
        finally:
            guard.cancel()
        rows[str(scale)] = outcome
        print(f"   {outcome}", flush=True)
        save(rows)
        if not outcome["ok"]:
            # Масштабы идут по возрастанию: если этот уже не тянет, больший
            # тем более не потянет, и мерить его — только жечь время.
            print("   дальше не растём, больший масштаб тем более не пройдёт", flush=True)
            break

    print("\n=== сводка ===")
    print(f"{'масштаб':>8} | {'кадр':>9} | {'с/шаг':>6} | {'пик, ГиБ':>8} | итог")
    for scale, row in sorted(rows.items(), key=lambda item: int(item[0])):
        frame = row.get("frame", "—")
        per_step = row.get("seconds_per_step", "—")
        peak = row.get("peak_vram_gib", "—")
        verdict = "ок" if row["ok"] else row["reason"]
        print(f"{scale:>8} | {frame:>9} | {per_step:>6} | {peak:>8} | {verdict}")
    print(f"\nперестановок энкодера: {int(residency.stats()['swaps'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

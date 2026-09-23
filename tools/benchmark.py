"""Измерение времени и памяти по пресетам качества и по числу референсов.

Оценки в спецификации были получены из пропускной способности карты и
оказались втрое-вчетверо оптимистичнее действительности. Этот скрипт заменяет
их измерениями.

Первый прогон каждого режима отбрасывается: он несёт разовую перестановку
текстового энкодера и прогрев ядер, а мерить надо установившееся состояние —
именно в нём пользователь проводит время, перебирая сиды на одном промте.

Запуск:
    .venv\\Scripts\\python tools\\benchmark.py
    .venv\\Scripts\\python tools\\benchmark.py --repeats 5 --references 0 1 5 10
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
import statistics
import time
from datetime import date

import torch
from PIL import Image

from fooocus_qwen import config, logging_setup
from fooocus_qwen.engine import loader, presets
from fooocus_qwen.engine.generator import GenerationRequest, Generator
from fooocus_qwen.prompting.styles import load_styles

PROMPT = (
    "a wooden desk with a brass lamp, an open notebook and a cup of tea, "
    "warm afternoon light through a window"
)
MULTI_PROMPT = "combine <image1> and <image2> into one quiet still life on a table"

def reference_images(count: int) -> tuple[Image.Image, ...]:
    """Синтетические референсы: важен их объём в токенах, а не содержание."""
    return tuple(
        Image.new("RGB", (768, 768), ((40 * index) % 256, 90, 160)) for index in range(count)
    )


def measure(engine: Generator, request: GenerationRequest, repeats: int) -> dict[str, object]:
    """Прогоняет запрос repeats+1 раз, отбрасывая прогревочный."""
    torch.cuda.reset_peak_memory_stats()
    durations: list[float] = []

    for attempt in range(repeats + 1):
        request.seed = 1000 + attempt
        started = time.perf_counter()
        produced = engine.generate(request)
        elapsed = time.perf_counter() - started
        if not produced:
            return {"error": "генерация не вернула изображений"}
        if attempt > 0:
            durations.append(elapsed)

    return {
        "seconds_median": round(statistics.median(durations), 2),
        "seconds_min": round(min(durations), 2),
        "seconds_max": round(max(durations), 2),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "size": f"{produced[0].parameters['width']}x{produced[0].parameters['height']}",
    }


def render(report: dict, card: str) -> str:
    lines = [
        "# Измерения производительности",
        "",
        f"Дата: {date.today().isoformat()}. Карта: {card}.",
        "",
        "Первый прогон каждого режима отброшен как прогревочный; приведена медиана",
        "установившихся прогонов — того режима, в котором пользователь перебирает сиды.",
        "",
        "## Пресеты качества",
        "",
        "| Пресет | Разрешение | Шагов | Медиана, с | Мин, с | Макс, с | Пик видеопамяти, ГиБ |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, values in report["presets"].items():
        preset = presets.get(name)
        if "error" in values:
            lines.append(f"| {name} | {preset.output_resolution} | {preset.num_inference_steps} | — | — | — | {values['error']} |")
            continue
        lines.append(
            f"| {name} | {values['size']} | {preset.num_inference_steps} | "
            f"{values['seconds_median']} | {values['seconds_min']} | "
            f"{values['seconds_max']} | {values['peak_vram_gib']} |"
        )

    lines += [
        "",
        "## Референсные изображения (пресет MiddleQuality)",
        "",
        "| Референсов | Медиана, с | Пик видеопамяти, ГиБ | Замечание |",
        "|---|---|---|---|",
    ]
    for count, values in report["references"].items():
        if "error" in values:
            lines.append(f"| {count} | — | — | {values['error']} |")
        else:
            lines.append(
                f"| {count} | {values['seconds_median']} | {values['peak_vram_gib']} | |"
            )

    residency = report["residency"]
    lines += [
        "",
        "## Кэш эмбеддингов и перестановки весов",
        "",
        f"- попаданий в кэш: {report['cache']['hits']}, промахов: {report['cache']['misses']}",
        f"- перестановок текстового энкодера: {int(residency['swaps'])}",
        f"- занято видеопамяти на конец прогона: {residency['allocated_gib']:.2f} ГиБ",
        "",
        "Перестановок должно быть заметно меньше, чем генераций: повторный прогон с тем же",
        "промтом обязан попадать в кэш и не поднимать энкодер.",
        "",
        f"## Загрузка модели\n\n- {report['load_seconds']:.1f} с до готовности.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Бенчмарк пресетов и числа референсов")
    parser.add_argument("--repeats", type=int, default=3, help="замеров после прогревочного")
    parser.add_argument("--references", type=int, nargs="*", default=[0, 1, 5, 10])
    parser.add_argument("--skip-presets", action="store_true", help="не перемерять пресеты")
    parser.add_argument("--carry", type=str, default=None,
                        help="json с уже измеренными пресетами, чтобы дописать к ним референсы")
    args = parser.parse_args()

    logging_setup.setup_logging(False)

    started = time.perf_counter()
    pipe, residency, cache = loader.load(config.MODEL_DIR)
    load_seconds = time.perf_counter() - started
    engine = Generator(pipe, residency, cache, load_styles(config.STYLES_DIR))

    report: dict = {"presets": {}, "references": {}, "load_seconds": load_seconds}

    if args.carry:
        carried = json.loads(Path(args.carry).read_text(encoding="utf-8"))
        report["presets"] = carried.get("presets", {})
        report["references"] = carried.get("references", {})
        report["load_seconds"] = carried.get("load_seconds", load_seconds)

    def flush() -> None:
        """Сохраняет промежуточный результат.

        Прогон идёт десятками минут, и обрыв на середине не должен уносить
        всё измеренное: один раз уже унёс.
        """
        report.setdefault("cache", {"hits": cache.hits, "misses": cache.misses})
        report["cache"] = {"hits": cache.hits, "misses": cache.misses}
        report["residency"] = residency.stats()
        (config.LOG_DIR / "benchmark.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (config.PROJECT_ROOT / "docs" / "BENCHMARK.md").write_text(
            render(report, torch.cuda.get_device_name(0)), encoding="utf-8"
        )

    for name in ([] if args.skip_presets else presets.NAMES):
        request = GenerationRequest(prompt=PROMPT, preset=presets.get(name))
        try:
            result = measure(engine, request, args.repeats)
        except torch.cuda.OutOfMemoryError:
            result = {"error": "нехватка видеопамяти"}
            torch.cuda.empty_cache()
        report["presets"][name] = result
        print(f"{name}: {result}", flush=True)
        flush()

    for count in args.references:
        request = GenerationRequest(
            prompt=PROMPT if count < 2 else MULTI_PROMPT,
            preset=presets.get("MiddleQuality"),
            references=reference_images(count),
        )
        try:
            result = measure(engine, request, 1)
        except torch.cuda.OutOfMemoryError:
            result = {"error": "нехватка видеопамяти"}
            torch.cuda.empty_cache()
        report["references"][str(count)] = result
        print(f"референсов {count}: {result}", flush=True)
        flush()

    report["cache"] = {"hits": cache.hits, "misses": cache.misses}
    report["residency"] = residency.stats()

    destination = config.PROJECT_ROOT / "docs" / "BENCHMARK.md"
    destination.write_text(render(report, torch.cuda.get_device_name(0)), encoding="utf-8")
    (config.LOG_DIR / "benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nОтчёт: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

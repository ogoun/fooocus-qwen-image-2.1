"""Дымовой прогон: сценарии, требующие настоящей генерации.

Скрипт грузит модель один раз и прогоняет по ней пункты чек-листа
``docs/SMOKE.md``, которые нельзя проверить без видеокарты. Численно
проверяемое проверяется здесь же и печатается вердиктом; изображения
складываются в ``logs/smoke`` для просмотра глазами — качество картинки
скрипт оценить не может, и притворяться, что может, было бы враньём.

Запуск:
    .venv\\Scripts\\python tools\\smoke.py
    .venv\\Scripts\\python tools\\smoke.py --only mask outpaint
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
import threading
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from fooocus_qwen import config, logging_setup
from fooocus_qwen.engine import loader, presets
from fooocus_qwen.engine.generator import (
    MASK_ANNOTATION,
    MASK_MASK,
    MASK_NONE,
    MASK_REGION,
    GenerationRequest,
    Generator,
    resolve_reference_scale,
)
from fooocus_qwen.imaging import aspect, masking, metadata, outpaint
from fooocus_qwen.prompting.styles import load_styles

OUT = config.LOG_DIR / "smoke"
RESULTS: list[tuple[str, str, str]] = []

def record(point: str, verdict: str, note: str = "") -> None:
    RESULTS.append((point, verdict, note))
    print(f"  [{verdict}] {point}" + (f" — {note}" if note else ""), flush=True)


def save(image: Image.Image, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    image.save(path)
    return path


def one(engine: Generator, **kwargs) -> Image.Image | None:
    """Одна генерация; возвращает изображение или None, если прервано."""
    produced = engine.generate(GenerationRequest(**kwargs))
    return produced[0].image if produced else None


# --- сценарии ---------------------------------------------------------------


def opaque_share(image: Image.Image, where: np.ndarray | None = None) -> float:
    """Доля непрозрачных пикселей — в процентах, в заданной области.

    Проверка появилась после того, как прогон засчитал заведомо испорченную
    дорисовку полей. Модель объявила всю новую площадь прозрачной, под альфой
    остался пурпур декодера, а скрипт смотрел только на размер кадра и
    рапортовал «ок». Качество картинки скрипт судить не может — а вот
    непрозрачность обязан: это не вкус, а число.
    """
    if image.mode != "RGBA":
        return 100.0
    alpha = np.asarray(image)[..., 3]
    selected = alpha if where is None else alpha[where]
    return float((selected >= 250).mean() * 100) if selected.size else 100.0


def scenario_presets(engine: Generator) -> None:
    print("\n== items 3-4: three presets and text legibility ==")
    prompt = 'a neon shop sign that reads "QWEN", rainy night, reflections on wet pavement'
    for name in presets.NAMES:
        started = time.perf_counter()
        image = one(engine, prompt=prompt, preset=presets.get(name), seed=42)
        elapsed = time.perf_counter() - started
        if image is None:
            record(f"preset {name}", "FAIL", "generation returned no image")
            continue
        path = save(image, f"preset-{name}")
        record(f"preset {name}", "ok", f"{image.size[0]}x{image.size[1]}, {elapsed:.1f} s -> {path.name}")


def scenario_transparent(engine: Generator) -> None:
    print("\n== item 5: transparent RGBA ==")
    prompt = (
        "This is an RGBA image with transparency. A cute cartoon dragon sticker. "
        "The image has alpha channel and the background is transparent."
    )
    image = one(engine, prompt=prompt, preset=presets.get("LowQuality"), seed=7)
    if image is None:
        record("transparent RGBA", "FAIL", "no image")
        return
    path = save(image, "transparent")
    array = np.asarray(image.convert("RGBA"))
    alpha = array[..., 3]
    transparent_share = float((alpha < 16).mean())
    verdict = "ok" if transparent_share > 0.05 else "FAIL"
    record(
        "transparent RGBA",
        verdict,
        f"mode {image.mode}, transparent pixels {transparent_share:.0%} -> {path.name}",
    )


def scenario_aspects(engine: Generator) -> None:
    print("\n== item 6: seven aspect ratios ==")
    for ratio in aspect.ASPECT_RATIOS:
        if ratio == aspect.FOLLOW_REFERENCE:
            continue
        expected = aspect.dimensions(ratio, presets.get("LowQuality").output_resolution)
        image = one(
            engine,
            prompt="a calm seascape at dawn",
            preset=presets.get("LowQuality"),
            aspect=ratio,
            seed=11,
        )
        if image is None:
            record(f"ratio {ratio}", "FAIL", "no image")
            continue
        verdict = "ok" if image.size == expected else "FAIL"
        record(f"ratio {ratio}", verdict, f"got {image.size}, expected {expected}")


def make_source(engine: Generator) -> Image.Image:
    """Исходный кадр для сценариев редактирования."""
    image = one(
        engine,
        prompt="a portrait photograph of a woman with long dark hair, plain grey studio background",
        preset=presets.get("LowQuality"),
        seed=5,
    )
    assert image is not None, "failed to generate the source frame"
    save(image, "source")
    return image


def scenario_edit_plain(engine: Generator, source: Image.Image) -> None:
    print("\n== item 7: prompt edit without a mask ==")
    image = one(
        engine,
        prompt="change the background to a sunlit autumn park",
        preset=presets.get("LowQuality"),
        aspect=aspect.FOLLOW_REFERENCE,
        source=source,
        mask_mode=MASK_NONE,
        seed=21,
    )
    if image is None:
        record("prompt edit", "FAIL", "no image")
        return
    path = save(image, "edit-plain")
    record("prompt edit", "ok", f"{image.size} -> {path.name} (check by eye)")


def scenario_edit_mask(engine: Generator, source: Image.Image) -> None:
    print("\n== item 8: masked edit and pixel preservation ==")
    width, height = source.size
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).ellipse(
        (int(width * 0.25), int(height * 0.05), int(width * 0.75), int(height * 0.45)), fill=255
    )
    save(mask.convert("RGB"), "edit-mask-input")

    image = one(
        engine,
        prompt="change the hair colour to platinum blonde",
        preset=presets.get("LowQuality"),
        aspect=aspect.FOLLOW_REFERENCE,
        source=source,
        mask=mask,
        mask_mode=MASK_MASK,
        keep_outside=True,
        seed=33,
    )
    if image is None:
        record("masked edit", "FAIL", "no image")
        return
    path = save(image, "edit-mask")

    refined = masking.refine(mask, grow=8, feather=12)
    outside = np.asarray(refined) == 0
    before = np.asarray(source.convert("RGBA"))
    after = np.asarray(image.convert("RGBA"))
    identical = bool(np.array_equal(after[outside], before[outside]))
    changed_share = float((after != before).any(axis=2).mean())
    record(
        "pixels outside the mask are byte-identical",
        "ok" if identical else "FAIL",
        f"changed {changed_share:.1%} of the frame -> {path.name}",
    )
    # Исходник непрозрачен, значит и правка обязана остаться непрозрачной.
    # Без этой проверки прозрачность, которую модель принимает за смысл маски,
    # прошла бы незамеченной — так и случилось с дорисовкой полей.
    inside = np.asarray(refined) > 0
    opaque = opaque_share(image, inside)
    record(
        "masked edit stayed opaque",
        "ok" if opaque >= 99.0 else "FAIL",
        f"opaque inside the mask {opaque:.1f} %",
    )


def scenario_annotation(engine: Generator, source: Image.Image) -> None:
    print("\n== item 9: annotation, two areas in two colours ==")
    width, height = source.size
    annotated = source.convert("RGBA").copy()
    draw = ImageDraw.Draw(annotated)
    draw.ellipse(
        (int(width * 0.28), int(height * 0.06), int(width * 0.72), int(height * 0.42)),
        outline=(255, 0, 0, 255),
        width=max(3, width // 120),
    )
    draw.ellipse(
        (int(width * 0.30), int(height * 0.62), int(width * 0.70), int(height * 0.95)),
        outline=(0, 0, 255, 255),
        width=max(3, width // 120),
    )
    save(annotated, "annotation-input")

    image = one(
        engine,
        prompt=(
            "change the hair in the red circle to bright red, "
            "and change the clothing in the blue circle to a green linen shirt"
        ),
        preset=presets.get("LowQuality"),
        aspect=aspect.FOLLOW_REFERENCE,
        source=annotated,
        mask_mode=MASK_ANNOTATION,
        seed=44,
    )
    if image is None:
        record("annotation", "FAIL", "no image")
        return
    path = save(image, "annotation")
    record("annotation", "ok", f"{image.size} -> {path.name} (check by eye: both areas, no circles)")


def scenario_region(engine: Generator, source: Image.Image) -> None:
    print("\n== item 10: precise region on a small detail ==")
    width, height = source.size
    # Область и просьба обязаны совпадать. В первой редакции маска лежала на
    # лбу (y от 0.22 до 0.34), а промт просил серьги — уши совсем в другом
    # месте, модель разумно не сделала ничего заметного, и пункт «проходил»,
    # ничего по существу не проверив. Теперь пятно на лбу и просьба про
    # украшение на лбу.
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).ellipse(
        (int(width * 0.45), int(height * 0.20), int(width * 0.55), int(height * 0.27)), fill=255
    )
    image = one(
        engine,
        prompt="a small round golden bindi jewel on her forehead",
        preset=presets.get("LowQuality"),
        aspect=aspect.FOLLOW_REFERENCE,
        source=source,
        mask=mask,
        mask_mode=MASK_REGION,
        keep_outside=True,
        seed=55,
    )
    if image is None:
        record("precise region", "FAIL", "no image")
        return
    path = save(image, "region")
    # Совпадения размера мало: при неудачно поставленной маске кадр вернётся
    # того же размера и байт в байт исходным, и пункт «пройдёт», не проверив
    # ничего. Правка обязана произойти внутри области и не выйти за неё.
    refined = masking.refine(mask, grow=8, feather=12)
    inside = np.asarray(refined) > 0
    before = np.asarray(source.convert("RGB")).astype(np.float32)
    after = np.asarray(image.convert("RGB")).astype(np.float32)
    delta = np.abs(before - after).mean(axis=2)
    changed_inside = float(delta[inside].mean())
    changed_outside = float(delta[~inside].max())
    good = image.size == source.size and changed_inside > 1.0 and changed_outside == 0.0
    record(
        "precise region",
        "ok" if good else "FAIL",
        f"size {image.size}, mean change inside {changed_inside:.1f}, "
        f"max outside {changed_outside:.1f} -> {path.name}",
    )


def scenario_outpaint(engine: Generator, source: Image.Image) -> None:
    """Расширение кадра одной кнопкой: серый холст, лора outpaint, вклейка оригинала.

    Мера — серое, оставшееся незаполненным в новой площади, и то, что кадр
    вырос до холста; оригинал вклеивается точно (``outpaint.stitch``).
    """
    print("\n== item 11: outpaint ==")
    from fooocus_qwen.engine import fetch, lora

    fetch.ensure_outpaint(config.OUTPAINT_DIR)
    path_lora = config.OUTPAINT_DIR / fetch.OUTPAINT_FILE
    outpaint_lora = lora.ResolvedLora(path_lora.stem, 1.0, path_lora, lora.short_hash(path_lora))
    preset = presets.get("LowQuality")
    for sides, tag in ((["right"], "right"), (["top"], "top"), (["left", "bottom"], "left-bottom")):
        plan = outpaint.plan(source.size, sides, 0.35)
        job = outpaint.prepare(source, plan, outpaint.generation_area(preset.output_resolution))
        drawn = one(
            engine, prompt=outpaint.prompt(), preset=preset, aspect=aspect.FOLLOW_REFERENCE,
            source=job.canvas, size=job.size, reference_scale=job.resolution, seed=66, loras=(outpaint_lora,),
        )
        if drawn is None:
            record(f"outpaint {tag}", "FAIL", "no image")
            continue
        image = outpaint.stitch(source, drawn, job)
        path = save(image, f"outpaint-{tag}")
        left, top, right, bottom = plan.paste_box
        array = np.asarray(image.convert("RGB")).astype(np.int16)
        new_area = np.ones(array.shape[:2], dtype=bool)
        new_area[top:bottom, left:right] = False
        gray = float((np.abs(array[new_area] - 128).max(axis=1) < 4).mean() * 100)
        good = image.size == plan.canvas_size and gray < 5.0
        record(
            f"outpaint {tag}",
            "ok" if good else "FAIL",
            f"{source.size} -> {image.size}, gray left unfilled {gray:.1f} % -> {path.name}",
        )

def scenario_references(engine: Generator) -> None:
    print("\n== items 12-13: ten references and one ==")
    palette = [(200, 60, 60), (60, 160, 90), (70, 90, 200), (210, 170, 60), (150, 80, 180),
               (80, 180, 190), (230, 120, 70), (110, 110, 110), (180, 60, 140), (60, 200, 130)]
    refs = tuple(Image.new("RGB", (768, 768), colour) for colour in palette)

    # Пресет средний намеренно: именно на нём десять референсов и были
    # неподъёмны до появления раздельного масштаба условных изображений.
    # Теперь автоматика обязана урезать его до 512 и сделать режим рабочим —
    # это и проверяется, вместе с самой работоспособностью десяти референсов.
    middle = presets.get("MiddleQuality")
    request = GenerationRequest(
        prompt="a tidy shelf holding ten coloured boxes",
        preset=middle,
        references=refs,
        seed=77,
    )
    chosen = resolve_reference_scale(request)
    record(
        "reference scale reduced automatically",
        "ok" if chosen < middle.output_resolution else "FAIL",
        f"{middle.output_resolution} -> {chosen} with ten references",
    )

    try:
        started = time.perf_counter()
        produced = engine.generate(request)
        image = produced[0].image if produced else None
        elapsed = time.perf_counter() - started
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        record("ten references", "FAIL", "out of video memory")
        image = None
    if image is not None:
        path = save(image, "references-10")
        # Размер кадра обязан остаться пресетным: урезается масштаб условных
        # изображений, а не кадр. Ради этого рычаг и разделяли.
        width, height = aspect.dimensions("1:1", middle.output_resolution)
        good = image.size == (width, height)
        record(
            "ten references",
            "ok" if good else "FAIL",
            f"{image.size} (preset frame {width}x{height}), {elapsed:.1f} s -> {path.name}",
        )

    single = one(
        engine,
        prompt="place this colour swatch on a wooden table",
        preset=presets.get("LowQuality"),
        references=(refs[0],),
        seed=88,
    )
    if single is None:
        record("one reference", "FAIL", "no image")
    else:
        path = save(single, "references-1")
        record("one reference", "ok", f"{single.size} -> {path.name}")


def scenario_interrupt(engine: Generator) -> None:
    print("\n== item 18: generation interrupt ==")
    stopper = threading.Timer(6.0, engine.interrupt)
    stopper.start()
    started = time.perf_counter()
    produced = engine.generate(
        GenerationRequest(
            prompt="an intricate cathedral interior, many columns",
            preset=presets.get("MaxQuality"),
            image_number=3,
            seed=99,
        )
    )
    elapsed = time.perf_counter() - started
    stopper.cancel()
    record(
        "interrupt",
        "ok" if not produced else "FAIL",
        f"images returned: {len(produced)} in {elapsed:.1f} s (an empty list was expected)",
    )


def scenario_swaps(engine: Generator, residency) -> None:
    print("\n== item 20: repeating a prompt does not load the encoder ==")
    request = GenerationRequest(
        prompt="a single red apple on a white plate", preset=presets.get("LowQuality"), seed=1
    )
    engine.generate(request)
    first = residency.stats()["swaps"]
    request.seed = 2
    engine.generate(request)
    second = residency.stats()["swaps"]
    record(
        "prompt repeat without a swap",
        "ok" if second == first else "FAIL",
        f"swaps {first:.0f} -> {second:.0f}",
    )


def scenario_metadata(engine: Generator) -> None:
    print("\n== item 17: parameters are restored from PNG ==")
    produced = engine.generate(
        GenerationRequest(
            prompt="a bowl of cherries", preset=presets.get("LowQuality"), aspect="3:2", seed=123
        )
    )
    if not produced:
        record("PNG metadata", "FAIL", "no image")
        return
    item = produced[0]
    path = metadata.save_png(item.image, OUT / "metadata.png", item.parameters)
    read_back = metadata.read_png(path)
    ok = read_back == item.parameters and read_back.get("seed") == 123 and read_back.get("aspect") == "3:2"
    record(
        "PNG metadata",
        "ok" if ok else "FAIL",
        f"seed {read_back.get('seed')}, ratio {read_back.get('aspect')}, keys {len(read_back)}",
    )


SCENARIOS = {
    "presets": scenario_presets,
    "transparent": scenario_transparent,
    "aspects": scenario_aspects,
    "edit": None,  # требует исходного кадра, обрабатывается отдельно
    "mask": None,
    "annotation": None,
    "region": None,
    "outpaint": None,
    "references": scenario_references,
    "interrupt": scenario_interrupt,
    "swaps": None,
    "metadata": scenario_metadata,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke run of the scenarios that need generation")
    parser.add_argument("--only", nargs="*", default=None, help="run only the named scenarios")
    args = parser.parse_args()
    chosen = set(args.only) if args.only else None

    def wanted(name: str) -> bool:
        return chosen is None or name in chosen

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    pipe, residency, cache = loader.load(config.MODEL_DIR)
    print(f"model ready in {time.perf_counter() - started:.1f} s", flush=True)
    engine = Generator(pipe, residency, cache, load_styles(config.STYLES_DIR))

    if wanted("presets"):
        scenario_presets(engine)
    if wanted("transparent"):
        scenario_transparent(engine)
    if wanted("aspects"):
        scenario_aspects(engine)

    needs_source = any(wanted(name) for name in ("edit", "mask", "annotation", "region", "outpaint"))
    source = make_source(engine) if needs_source else None
    if source is not None:
        if wanted("edit"):
            scenario_edit_plain(engine, source)
        if wanted("mask"):
            scenario_edit_mask(engine, source)
        if wanted("annotation"):
            scenario_annotation(engine, source)
        if wanted("region"):
            scenario_region(engine, source)
        if wanted("outpaint"):
            scenario_outpaint(engine, source)

    if wanted("references"):
        scenario_references(engine)
    if wanted("metadata"):
        scenario_metadata(engine)
    if wanted("swaps"):
        scenario_swaps(engine, residency)
    if wanted("interrupt"):
        scenario_interrupt(engine)

    print("\n=== SUMMARY ===")
    failed = [row for row in RESULTS if row[1] != "ok"]
    for point, verdict, note in RESULTS:
        print(f"{verdict:6s} | {point} | {note}")
    print(f"\ntotal {len(RESULTS)}, failed {len(failed)}")
    print(f"images: {OUT}")
    print(f"cache: {cache.hits} hits / {cache.misses} misses; {residency.stats()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

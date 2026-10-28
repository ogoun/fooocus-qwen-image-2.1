r"""Опыт: как показать модели область правки — ч/б маской или цветной пометкой (по Kokon).

Повод (2026-10-28): Kokon (GabrieleLozupone/kokon, ``crates/qls-core/src/
pipeline.rs``) указывает область иначе, чем мы:
- вторым изображением идёт копия кадра, где область закрашена цветом и
  обведена; цвет выбирается тот, которого в области мало;
- промт шаблоном называет этот цвет: «… the area marked in red in <image2>»;
- при удалении объект сначала замазывается прямо в <image1>, чтобы модель
  строила фон, а не перерисовывала объект обратно.

Наш способ — пара «исходник + резкая ч/б маска» и голый промт
(``docs/research/2026-09-22-maska-kak-alfa.md``: слова о маске вредили).
Цветную пометку мы не мерили, удаление — тоже.

Варианты (все — рабочий путь ``Generator.generate``, пресет QwenTurbo, без
склейки, чтобы мерить ответ самой модели):

  ``pair``        — как сейчас: маска режима «Маска», голый промт;
  ``guide``       — исходник + цветная копия, промт-шаблон Kokon;
  ``guide+fill``  — то же, но объект замазан в исходнике (только удаление);
  ``pair+fill``   — наша пара с замазанным исходником: что даёт сама замазка.

Сюжеты: «добавить розу в волосы» (``logs/smoke/source.png``, тот же, что у
``mask_local.py``) и «удалить яблоко» — кадр делает сама модель, яблоко
находится по цвету.

Меры:
- ``outside`` — средняя разница с исходником вне области; ``outside %`` —
  доля пикселей вне области, изменившихся больше чем на 16 уровней;
- для розы ``red inside`` — доля красных пикселей в области (роза есть);
- для яблока ``apple left`` — доля красных пикселей на месте яблока
  (должна упасть к нулю), ``ghost`` — средняя разница с фоном,
  восстановленным ``cv2.inpaint``: опора грубая, но объект она не рисует.

Кадры — в ``tmp/mask_guide/``, их надо смотреть глазами.

Запуск (нужна видеокарта):
    .venv\Scripts\python tools\experiments\mask_guide.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import generator as gen  # noqa: E402
from fooocus_qwen.engine import presets  # noqa: E402
from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402

WORK = ROOT / "tmp" / "mask_guide"
PRESET = "QwenTurbo"
ROSE = "a small red rose tucked into her hair"
APPLE_SCENE = ("a photograph of a single red apple standing in the middle of an empty light oak table, "
               "a plain white wall behind it, soft daylight from the left")
HIGHLIGHTS = (("red", (255, 0, 0)), ("green", (0, 230, 0)), ("blue", (0, 80, 255)), ("magenta", (255, 0, 255)))


def kokon_prompt(action: str, what: str, color: str, filled: bool) -> str:
    """Шаблоны ``inpaint_prompt`` Kokon дословно."""
    area = f"the area marked in {color} in <image2>"
    keep = "Keep everything outside the marked area exactly as it is."
    if action == "add":
        return (f"In <image1>, add {what} inside {area}, fitting it naturally into the scene with matching "
                f"lighting, perspective, scale and shadows. {keep}")
    if filled:
        return (f"In <image1>, {area} has been blurred out to remove {what}. Replace the blurred area with the "
                "background that would naturally be behind it, continuing the surrounding scene, textures, lighting "
                f"and perspective, so that no trace, outline or shadow of the removed content remains. {keep}")
    return (f"In <image1>, remove {what} inside {area} and replace it with the background that would naturally be "
            "behind it, continuing the surrounding scene, textures, lighting and perspective, so that no trace, "
            f"outline or shadow of it remains. {keep}")


def red(array: np.ndarray) -> np.ndarray:
    r, g, b = (array[..., i].astype(np.int32) for i in range(3))
    return (r > 110) & (r > g * 1.6) & (r > b * 1.6)


def pick_highlight(image: Image.Image, mask: np.ndarray) -> tuple[str, tuple[int, int, int]]:
    """Как ``imaging::pick_highlight`` Kokon: красный, если в области его мало, иначе самый редкий там."""
    pixels = np.asarray(image.convert("RGB")).astype(np.int32)[mask]
    near = [int(((pixels - np.array(c)) ** 2).sum(axis=1).__lt__(140 * 140).sum()) for _, c in HIGHLIGHTS]
    if not len(pixels) or near[0] * 100 < len(pixels) * 15:
        return HIGHLIGHTS[0]
    return HIGHLIGHTS[int(np.argmin(near))]


def guide(image: Image.Image, mask: np.ndarray, color) -> Image.Image:
    """Цветная копия: заливка цветом наполовину и сплошная обводка (как ``imaging::highlight``)."""
    base = np.asarray(image.convert("RGB")).astype(np.float32)
    inside = mask.astype(np.uint8)
    ring = cv2.dilate(inside, np.ones((9, 9), np.uint8)) - inside
    tinted = base.copy()
    tinted[mask] = base[mask] * 0.45 + np.array(color, np.float32) * 0.55
    tinted[ring.astype(bool)] = color
    return Image.fromarray(tinted.clip(0, 255).astype(np.uint8))


def fill(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Замазка: фон дорисован ``cv2.inpaint`` и размыт внутри области (``fill_masked`` Kokon)."""
    rgb = np.asarray(image.convert("RGB"))
    painted = cv2.inpaint(rgb, mask.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
    blurred = cv2.GaussianBlur(painted, (0, 0), 12)
    out = rgb.copy()
    out[mask] = blurred[mask]
    return Image.fromarray(out)


def outside_scores(source: Image.Image, produced: Image.Image, mask: np.ndarray) -> dict:
    a = np.asarray(source.convert("RGB")).astype(np.float32)
    b = np.asarray(produced.convert("RGB").resize(source.size, Image.LANCZOS)).astype(np.float32)
    delta = np.abs(a - b).mean(axis=2)
    return {"outside": round(float(delta[~mask].mean()), 2), "outside_pct": round(float((delta[~mask] > 16).mean() * 100), 1)}


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(description="Edit region: black-and-white mask against a colour mark")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--only", nargs="*", default=None, help="tasks: rose apple")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR = WORK / "outputs"
    config.SETTINGS_FILE = WORK / "settings.json"
    config.SETTINGS_FILE.write_text(json.dumps({"precision": args.precision, "sage_attention": True}), encoding="utf-8")

    from fooocus_qwen.ui.state import Studio

    generator = Studio(config.AppConfig(lang="en", preload=False)).generator
    preset = presets.get(PRESET)

    def run(prompt, seed, **fields):
        started = time.perf_counter()
        image = generator.generate(gen.GenerationRequest(prompt=prompt, preset=preset, seed=seed, **fields))[0].image
        return image, time.perf_counter() - started

    # --- сюжет «роза» --------------------------------------------------------
    portrait = Image.open(ROOT / "logs" / "smoke" / "source.png").convert("RGB")
    w, h = portrait.size
    raw = Image.new("L", portrait.size, 0)
    ImageDraw.Draw(raw).ellipse((int(w * 0.60), int(h * 0.22), int(w * 0.76), int(h * 0.38)), fill=255)
    rose_mask = np.asarray(raw) >= 128

    # --- сюжет «яблоко»: кадр делает модель, яблоко — по цвету ----------------
    table, _ = run(APPLE_SCENE, 11, aspect="1:1")
    table = table.convert("RGB")
    table.save(WORK / "apple_source.png")
    apple = red(np.asarray(table))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(apple.astype(np.uint8))
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if count > 1 else 0
    apple = labels == biggest
    # Тёмный бок яблока по цвету «красным» не считается — без выпуклой оболочки
    # маска выходила с вырезом, и замазка оставляла кусок яблока (первый прогон
    # 2026-10-28). Как человек кистью: всё яблоко и запас вокруг.
    hull = cv2.convexHull(np.argwhere(apple)[:, ::-1].astype(np.int32))
    apple = cv2.fillConvexPoly(np.zeros(apple.shape, np.uint8), hull, 1).astype(bool)
    apple_mask = cv2.dilate(apple.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    reference = Image.fromarray(cv2.inpaint(np.asarray(table), apple_mask.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA))
    print(f"apple: {int(apple.sum())} px, area {int(apple_mask.sum())} px", flush=True)

    tasks = {
        "rose": dict(source=portrait, mask=rose_mask, action="add", what=ROSE, bare=ROSE),
        "apple": dict(source=table, mask=apple_mask, action="remove", what="the red apple", bare="remove the red apple"),
    }
    rows: list[dict] = []
    for task, spec in tasks.items():
        if args.only and task not in args.only:
            continue
        source, mask = spec["source"], spec["mask"]
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        color_name, color = pick_highlight(source, mask)
        filled = fill(source, mask)
        variants = {"pair": (source, False), "guide": (source, True)}
        if spec["action"] == "remove":
            variants.update({"pair+fill": (filled, False), "guide+fill": (filled, True)})
        for variant, (image1, guided) in variants.items():
            for seed in range(1, args.seeds + 1):
                if guided:
                    prompt = kokon_prompt(spec["action"], spec["what"], color_name, image1 is filled)
                    produced, seconds = run(prompt, seed, source=image1, mask_mode=gen.MASK_NONE,
                                            references=(guide(image1, mask, color),))
                else:
                    produced, seconds = run(spec["bare"], seed, source=image1, mask=mask_image,
                                            mask_mode=gen.MASK_MASK, keep_outside=False, mask_feather=0)
                produced = produced.convert("RGB").resize(source.size, Image.LANCZOS)
                produced.save(WORK / f"{task}_{variant}_s{seed}.png")
                row = {"task": task, "variant": variant, "seed": seed, "seconds": round(seconds, 1),
                       **outside_scores(source, produced, mask)}
                array = np.asarray(produced)
                if task == "rose":
                    row["red_inside"] = round(float(red(array)[mask].mean() * 100), 1)
                else:
                    row["apple_left"] = round(float(red(array)[apple].mean() * 100), 1)
                    row["ghost"] = round(float(np.abs(array.astype(np.float32) - np.asarray(reference, np.float32))
                                               .mean(axis=2)[apple].mean()), 1)
                rows.append(row)
                print(row, flush=True)
                (WORK / "scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        if task == "rose":
            guide(source, mask, color).save(WORK / "rose_guide.png")
        else:
            guide(filled, mask, color).save(WORK / "apple_guide.png")
            filled.save(WORK / "apple_filled.png")

    print("\n=== means over seeds ===")
    keys = sorted({(row["task"], row["variant"]) for row in rows}, key=lambda k: (k[0], k[1]))
    for task, variant in keys:
        group = [row for row in rows if (row["task"], row["variant"]) == (task, variant)]
        mean = {name: round(float(np.mean([row[name] for row in group])), 2)
                for name in group[0] if name not in ("task", "variant", "seed")}
        print(f"{task:6} {variant:11} {mean}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

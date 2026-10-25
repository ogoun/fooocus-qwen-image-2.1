r"""Опыт: официальные системные промты Qwen против сводного промта «Sol» и против доработок.

Вопрос пришёл от пользователя: сторонняя модель предложила один сводный
системный промт на генерацию и правку (``prompts/sol_system_prompt.txt``),
шаблон сообщения со списком ``<image1> = first attached image`` и поле
``mode`` в ответе. Наши промты — официальные из QwenLM/Qwen-Image-2.1
(``prompt_rewrite/prompts``), байт в байт. Сравниваем четыре варианта:

  ``official``  — как в оболочке сейчас: официальный промт, наше сообщение,
                  изображения после текста, в полном размере;
  ``sol``       — сводный промт и шаблон сообщения Sol, та же доставка;
  ``transport`` — official, но доставка как в официальном коде
                  (``prompt_rewrite/pe_core.py``): изображения перед текстом
                  и не крупнее 1024² пикселей, как при обучении переписывателя;
  ``tuned``     — transport плюс две пометки: о прозрачном фоне (T2I, формат
                  из README Qwen-Image-2.1) и о цветных пометках (режим
                  «Аннотация»).

Для генерации ``transport`` совпадает с ``official`` (изображений нет) и не
гоняется. Меры автоматические, по каждому ответу: JSON разобрался, описание
по-английски (``boost.off_language``), длина, задержка, и мера случая —
см. ``CASES``. Все ответы пишутся в ``tmp/boost_variants/answers.jsonl``
для чтения глазами.

Изображения синтетические (дом, дерево, небо): личные фото пользователя
в опытах не используются.

Запуск (нужен настроенный llm_endpoint.txt; видеокарта не нужна):
    .venv\Scripts\python tools\experiments\boost_prompt_variants.py --runs 4
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image, ImageDraw

from fooocus_qwen import config
from fooocus_qwen.logging_setup import use_utf8_console
from fooocus_qwen.prompting import boost
from fooocus_qwen.ui.state import Studio

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tmp" / "boost_variants"
SOL = Path(__file__).resolve().parent / "prompts" / "sol_system_prompt.txt"

# Предел из pe_core.Profile.image_max_pixels: «Cap on each source image's
# pixel count, matching training».
MAX_PIXELS = 1024 * 1024

RGBA_OPEN = "This is an RGBA image with transparency."
RGBA_CLOSE = "The image has alpha channel and the background is transparent."

T2I_NOTE = (
    "\n\n## Transparent images\n\n"
    "If the user asks for a transparent background, a cut-out, or an isolated asset with "
    f"no background, the description begins with exactly \"{RGBA_OPEN}\" and ends with "
    f"exactly \"{RGBA_CLOSE}\" — this is the prompt format Qwen-Image-2.1 is trained on "
    "for transparent output. Between them describe only the subject: no backdrop, floor, "
    "surface or background colour, and no white or checkerboard background. Never use "
    "these two sentences when transparency was not asked for."
)

ANNOTATION_NOTE = (
    "\n\nNote: the coloured strokes, circles and arrows drawn over the input image are the "
    "user's annotation marks. They only show where each edit applies. Refer to each marked "
    "area by its colour and by the object under it, and state that the marks themselves "
    "are gone from the result."
)


def scene(size: int = 1536) -> Image.Image:
    """Дом слева, дерево справа, небо и трава: чтобы было что назвать."""
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    horizon = size * 3 // 5
    for y in range(horizon):
        shade = 255 - y * 90 // horizon
        draw.line([(0, y), (size, y)], fill=(110, 170, shade))
    draw.rectangle([0, horizon, size, size], fill=(70, 140, 60))
    u = size // 16
    draw.rectangle([2 * u, 6 * u, 7 * u, 11 * u], fill=(190, 60, 50))
    draw.polygon([(u + u // 2, 6 * u), (4 * u + u // 2, 3 * u), (7 * u + u // 2, 6 * u)], fill=(100, 60, 40))
    draw.rectangle([4 * u, 8 * u, 5 * u, 11 * u], fill=(80, 50, 30))
    draw.rectangle([11 * u + u // 2, 7 * u, 12 * u + u // 2, 11 * u], fill=(90, 60, 30))
    draw.ellipse([9 * u, 3 * u, 15 * u, 8 * u], fill=(40, 110, 40))
    return image


def annotated(size: int = 1536) -> Image.Image:
    image = scene(size)
    u = size // 16
    ImageDraw.Draw(image).ellipse([8 * u, 2 * u, 16 * u - 8, 12 * u], outline=(255, 0, 0), width=14)
    return image


def star(size: int = 512) -> Image.Image:
    image = Image.new("RGB", (size, size), (255, 255, 255))
    c, r = size / 2, size * 0.42
    points = []
    for k in range(10):
        radius = r if k % 2 == 0 else r * 0.42
        angle = math.pi / 2 + k * math.pi / 5
        points.append((c + radius * math.cos(angle), c - radius * math.sin(angle)))
    ImageDraw.Draw(image).polygon(points, fill=(240, 200, 20), outline=(150, 110, 0))
    return image


QUOTED = re.compile(r'["“”]([^"“”]*)["“”]')


def has_quoted(text: str, needle: str) -> bool:
    return any(needle in match for match in QUOTED.findall(text))


def mark_removed(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(
        r"(remov|eras|delet|no longer|not visible|not appear|disappear|gone|leave no|without (any|the))"
        r"[^.]{0,80}(circle|ring|outline|mark|annotation|stroke)"
        r"|(circle|ring|outline|mark|annotation|stroke)[^.]{0,80}"
        r"(removed|erased|gone|disappear|not (appear|visible|remain|be visible))",
        lowered,
    ))


def mark_kept(text: str) -> bool:
    lowered = text.lower()
    return bool(re.search(
        r"(preserv|keep|retain|maintain)[^.]{0,60}(red (circle|ring|outline|mark)|annotation)", lowered,
    ))


# Случай: (режим, инструкция, изображения, теги для official/transport/tuned, мера).
def cases() -> dict[str, tuple]:
    plain, marked, ref = scene(), annotated(), star()
    return {
        "t2i_plain": ("t2i", "рыжая лиса в заснеженном берёзовом лесу на закате", [], None,
                      lambda p, r: {"rgba_false": "RGBA" in p, "ratio": bool(r.wh_ratio)}),
        "t2i_rgba": ("t2i", "наклейка: кот-астронавт в скафандре, прозрачный фон", [], None,
                     lambda p, r: {"rgba": p.startswith(RGBA_OPEN) and "background is transparent" in p[-120:],
                                   "rgba_any": "RGBA" in p or "transparen" in p.lower()}),
        "t2i_text": ("t2i", "постер джазового фестиваля с надписью «Белые ночи» и датой «21 июня»", [], None,
                     lambda p, r: {"text_exact": has_quoted(p, "Белые ночи") and has_quoted(p, "21 июня"),
                                   "rgba_false": "RGBA" in p}),
        "edit_single": ("edit", "сделай небо закатным", [plain], None,
                        lambda p, r: {"follow_image1": r.ratio_follow == "<image1>" and not r.wh_ratio,
                                      "no_tags": "<image" not in p}),
        # Правка по маске: исходник <image1>, маска <image2> (переписывателю не
        # показывается), звезда-референс — <image3>.
        "edit_mask_ref": ("edit", "поставь на лужайку перед домом жёлтую звезду как на <image3>",
                          [plain, ref], ["<image1>", "<image3>"],
                          lambda p, r: {"tag3": "<image3>" in p and "<image2>" not in p}),
        "edit_annotation": ("edit", "убери дерево в красном круге", [marked], None,
                            lambda p, r: {"mark_removed": mark_removed(p), "mark_kept": mark_kept(p),
                                          "names_red": "red" in p.lower()}),
    }


def sol_user(prompt: str, count: int) -> str:
    lines = ["Improve the following request for Qwen-Image-2.1.", "", f"Input image count: {count}"]
    if count:
        lines += ["", "The images are attached in this exact order:",
                  "<image1> = first attached image", "<image2> = second attached image",
                  "<image3> = third attached image", "Continue sequentially if more images are attached."]
    lines += ["", "Original user request:", prompt]
    return "\n".join(lines)


def cap(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w * h <= MAX_PIXELS:
        return image
    s = (MAX_PIXELS / (w * h)) ** 0.5
    return image.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def official_transport(client) -> None:
    """Изображения перед текстом и не крупнее 1024², как в pe_core.build_messages."""
    original = client._build_body

    def build(system, user, images, temperature, max_tokens):
        body = original(system, user, [cap(i) for i in images] if images else images, temperature, max_tokens)
        content = body["messages"][1]["content"]
        if isinstance(content, list):
            body["messages"][1]["content"] = content[1:] + content[:1]
        return body

    client._build_body = build


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(description="Experiment: official Qwen rewrite prompts vs Sol's merged prompt.")
    parser.add_argument("--runs", type=int, default=4, help="repeats per case and variant")
    parser.add_argument("--only", nargs="*", help="case names to run")
    args = parser.parse_args()

    studio = Studio(config.AppConfig())
    plain_client, fast_client = studio.llm_client(), studio.llm_client()
    official_transport(fast_client)
    official = {name: (config.SYSTEM_PROMPT_DIR / f"system_prompt_{name}.txt").read_text(encoding="utf-8")
                for name in ("t2i", "edit")}
    sol = SOL.read_text(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    log = (OUT / "answers.jsonl").open("w", encoding="utf-8", newline="\n")
    print(f"model: {plain_client.model or '(unnamed)'}\n")

    def message(variant, case_name, mode, prompt, images, tags):
        if variant == "sol":
            return sol, sol_user(prompt, len(images))
        system = official[mode]
        user = boost.build_user_message(prompt, len(images), tags)
        if variant == "tuned":
            if mode == "t2i":
                system += T2I_NOTE
            elif case_name == "edit_annotation":
                user += ANNOTATION_NOTE
        return system, user

    totals: dict[tuple[str, str], dict[str, float]] = {}
    for case_name, (mode, prompt, images, tags, measure) in cases().items():
        if args.only and case_name not in args.only:
            continue
        variants = ["official", "sol", "tuned"] if mode == "t2i" else ["official", "sol", "transport", "tuned"]
        for variant in variants:
            client = plain_client if variant in ("official", "sol") else fast_client
            system, user = message(variant, case_name, mode, prompt, images, tags)
            row = totals.setdefault((case_name, variant), {"n": 0, "json": 0, "english": 0, "words": 0, "s": 0.0})
            for _ in range(args.runs):
                started = time.time()
                raw = client.complete(system, user, images=images or None)
                seconds = time.time() - started
                result = boost.parse_response(raw)
                text = result.prompt
                parsed = text != boost._THINK.sub("", raw).strip()  # проза уходит как есть
                scores = measure(text, result)
                row["n"] += 1
                row["json"] += parsed
                row["english"] += not boost.off_language(prompt, text)
                row["words"] += len(text.split())
                row["s"] += seconds
                for key, value in scores.items():
                    row[key] = row.get(key, 0) + bool(value)
                log.write(json.dumps({"case": case_name, "variant": variant, "seconds": round(seconds, 1),
                                      "scores": scores, "wh_ratio": result.wh_ratio,
                                      "ratio_follow": result.ratio_follow, "prompt": text},
                                     ensure_ascii=False) + "\n")
                log.flush()
                print(f"[{case_name}/{variant}] {seconds:.1f}s {scores} {text[:100]!r}", flush=True)

    print("\ncase            | variant   | n | json | english | words | s    | case measures")
    for (case_name, variant), row in totals.items():
        n = row["n"]
        extra = ", ".join(f"{k} {row[k]:.0f}/{n}" for k in row if k not in ("n", "json", "english", "words", "s"))
        print(f"{case_name:15} | {variant:9} | {n} | {row['json']:.0f}/{n}  | {row['english']:.0f}/{n}     "
              f"| {row['words'] / n:5.0f} | {row['s'] / n:4.1f} | {extra}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

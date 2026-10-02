"""Расширение кадра: серый холст для лоры outpaint и вклейка оригинала обратно.

Рецепт — лоры ``ausboss/Qwen-Image-2.1-Outpaint-LoRA`` (разбор —
``docs/research/2026-10-02-outpaint.md``): картинка ставится на холст,
новая площадь заливается ровным серым ``#808080``, холст целиком подаётся
модели исходником правки **без маски** (маска на Qwen 2.1 оставляет на шве
видимую рамку), и лора заполняет серое продолжением сцены, не сдвигая сам
кадр. Без лоры Qwen 2.1 кадр перекомпонует или сжимает: сохранённая часть
расходится с оригиналом на 16–25 дБ PSNR против 34 дБ с лорой.

Здесь — геометрия и пиксели, без модели:

* ``plan`` / ``plan_for_ratio`` — где окажется оригинал на холсте: по
  сторонам и доле или до заданного соотношения сторон;
* ``prepare`` — холст в размере генерации: 1–2 Мп (на них учили лору),
  стороны кратны 32, оригинал — в том же масштабе;
* ``stitch`` — результат в масштаб исходника, оригинал поверх с
  растушёвкой по краю и выравниванием тона: лора держит кадр на месте, и
  вклейка ложится без шва.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .aspect import ASPECT_RATIOS, FOLLOW_REFERENCE, MULTIPLE

SIDES: tuple[str, ...] = ("left", "right", "top", "bottom")
FILL = (128, 128, 128)
# Растушёвка вклейки по краю оригинала (пикселей исходника): лора держит
# кадр на месте с точностью до пикселя, шире незачем.
FEATHER = 32
# Площадь генерации: лору v2 учили на 1–2 Мп.
MIN_AREA = 1024 * 1024
MAX_AREA = 2 * 1024 * 1024


@dataclass(frozen=True)
class OutpaintPlan:
    """Куда вырастет холст и где на нём окажется оригинал (пиксели исходника)."""

    canvas_size: tuple[int, int]
    paste_box: tuple[int, int, int, int]


def _ceil_multiple(value: int) -> int:
    """Вверх до кратности 32, не ниже одной кратности: холст обязан вместить прирост целиком."""
    return max(MULTIPLE, -(-value // MULTIPLE) * MULTIPLE)


def plan(size: tuple[int, int], sides: Sequence[str], amount: float) -> OutpaintPlan:
    """Холст, выросший на ``amount`` своей стороны в каждую из ``sides``."""
    unknown = [side for side in sides if side not in SIDES]
    if unknown:
        raise ValueError(f"Unknown outpaint side: {', '.join(unknown)}")

    width, height = size
    if not sides or amount <= 0:
        return OutpaintPlan(canvas_size=(width, height), paste_box=(0, 0, width, height))

    left = int(width * amount) if "left" in sides else 0
    right = int(width * amount) if "right" in sides else 0
    top = int(height * amount) if "top" in sides else 0
    bottom = int(height * amount) if "bottom" in sides else 0

    # Ось, которую не просили расширять, остаётся в точности исходной.
    canvas_width = _ceil_multiple(width + left + right) if (left or right) else width
    canvas_height = _ceil_multiple(height + top + bottom) if (top or bottom) else height

    # Остаток округления — той стороне (или обеим), что и так растёт.
    slack_x = canvas_width - width - left - right
    if left and right:
        offset_x = left + slack_x // 2
    elif left:
        offset_x = canvas_width - width
    else:
        offset_x = 0

    slack_y = canvas_height - height - top - bottom
    if top and bottom:
        offset_y = top + slack_y // 2
    elif top:
        offset_y = canvas_height - height
    else:
        offset_y = 0

    return OutpaintPlan(
        canvas_size=(canvas_width, canvas_height),
        paste_box=(offset_x, offset_y, offset_x + width, offset_y + height),
    )


def plan_for_ratio(size: tuple[int, int], ratio: str) -> OutpaintPlan:
    """Холст заданного соотношения сторон, оригинал — по центру растущей оси.

    Растёт только та ось, которой не хватает; кадр уже нужного соотношения не
    меняется (пустой план).
    """
    if ratio not in ASPECT_RATIOS or ratio == FOLLOW_REFERENCE:
        raise ValueError(f"Unknown aspect ratio: {ratio}")
    ratio_width, ratio_height = (int(part) for part in ratio.split(":"))
    target = ratio_width / ratio_height
    width, height = size
    if width / height < target - 1e-3:
        canvas = (math.ceil(height * target), height)
    elif width / height > target + 1e-3:
        canvas = (width, math.ceil(width / target))
    else:
        return OutpaintPlan(canvas_size=size, paste_box=(0, 0, width, height))
    left = (canvas[0] - width) // 2
    top = (canvas[1] - height) // 2
    return OutpaintPlan(canvas_size=canvas, paste_box=(left, top, left + width, top + height))


@dataclass(frozen=True)
class OutpaintJob:
    """Холст для модели и то, как вернуть результат к масштабу исходника."""

    canvas: Image.Image
    plan: OutpaintPlan
    scale: float  # размер генерации / размер холста в пикселях исходника

    @property
    def size(self) -> tuple[int, int]:
        return self.canvas.size

    @property
    def resolution(self) -> int:
        """Сторона квадрата той же площади — ``output_resolution`` пайплайна.

        Пайплайн приводит исходник к площади ``resolution²`` с тем же
        соотношением сторон и округлением до 32; подобрано так, чтобы это
        дало ровно ``size`` — холст идёт в модель без пересчёта.
        """
        return generation_resolution(self.size)


def generation_resolution(size: tuple[int, int]) -> int:
    width, height = size
    base = round(math.sqrt(width * height))
    for delta in (0, 1, -1, 2, -2, 3, -3, 4, -4):
        candidate = base + delta
        side = math.sqrt(candidate * candidate * width / height)
        if (round(side / MULTIPLE) * MULTIPLE, round(side / (width / height) / MULTIPLE) * MULTIPLE) == (width, height):
            return candidate
    return base


def generation_area(preset_resolution: int) -> int:
    """Площадь генерации для пресета: в пределах 1–2 Мп, на которых учили лору.

    Маленький исходник при этом считается крупнее себя — модель видит кадр
    в привычном масштабе, — а результат возвращается к масштабу исходника.
    """
    return max(MIN_AREA, min(preset_resolution * preset_resolution, MAX_AREA))


def generation_size(canvas_size: tuple[int, int], area: int) -> tuple[int, int]:
    """Размер генерации: площадь около ``area``, стороны кратны 32, соотношение — холста."""
    width, height = canvas_size
    scale = math.sqrt(area / (width * height))
    return (max(MULTIPLE, round(width * scale / MULTIPLE) * MULTIPLE),
            max(MULTIPLE, round(height * scale / MULTIPLE) * MULTIPLE))


def _flat(image: Image.Image) -> Image.Image:
    """RGB поверх серого: прозрачное становится тем же «пустым местом», что и новая площадь."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        backdrop = Image.new("RGBA", rgba.size, FILL + (255,))
        return Image.alpha_composite(backdrop, rgba).convert("RGB")
    return image.convert("RGB")


def prepare(image: Image.Image, outpaint_plan: OutpaintPlan, area: int = MIN_AREA) -> OutpaintJob:
    """Серый холст в размере генерации (площадь ``area``) с оригиналом на своём месте."""
    gen_width, gen_height = generation_size(outpaint_plan.canvas_size, area)
    scale_x = gen_width / outpaint_plan.canvas_size[0]
    scale_y = gen_height / outpaint_plan.canvas_size[1]
    left, top, right, bottom = outpaint_plan.paste_box
    box = (round(left * scale_x), round(top * scale_y), round(right * scale_x), round(bottom * scale_y))
    canvas = Image.new("RGB", (gen_width, gen_height), FILL)
    picture = _flat(image).resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    canvas.paste(picture, box[:2])
    return OutpaintJob(canvas=canvas, plan=outpaint_plan, scale=(scale_x + scale_y) / 2)


def _seam_mask(outpaint_plan: OutpaintPlan, feather: int) -> Image.Image:
    """Белое — оригинал, к краям, граничащим с новой площадью, — спад в ``feather`` пикселей.

    Края оригинала, совпадающие с краем холста, не растушёвываются: по ту
    сторону нет ничего, с чем смешивать.
    """
    width, height = outpaint_plan.canvas_size
    left, top, right, bottom = outpaint_plan.paste_box
    inner_w, inner_h = right - left, bottom - top
    ramp_x = np.ones(inner_w, dtype=np.float32)
    ramp_y = np.ones(inner_h, dtype=np.float32)
    steps = (np.arange(feather, dtype=np.float32) + 0.5) / feather
    if left > 0:
        ramp_x[: min(feather, inner_w)] = np.minimum(ramp_x[: min(feather, inner_w)], steps[: min(feather, inner_w)])
    if right < width:
        ramp_x[-min(feather, inner_w):] = np.minimum(ramp_x[-min(feather, inner_w):], steps[: min(feather, inner_w)][::-1])
    if top > 0:
        ramp_y[: min(feather, inner_h)] = np.minimum(ramp_y[: min(feather, inner_h)], steps[: min(feather, inner_h)])
    if bottom < height:
        ramp_y[-min(feather, inner_h):] = np.minimum(ramp_y[-min(feather, inner_h):], steps[: min(feather, inner_h)][::-1])
    inner = np.outer(ramp_y, ramp_x)
    mask = np.zeros((height, width), dtype=np.float32)
    mask[top:bottom, left:right] = inner
    return Image.fromarray(np.round(mask * 255).astype(np.uint8), mode="L")


def stitch(original: Image.Image, generated: Image.Image, job: OutpaintJob, feather: int = FEATHER) -> Image.Image:
    """Результат в масштаб исходника, оригинал — поверх, без шва.

    Тон: средний цвет результата там, где лежит оригинал, сравнивается с
    оригиналом, и разница вычитается из всего результата — модель могла
    сдвинуть баланс, а новая площадь должна сойтись с вклеенным кадром.
    """
    plan_ = job.plan
    full = generated.convert("RGB").resize(plan_.canvas_size, Image.Resampling.LANCZOS)
    source = _flat(original)
    left, top, right, bottom = plan_.paste_box
    produced = np.asarray(full, dtype=np.float32)
    inner = produced[top:bottom, left:right]
    target = np.asarray(source, dtype=np.float32)
    if inner.size:
        # По средним: детали модель рисует по-своему, а тон — общий для кадра.
        shift = target.reshape(-1, 3).mean(axis=0) - inner.reshape(-1, 3).mean(axis=0)
        produced = np.clip(produced + shift, 0, 255)
    result = Image.fromarray(produced.astype(np.uint8), mode="RGB")
    placed = Image.new("RGB", plan_.canvas_size, FILL)
    placed.paste(source, (left, top))
    return Image.composite(placed, result, _seam_mask(plan_, max(1, feather)))


# --- промт -----------------------------------------------------------------------

# Инструкция — триггер лоры (поле ``instruction`` в метаданных её файла):
# идёт первой и без изменений.
INSTRUCTION = (
    "Outpaint the image: replace the solid gray areas with a seamless continuation of the scene, "
    "keeping the existing picture unchanged."
)
# Как языковой модели описать кадр — так, как написаны подписи, на которых
# учили лору (из сценария автора).
DESCRIBE_INSTRUCTION = (
    "Write the text-to-image prompt that would generate this exact picture. One paragraph of 50 to 90 words. "
    "Begin with the medium and style in a few words, such as \"Photograph\", \"Smartphone photo\", "
    "\"Anime illustration\", \"Oil painting\" or \"3D render\". Then describe the subject, the setting, and what "
    "fills every part of the frame from edge to edge, including the background and the areas near the borders. "
    "Any flat mid-gray parts, such as corners or a background, are blank space that will be painted to continue "
    "the scene: describe the scene as if it filled the whole frame, and don't describe a border, frame or "
    "vignette around it. Include materials, colours, lighting, time of day, camera angle and framing. Quote any "
    "readable text exactly. Use plain factual language; do not start with 'The image' or 'This picture' and do "
    "not give opinions. Output only the prompt."
)


def prompt(description: str = "", wishes: str = "") -> str:
    """Инструкция-триггер, затем описание сцены и пожелания человека."""
    parts = [INSTRUCTION]
    if description.strip():
        parts.append(f"Scene: {description.strip()}")
    if wishes.strip():
        parts.append(f"Also include in the prompt: {wishes.strip()}")
    return " ".join(parts)

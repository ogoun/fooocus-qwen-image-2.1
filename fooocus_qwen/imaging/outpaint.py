"""Расширение холста: подготовка условного изображения и маски.

Дорисовка за границами кадра — это тот же локальный правочный сценарий: новая
площадь объявляется маской, а исходное изображение остаётся нетронутым.

Новая площадь заполняется продолжением краевых пикселей. Пустой холст модель
трактует как часть композиции и дорисовывает границу изображения внутри кадра;
продолжение края такой подсказки не даёт.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .aspect import snap

SIDES: tuple[str, ...] = ("left", "right", "top", "bottom")


@dataclass(frozen=True)
class OutpaintPlan:
    """Куда вырастет холст и где на нём окажется оригинал."""

    canvas_size: tuple[int, int]
    paste_box: tuple[int, int, int, int]


def plan(size: tuple[int, int], sides: Sequence[str], amount: float) -> OutpaintPlan:
    unknown = [side for side in sides if side not in SIDES]
    if unknown:
        raise ValueError(f"Неизвестная сторона расширения: {', '.join(unknown)}")

    width, height = size
    if not sides or amount <= 0:
        return OutpaintPlan(canvas_size=(width, height), paste_box=(0, 0, width, height))

    left = int(width * amount) if "left" in sides else 0
    right = int(width * amount) if "right" in sides else 0
    top = int(height * amount) if "top" in sides else 0
    bottom = int(height * amount) if "bottom" in sides else 0

    canvas_width = snap(width + left + right)
    canvas_height = snap(height + top + bottom)

    # Округление холста до кратности 32 съедает или добавляет пиксели; отдаём
    # разницу тем полям, которые и так растут, чтобы оригинал не деформировался.
    offset_x = min(left, max(0, canvas_width - width))
    offset_y = min(top, max(0, canvas_height - height))

    return OutpaintPlan(
        canvas_size=(canvas_width, canvas_height),
        paste_box=(offset_x, offset_y, offset_x + width, offset_y + height),
    )


def expand(image: Image.Image, outpaint_plan: OutpaintPlan) -> tuple[Image.Image, Image.Image]:
    """Возвращает пару «условное изображение, маска новой площади»."""
    canvas_width, canvas_height = outpaint_plan.canvas_size
    left, top, right, bottom = outpaint_plan.paste_box

    source = np.asarray(image.convert("RGBA"))
    padded = cv2.copyMakeBorder(
        source,
        top=top,
        bottom=canvas_height - bottom,
        left=left,
        right=canvas_width - right,
        borderType=cv2.BORDER_REPLICATE,
    )
    canvas = Image.fromarray(padded, mode="RGBA")

    mask_array = np.full((canvas_height, canvas_width), 255, dtype=np.uint8)
    mask_array[top:bottom, left:right] = 0
    return canvas, Image.fromarray(mask_array, mode="L")

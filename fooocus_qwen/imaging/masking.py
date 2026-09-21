"""Маски: построение из редактора, уточнение края, склейка с оригиналом.

У Qwen-Image-2.1 нет параметра ``mask_image``: маска подаётся вторым условным
изображением. Модель понимает её семантически, поэтому машинерия inpaint из
Fooocus (заливка области, подмена латентов, патч модели) здесь не нужна — она
существовала ради SDXL, который маску не понимает.

Остаётся одна вещь, которую модель гарантировать не может: неприкосновенность
пикселей вне маски. Её обеспечивает ``blend``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .aspect import MULTIPLE


def mask_from_editor(value: Mapping[str, Any] | Image.Image) -> Image.Image:
    """Собирает маску из значения ``gr.ImageEditor`` или из готового изображения.

    Редактор отдаёт словарь с фоном, слоями и сведённой картинкой. Маска — это
    объединение альфа-каналов слоёв: кисть рисует по прозрачному слою, и всё,
    что непрозрачно, пользователь пометил.
    """
    if isinstance(value, Image.Image):
        return value.convert("L")

    background = value.get("background")
    if background is None:
        raise ValueError("В значении редактора нет фонового изображения")

    size = background.size
    combined = np.zeros((size[1], size[0]), dtype=np.uint8)
    for layer in value.get("layers") or []:
        if layer is None:
            continue
        alpha = np.asarray(layer.convert("RGBA").resize(size, Image.NEAREST))[..., 3]
        combined = np.maximum(combined, alpha)

    return Image.fromarray(combined, mode="L")


def is_empty(mask: Image.Image) -> bool:
    return not np.asarray(mask).any()


def refine(mask: Image.Image, grow: int = 0, feather: int = 0) -> Image.Image:
    """Расширяет маску и смягчает её край.

    Запас нужен потому, что правка почти всегда должна захватить немного соседних
    пикселей: контур объекта редко совпадает с движением кисти. Растушёвка
    убирает границу склейки — резкий переход виден даже при идеальном совпадении
    содержимого.
    """
    array = np.asarray(mask.convert("L"))

    if grow > 0:
        kernel_size = 2 * grow + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        array = cv2.dilate(array, kernel)

    if feather > 0:
        # Ядро размытия обязано быть нечётным, иначе OpenCV откажется работать.
        kernel_size = 2 * feather + 1
        array = cv2.GaussianBlur(array, (kernel_size, kernel_size), 0)

    return Image.fromarray(array, mode="L")


def as_condition(mask: Image.Image) -> Image.Image:
    """Превращает маску в условное изображение: белое — править, чёрное — оставить."""
    array = np.asarray(mask.convert("L"))
    return Image.fromarray(np.repeat(array[..., None], 3, axis=2), mode="RGB")


def blend(original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
    """Накладывает результат на оригинал по маске.

    Там, где маска строго нулевая, берётся исходный пиксель без каких-либо
    вычислений — это и есть гарантия неприкосновенности кадра вне правки.
    """
    base = original.convert("RGBA")
    patch = generated.convert("RGBA")
    if patch.size != base.size:
        patch = patch.resize(base.size, Image.LANCZOS)

    soft = mask.convert("L")
    if soft.size != base.size:
        soft = soft.resize(base.size, Image.LANCZOS)

    base_array = np.asarray(base).astype(np.float32)
    patch_array = np.asarray(patch).astype(np.float32)
    weight = (np.asarray(soft).astype(np.float32) / 255.0)[..., None]

    mixed = np.rint(base_array * (1.0 - weight) + patch_array * weight)
    untouched = weight == 0.0
    result = np.where(untouched, base_array, mixed).astype(np.uint8)

    return Image.fromarray(result, mode="RGBA")


def region_box(
    mask: Image.Image,
    padding: float = 0.25,
    multiple: int = MULTIPLE,
) -> tuple[int, int, int, int] | None:
    """Прямоугольник вокруг маски с контекстным запасом, кратный ``multiple``.

    Запас даёт модели увидеть окружение: вырезанный впритык фрагмент лишён
    контекста, и модель дорисовывает в нём что угодно. Итог всегда лежит внутри
    холста — выход за границы породил бы кадр не того размера.
    """
    array = np.asarray(mask.convert("L"))
    rows = np.flatnonzero(array.any(axis=1))
    columns = np.flatnonzero(array.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None

    height, width = array.shape
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1

    pad_x = int((right - left) * padding)
    pad_y = int((bottom - top) * padding)
    left, right = max(0, left - pad_x), min(width, right + pad_x)
    top, bottom = max(0, top - pad_y), min(height, bottom + pad_y)

    left, right = _snap_span(left, right, width, multiple)
    top, bottom = _snap_span(top, bottom, height, multiple)
    return left, top, right, bottom


def _snap_span(start: int, end: int, limit: int, multiple: int) -> tuple[int, int]:
    """Растягивает отрезок до кратной длины, не вылезая за ``limit``."""
    span = end - start
    target = min(((span + multiple - 1) // multiple) * multiple, (limit // multiple) * multiple)
    target = max(target, multiple)
    if target >= limit:
        return 0, limit

    start = max(0, start - (target - span) // 2)
    if start + target > limit:
        start = limit - target
    return start, start + target


def stitch(
    original: Image.Image,
    patch: Image.Image,
    box: tuple[int, int, int, int],
    mask: Image.Image,
) -> Image.Image:
    """Вклеивает обработанный фрагмент обратно, соблюдая маску."""
    left, top, right, bottom = box
    canvas = original.convert("RGBA").copy()

    resized = patch.convert("RGBA")
    if resized.size != (right - left, bottom - top):
        resized = resized.resize((right - left, bottom - top), Image.LANCZOS)

    full = canvas.copy()
    full.paste(resized, (left, top))
    return blend(canvas, full, mask)

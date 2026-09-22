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
    """Превращает маску в условное изображение: белое — править, чёрное — оставить.

    Маска здесь **бинаризуется**, и это не косметика. Растушёвка из
    ``refine`` нужна склейке: без неё граница вклейки видна даже при
    идеальном совпадении содержимого. Модели же серый ореол прямо вреден —
    официальная рекомендация Qwen требует резких краёв, чисто белого и чисто
    чёрного, потому что «серые и размытые переходы сбивают модель с толку».

    Насколько вреден, измерено (`docs/research/2026-09-22-maska-kak-alfa.md`):
    при растушёванной маске модель принимает её за альфа-матовку и делает
    белую область **прозрачной** — на дорисовке полей это давало пурпур
    вместо продолжения сцены. С резкой маской непрозрачность держится на ста
    процентах, а изменение вне маски падает с 63 до 58 при той же задаче.

    Поэтому у двух потребителей маски — модели и склейки — теперь разные
    представления одной и той же маски, и получают они их из разных мест:
    склейка берёт ``prepared.mask`` как есть, модель — только через эту
    функцию.
    """
    array = np.asarray(mask.convert("L"))
    binary = np.where(array >= 128, 255, 0).astype(np.uint8)
    return Image.fromarray(np.repeat(binary[..., None], 3, axis=2), mode="RGB")


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
    контекста, и модель дорисовывает в нём что угодно. Прямоугольник обязан
    полностью содержать маску — иначе часть нарисованной пользователем области
    молча выпадёт из правки. Если кратное окно, вмещающее маску и запас, не
    помещается в холст, отдаётся холст целиком; тогда результат может быть не
    кратен ``multiple`` — это осознанный компромисс, а не проглядели: конвейер
    сам округляет условные изображения, а ``stitch`` умеет подгонять размер
    патча под любую сторону.
    """
    array = np.asarray(mask.convert("L"))
    rows = np.flatnonzero(array.any(axis=1))
    columns = np.flatnonzero(array.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None

    height, width = array.shape
    mask_top, mask_bottom = int(rows[0]), int(rows[-1]) + 1
    mask_left, mask_right = int(columns[0]), int(columns[-1]) + 1

    pad_x = int((mask_right - mask_left) * padding)
    pad_y = int((mask_bottom - mask_top) * padding)
    pad_left, pad_right = max(0, mask_left - pad_x), min(width, mask_right + pad_x)
    pad_top, pad_bottom = max(0, mask_top - pad_y), min(height, mask_bottom + pad_y)

    left, right = _snap_span(pad_left, pad_right, mask_left, mask_right, width, multiple)
    top, bottom = _snap_span(pad_top, pad_bottom, mask_top, mask_bottom, height, multiple)
    return left, top, right, bottom


def _snap_span(
    pad_start: int,
    pad_end: int,
    mask_start: int,
    mask_end: int,
    limit: int,
    multiple: int,
) -> tuple[int, int]:
    """Растягивает отрезок до кратной длины, гарантируя, что маска не вылезет наружу.

    Желаемая длина окна — это запрошенный запас, но не короче самой маски:
    запасом обрезать маску нельзя. Если округлённое вверх до кратности окно не
    умещается в холст, окно — это холст целиком (см. докстринг ``region_box``).
    Иначе окно сдвигают в пределах ``[lower, upper]``: правее ``lower`` нельзя —
    маска вылезет слева, левее ``upper`` нельзя — маска вылезет справа.
    ``lower <= upper`` соблюдается всегда, потому что ``needed <= target <=
    limit`` по построению, отдельной защитной проверки не требуется.
    """
    needed = mask_end - mask_start
    desired = max(pad_end - pad_start, needed)
    target = ((desired + multiple - 1) // multiple) * multiple
    if target > limit:
        return 0, limit

    lower = max(0, mask_end - target)
    upper = min(limit - target, mask_start)
    start = min(max(mask_start - (target - needed) // 2, lower), upper)
    return start, start + target


def paste_region(
    original: Image.Image,
    patch: Image.Image,
    box: tuple[int, int, int, int],
) -> Image.Image:
    """Возвращает фрагмент на его место в кадре, без смешивания по маске.

    Нужна отдельно от ``stitch`` для случая, когда склейку с оригиналом
    выключили галочкой «сохранять кадр вне маски»: смешивать тогда нельзя, но
    и отдавать пользователю вырезку вместо кадра — тоже. Масштабирование здесь
    по той же причине, что и в ``stitch``: пайплайн округляет стороны
    условного изображения вниз до кратности 32 и может вернуть патч чуть
    меньше запрошенного прямоугольника.
    """
    left, top, right, bottom = box
    canvas = original.convert("RGBA").copy()

    resized = patch.convert("RGBA")
    if resized.size != (right - left, bottom - top):
        resized = resized.resize((right - left, bottom - top), Image.LANCZOS)

    canvas.paste(resized, (left, top))
    return canvas


def stitch(
    original: Image.Image,
    patch: Image.Image,
    box: tuple[int, int, int, int],
    mask: Image.Image,
) -> Image.Image:
    """Вклеивает обработанный фрагмент обратно, соблюдая маску."""
    canvas = original.convert("RGBA")
    return blend(canvas, paste_region(canvas, patch, box), mask)

def clipped_share(original: Image.Image, generated: Image.Image, mask: Image.Image) -> float:
    """Доля кадра **вне** маски, которую модель изменила, а склейка обрезала.

    Возвращает проценты. Величина нужна, чтобы заметить и назвать вслух один
    конкретный случай: просьба глобальна по смыслу («сделай волосы
    платиновыми»), модель перекрашивает все волосы связно и красиво, а
    склейка обрезает её работу ровно по границе маски — и в кадре остаётся
    резкий шов. Растушёвка тут не спасает: при перепаде яркости в шестьдесят
    уровней двенадцать пикселей перехода дают около четырёх уровней на
    пиксель, что глаз видит прекрасно.

    Дефектом это не является: неприкосновенность кадра вне маски — данное
    обещание, и нарушать его нельзя. Но у случая есть штатный выход — снять
    «Сохранять кадр вне маски», — и пользователь должен узнать о нём тогда,
    когда случай наступил, а не из документации задним числом.

    Сравнение ведётся по порогу, а не по точному равенству: пайплайн
    перерисовывает кадр целиком, и вне маски он никогда не совпадает с
    оригиналом побайтово — совпадает лишь то, что выдаёт ``blend``.
    """
    base = np.asarray(original.convert("RGB")).astype(np.float32)
    patch = generated.convert("RGB")
    if patch.size != original.size:
        patch = patch.resize(original.size, Image.LANCZOS)
    soft = mask.convert("L")
    if soft.size != original.size:
        soft = soft.resize(original.size, Image.LANCZOS)

    outside = np.asarray(soft) == 0
    if not outside.any():
        return 0.0
    delta = np.abs(base - np.asarray(patch).astype(np.float32)).mean(axis=2)
    return float((delta[outside] > 16).mean() * 100)

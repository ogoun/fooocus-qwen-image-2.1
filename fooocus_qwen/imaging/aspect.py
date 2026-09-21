"""Соотношения сторон и приведение размеров кадра.

Канон — таблица из карточки модели. Внутренняя функция пайплайна
``calculate_dimensions`` считает стороны из площади и для 16:9 при
``output_resolution = 2048`` даёт 2720×1536, а не 2752×1536 из карточки.
Расхождение небольшое, но оно проявилось бы как необъяснимое несовпадение
размеров, поэтому размеры передаются в пайплайн явно, а не выводятся им.
"""

from __future__ import annotations

MULTIPLE = 32
REFERENCE_RESOLUTION = 2048

FOLLOW_REFERENCE = "от референса"

# Размеры из карточки модели Qwen-Image-2.1 для 2K.
CANONICAL: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

ASPECT_RATIOS: tuple[str, ...] = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", FOLLOW_REFERENCE)


def snap(value: int) -> int:
    """Приводит размер к ближайшей кратности 32, но не к нулю."""
    return max(MULTIPLE, round(value / MULTIPLE) * MULTIPLE)


def dimensions(ratio: str, output_resolution: int) -> tuple[int | None, int | None]:
    """Размеры кадра для соотношения и целевого разрешения.

    Для ``FOLLOW_REFERENCE`` возвращает пару ``None``: тогда пайплайн сам выведет
    размеры из соотношения сторон последнего условного изображения.
    """
    if ratio == FOLLOW_REFERENCE:
        return None, None

    width, height = CANONICAL[ratio]
    if output_resolution == REFERENCE_RESOLUTION:
        return width, height

    scale = output_resolution / REFERENCE_RESOLUTION
    return snap(int(width * scale)), snap(int(height * scale))


def label(ratio: str, output_resolution: int) -> str:
    """Подпись для выпадающего списка: соотношение и настоящие пиксели."""
    if ratio == FOLLOW_REFERENCE:
        return FOLLOW_REFERENCE
    width, height = dimensions(ratio, output_resolution)
    return f"{ratio} — {width}×{height}"

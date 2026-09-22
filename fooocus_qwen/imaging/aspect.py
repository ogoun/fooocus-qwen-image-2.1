"""Соотношения сторон и приведение размеров кадра.

Канон — таблица из карточки модели. Внутренняя функция пайплайна
``calculate_dimensions`` считает стороны из площади и для 16:9 при
``output_resolution = 2048`` даёт 2720×1536, а не 2752×1536 из карточки.
Расхождение небольшое, но оно проявилось бы как необъяснимое несовпадение
размеров, поэтому размеры передаются в пайплайн явно, а не выводятся им.
"""

from __future__ import annotations

import math

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

def frame_for(size: tuple[int, int], output_resolution: int) -> tuple[int, int]:
    """Кадр площадью ``output_resolution²`` с соотношением сторон ``size``.

    Повторяет ``calculate_dimensions`` из пайплайна (строки 149-156): тот же
    корень из площади, то же округление к ближайшей кратности 32. Нужна там,
    где размер кадра приходится вычислять самим, — а именно когда детальность
    референсов задана отдельно от разрешения пресета. В этом случае
    ``output_resolution`` уходит на референсы, и оставить ``height``/``width``
    пустыми нельзя: пайплайн вывел бы кадр из той же уменьшенной величины и
    кадр съёжился бы вместе с референсами.

    Совпадение с пайплайном проверяется тестом: разойдись формулы, кадр в
    режиме «от референса» поехал бы ровно на величину расхождения.
    """
    width, height = size
    ratio = width / height
    frame_width = math.sqrt(output_resolution * output_resolution * ratio)
    return snap_nearest(frame_width), snap_nearest(frame_width / ratio)


def snap_nearest(value: float) -> int:
    """Округление к ближайшей кратности 32 — ровно как в пайплайне.

    Отличается от ``snap`` только тем, что не поднимает результат до одной
    кратности: пайплайн этого не делает, а расхождение здесь и есть то, что
    ``frame_for`` обязана исключить.
    """
    return round(value / MULTIPLE) * MULTIPLE

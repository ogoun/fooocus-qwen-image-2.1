"""Альфа-канал результата: прозрачность — только там, где её просили.

Декодер Qwen-Image-2.1 всегда отдаёт RGBA, и на непрозрачных картинках часть
пикселей выходит чуть прозрачной. На наших результатах 2026-10-20…24 таких
было от 0.3 до 5 % при альфе 196–254 (24 из 25 снимков; то же отмечает
Kokon, docs/engineering-notes.md: «≈1 % of pixels at alpha 247–254»). Вреда
три:
- крапинки на тёмном фоне просмотрщика;
- файл крупнее;
- такая картинка, отправленная в правку или в референсы, несёт альфу
  дальше, а модель понимает альфу буквально (её прозрачность она охотно
  строит: docs/research/2026-09-22-maska-kak-alfa.md).

Поэтому результат делается непрозрачным, если прозрачность не просили. Сама
прозрачность (не шум) не трогается никогда: на её месте декодер оставляет
пурпур, и снять альфу значило бы показать его вместо честной дыры.
"""

from __future__ import annotations

import re

import numpy as np
from PIL import Image

# Шум — альфа выше половины; ниже — уже прозрачность.
NOISE_FLOOR = 128
# Доля пикселей ниже ``NOISE_FLOOR``, до которой картинка считается
# непрозрачной. На 25 замеренных снимках она была 0.00 %.
STRAY_SHARE = 0.001

# Просьба о прозрачности словами: формат README модели («This is an RGBA image
# with transparency…») и обычные слова — английские и русские: без буста
# промт уходит как написан.
_WANTS = re.compile(
    r"rgba|alpha channel|transparen|no background|without (?:a |any )?background|cut[- ]?out"
    r"|прозрачн|альфа|без фона",
    re.IGNORECASE,
)


def wants_transparency(prompt: str) -> bool:
    return bool(_WANTS.search(prompt or ""))


def _transparent_share(image: Image.Image) -> float:
    alpha = np.asarray(image.getchannel("A"))
    return float((alpha < NOISE_FLOOR).mean()) if alpha.size else 0.0


def has_transparency(image: Image.Image | None) -> bool:
    """Настоящая прозрачность во входе (не шум декодера)."""
    if image is None or "A" not in image.getbands():
        return False
    return _transparent_share(image) > STRAY_SHARE


def drop_stray_alpha(image: Image.Image) -> Image.Image:
    """Непрозрачная по сути картинка — в RGB; настоящая прозрачность остаётся как есть."""
    if "A" not in image.getbands() or _transparent_share(image) > STRAY_SHARE:
        return image
    return image.convert("RGB")


def keep_alpha(prompt: str, inputs) -> bool:
    """Прозрачность нужна: её просит промт или она есть в исходнике или референсе."""
    return wants_transparency(prompt) or any(has_transparency(item) for item in inputs)

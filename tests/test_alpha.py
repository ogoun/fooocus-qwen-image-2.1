"""Альфа-канал результата: шум декодера снимается, просимая прозрачность остаётся."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fooocus_qwen.imaging import alpha


def _rgba(stray=0.01, low=255, hole=0.0, size=100):
    """Непрозрачная картинка с долей ``stray`` пикселей альфы ``low`` и дырой доли ``hole``."""
    array = np.full((size, size, 4), 200, dtype=np.uint8)
    array[..., 3] = 255
    flat = array.reshape(-1, 4)
    flat[: int(stray * flat.shape[0]), 3] = low
    if hole:
        flat[-int(hole * flat.shape[0]):, 3] = 0
    return Image.fromarray(array, mode="RGBA")


def test_decoder_noise_is_dropped():
    # Как на снимках 2026-10-24: ~4 % пикселей с альфой 196–254.
    image = alpha.drop_stray_alpha(_rgba(stray=0.04, low=196))
    assert image.mode == "RGB"


def test_real_transparency_is_kept():
    # Прозрачная дыра — не шум: под ней пурпур декодера, снимать альфу нельзя.
    image = _rgba(hole=0.2)
    assert alpha.drop_stray_alpha(image) is image


def test_rgb_stays_rgb():
    image = Image.new("RGB", (8, 8))
    assert alpha.drop_stray_alpha(image) is image


@pytest.mark.parametrize("prompt", [
    "This is an RGBA image with transparency. A sticker. The image has alpha channel and the background is "
    "transparent.",
    "a cat sticker, transparent background",
    "logo without background",
    "наклейка с котом, прозрачный фон",
    "кот без фона",
])
def test_asked_transparency_is_recognised(prompt):
    assert alpha.wants_transparency(prompt)


@pytest.mark.parametrize("prompt", ["a red fox in the snow", "рыжая лиса в снегу", ""])
def test_ordinary_prompts_do_not_ask_for_it(prompt):
    assert not alpha.wants_transparency(prompt)


def test_a_transparent_input_keeps_the_alpha():
    assert alpha.keep_alpha("remove the hat", [None, _rgba(hole=0.3)])
    assert not alpha.keep_alpha("remove the hat", [None, _rgba(stray=0.02, low=240), Image.new("RGB", (4, 4))])

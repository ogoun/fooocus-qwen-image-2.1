"""Расширение холста и маска для дорисовки."""

import numpy as np
import pytest
from PIL import Image

from fooocus_qwen.imaging import outpaint


def test_plan_grows_only_requested_sides():
    result = outpaint.plan((512, 512), ["right"], 0.5)
    width, height = result.canvas_size
    assert height == 512
    assert width > 512
    assert result.paste_box[0] == 0  # оригинал прижат влево


def test_plan_centres_the_original_when_both_sides_grow():
    result = outpaint.plan((512, 512), ["left", "right"], 0.25)
    left, top, right, bottom = result.paste_box
    width, _ = result.canvas_size
    assert left > 0
    assert abs(left - (width - right)) <= 32


def test_canvas_is_snapped_to_multiples_of_32():
    result = outpaint.plan((500, 300), ["top", "bottom", "left", "right"], 0.3)
    width, height = result.canvas_size
    assert width % 32 == 0 and height % 32 == 0


def test_no_sides_means_no_change():
    result = outpaint.plan((256, 256), [], 0.5)
    assert result.canvas_size == (256, 256)
    assert result.paste_box == (0, 0, 256, 256)


def test_expand_places_the_original_and_masks_only_new_area():
    image = Image.new("RGBA", (64, 64), (200, 30, 30, 255))
    result = outpaint.plan((64, 64), ["right"], 0.5)
    canvas, mask = outpaint.expand(image, result)

    assert canvas.size == result.canvas_size
    assert mask.size == result.canvas_size

    canvas_array = np.asarray(canvas)
    mask_array = np.asarray(mask)
    assert tuple(canvas_array[32, 10]) == (200, 30, 30, 255)
    assert mask_array[32, 10] == 0            # исходная область не правится
    assert mask_array[32, result.canvas_size[0] - 4] == 255  # новая область правится


def test_new_area_is_filled_by_edge_replication():
    # Пустой холст сбивает модель: край изображения должен продолжаться,
    # а не обрываться в чёрное.
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
    image.paste((0, 200, 0, 255), (60, 0, 64, 64))
    canvas, _ = outpaint.expand(image, outpaint.plan((64, 64), ["right"], 0.5))

    array = np.asarray(canvas)
    assert tuple(array[32, 70]) == (0, 200, 0, 255)


def test_unknown_side_is_rejected():
    with pytest.raises(ValueError):
        outpaint.plan((64, 64), ["diagonal"], 0.5)

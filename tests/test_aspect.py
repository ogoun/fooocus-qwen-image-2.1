"""Соотношения сторон и приведение размеров к кратности 32."""

import pytest

from fooocus_qwen.imaging import aspect


def test_canonical_sizes_match_the_model_card():
    assert aspect.CANONICAL["1:1"] == (2048, 2048)
    assert aspect.CANONICAL["16:9"] == (2752, 1536)
    assert aspect.CANONICAL["3:2"] == (2528, 1696)
    assert aspect.CANONICAL["2:3"] == (1696, 2528)


def test_all_canonical_sizes_are_multiples_of_32():
    for width, height in aspect.CANONICAL.values():
        assert width % 32 == 0 and height % 32 == 0


def test_max_quality_reproduces_the_card_exactly():
    # При 2K берём размеры карточки как есть, без пересчёта.
    assert aspect.dimensions("16:9", 2048) == (2752, 1536)
    assert aspect.dimensions("1:1", 2048) == (2048, 2048)


def test_lower_resolution_scales_and_snaps():
    width, height = aspect.dimensions("1:1", 1024)
    assert (width, height) == (1024, 1024)

    width, height = aspect.dimensions("16:9", 1024)
    assert width % 32 == 0 and height % 32 == 0
    assert width > height
    # Площадь держится около квадрата стороны output_resolution.
    assert 0.85 <= (width * height) / (1024 * 1024) <= 1.15


def test_aspect_order_puts_square_first():
    assert aspect.ASPECT_RATIOS[0] == "1:1"
    assert aspect.FOLLOW_REFERENCE in aspect.ASPECT_RATIOS


def test_follow_reference_has_no_dimensions():
    assert aspect.dimensions(aspect.FOLLOW_REFERENCE, 1024) == (None, None)


def test_snap_rounds_to_nearest_multiple_of_32():
    assert aspect.snap(1000) == 992
    assert aspect.snap(1023) == 1024
    assert aspect.snap(16) == 32  # ноль недопустим


def test_unknown_ratio_is_an_error():
    with pytest.raises(KeyError):
        aspect.dimensions("7:5", 1024)


def test_label_shows_actual_pixels():
    assert aspect.label("1:1", 2048) == "1:1 — 2048×2048"

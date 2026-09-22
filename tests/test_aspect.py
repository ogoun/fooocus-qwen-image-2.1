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


def test_frame_for_matches_the_pipeline_formula():
    """Наша формула обязана совпадать с пайплайновой до пикселя.

    ``frame_for`` вычисляет то, что иначе вычислил бы сам пайплайн
    (``calculate_dimensions``, строки 149-156). Расхождение проявилось бы не
    ошибкой, а тихим сдвигом размера кадра в режиме «от референса» ровно на
    величину расхождения — поэтому формула здесь воспроизводится независимо
    и сверяется.
    """
    import math

    def like_the_pipeline(size, resolution):
        ratio = size[0] / size[1]
        width = math.sqrt(resolution * resolution * ratio)
        return round(width / 32) * 32, round((width / ratio) / 32) * 32

    for size in ((1024, 1024), (1408, 1024), (600, 800), (2752, 1536), (97, 241)):
        for resolution in (512, 768, 1024, 1536, 2048):
            assert aspect.frame_for(size, resolution) == like_the_pipeline(size, resolution), (
                f"расхождение на {size} при {resolution}"
            )


def test_frame_for_preserves_the_aspect_ratio():
    width, height = aspect.frame_for((1600, 900), 1024)
    assert abs(width / height - 1600 / 900) < 0.02

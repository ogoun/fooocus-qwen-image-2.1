"""Геометрия ролика H3: сетка кадров, зажим длительности, холст, тайм-коды.

Правила повторяют движок FreeVideo и саму модель (24 к/с, ``17·n + 5``
кадров, стороны кратны 32). Ошибка здесь не роняет ничего сразу — она даёт
ролик не той длины или холст, который движок тихо переделает, а промт с
тайм-кодами разойдётся с картинкой. Поэтому числа проверяются точно.
"""

from __future__ import annotations

import math

import pytest

from fooocus_qwen.video import geometry

# --- кадры и длительность ---


@pytest.mark.parametrize(
    ("seconds", "frames"),
    [(5, 124), (8, 192), (1, 39), (10, 243)],
)
def test_frames_round_up_to_the_17n_plus_5_grid(seconds, frames):
    assert geometry.frames_for(seconds) == frames
    assert frames % geometry.FRAME_UNIT == geometry.FRAME_OFFSET


def test_a_length_already_on_the_grid_stays_put():
    # 124 кадра ровно — 5.1666… с; округление вверх не должно увести на 141.
    assert geometry.frames_for(124 / geometry.FPS) == 124


@pytest.mark.parametrize("bad", [0, -1, math.inf, math.nan])
def test_frames_reject_a_non_positive_or_infinite_length(bad):
    with pytest.raises(ValueError):
        geometry.frames_for(bad)


def test_duration_reports_the_actual_seconds_of_the_grid():
    found = geometry.duration(5)
    assert found.frames == 124
    assert found.seconds == pytest.approx(124 / 24)
    assert found.label == "5.17"


@pytest.mark.parametrize("short", [0.0, 0.5, -3])
def test_too_short_is_clamped_to_the_minimum(short):
    assert geometry.duration(short) == geometry.duration(geometry.MIN_SECONDS)
    assert geometry.duration(short).seconds >= geometry.MIN_SECONDS


def test_too_long_is_clamped_to_the_maximum():
    assert geometry.duration(600) == geometry.duration(geometry.MAX_SECONDS)
    # Сетка округляет вверх: чуть больше минуты, но не десять минут.
    assert geometry.MAX_SECONDS <= geometry.duration(600).seconds < geometry.MAX_SECONDS + 1


# --- холст ---


@pytest.mark.parametrize(
    ("resolution", "aspect", "size"),
    [
        ("480p", "16:9", (864, 480)),
        ("480p", "9:16", (480, 864)),
        ("768p", "16:9", (1344, 768)),
        ("576p", "16:9", (1024, 576)),
    ],
)
def test_canvas_of_the_known_classes(resolution, aspect, size):
    assert geometry.canvas(resolution, aspect) == size


@pytest.mark.parametrize("resolution", list(geometry.RESOLUTIONS))
@pytest.mark.parametrize("aspect", list(geometry.ASPECT_RATIOS))
def test_every_canvas_is_a_multiple_of_32_and_not_tiny(resolution, aspect):
    width, height = geometry.canvas(resolution, aspect)
    assert width % geometry.SIDE_STEP == 0 and height % geometry.SIDE_STEP == 0
    assert min(width, height) >= geometry.MIN_SIDE


def test_vertical_costs_the_same_as_horizontal():
    # Класс — площадь, а не короткая сторона: та же цена по памяти.
    width, height = geometry.canvas("480p", "16:9")
    v_width, v_height = geometry.canvas("480p", "9:16")
    square = geometry.canvas("480p", "1:1")
    assert width * height == v_width * v_height
    assert abs(square[0] * square[1] - width * height) / (width * height) < 0.05


def test_follow_image_takes_the_ratio_of_the_frame():
    assert geometry.canvas("480p", geometry.FOLLOW_IMAGE, (1000, 1000)) == geometry.canvas("480p", "1:1")
    assert geometry.canvas("480p", geometry.FOLLOW_IMAGE, (900, 1600)) == geometry.canvas("480p", "9:16")


def test_follow_image_without_a_frame_is_16_9():
    assert geometry.canvas("480p", geometry.FOLLOW_IMAGE, None) == (864, 480)
    assert geometry.canvas("480p", geometry.FOLLOW_IMAGE, (100, 0)) == (864, 480)


def test_an_extreme_ratio_is_clamped_to_21_9():
    assert geometry.canvas("480p", geometry.FOLLOW_IMAGE, (10_000, 100)) == geometry.canvas("480p", "21:9")
    tall = geometry.canvas("480p", geometry.FOLLOW_IMAGE, (100, 10_000))
    wide = geometry.canvas("480p", "21:9")
    assert tall == (wide[1], wide[0])


def test_an_unknown_resolution_falls_back_to_the_default():
    assert geometry.canvas("1080p", "16:9") == geometry.canvas(geometry.DEFAULT_RESOLUTION, "16:9")


def test_a_side_never_drops_below_256():
    assert geometry._round_side(10) == geometry.MIN_SIDE
    assert geometry._round_side(300) == 288


# --- тайм-коды ---


@pytest.mark.parametrize(
    ("seconds", "code"),
    [(0, "00:00.000"), (124 / 24, "00:05.167"), (65.5, "01:05.500"), (-2, "00:00.000"), (59.9996, "01:00.000")],
)
def test_timecode(seconds, code):
    assert geometry.timecode(seconds) == code

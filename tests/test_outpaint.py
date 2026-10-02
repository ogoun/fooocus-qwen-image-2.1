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


def test_unknown_side_is_rejected():
    with pytest.raises(ValueError):
        outpaint.plan((64, 64), ["diagonal"], 0.5)


def test_plan_does_not_shrink_the_source_when_only_top_grows():
    # 80 не кратно 32: aspect.snap(80) == 64 меньше оригинала, и с округлением
    # к ближайшей кратности cv2.copyMakeBorder получал отрицательный бордюр.
    image = Image.new("RGBA", (80, 48), (5, 5, 5, 255))
    result = outpaint.plan((80, 48), ["top"], 0.4)

    canvas_width, canvas_height = result.canvas_size
    left, top, right, bottom = result.paste_box
    assert canvas_width >= 80 and canvas_height >= 48
    assert left >= 0 and top >= 0
    assert canvas_width - right >= 0 and canvas_height - bottom >= 0

    job = outpaint.prepare(image, result)  # не должно бросать исключение
    assert job.plan == result


def test_plan_does_not_shrink_the_source_when_only_left_grows():
    image = Image.new("RGBA", (80, 48), (5, 5, 5, 255))
    result = outpaint.plan((80, 48), ["left"], 0.4)

    canvas_width, canvas_height = result.canvas_size
    left, top, right, bottom = result.paste_box
    assert canvas_width >= 80 and canvas_height >= 48
    assert left >= 0 and top >= 0
    assert canvas_width - right >= 0 and canvas_height - bottom >= 0

    job = outpaint.prepare(image, result)  # не должно бросать исключение
    assert job.plan == result


def test_a_ratio_grows_only_the_short_axis_and_centres_the_picture():
    result = outpaint.plan_for_ratio((1200, 900), "16:9")
    assert result.canvas_size == (1600, 900) and result.paste_box == (200, 0, 1400, 900)
    tall = outpaint.plan_for_ratio((1200, 900), "9:16")
    assert tall.canvas_size[0] == 1200 and tall.paste_box[1] > 0
    assert outpaint.plan_for_ratio((1600, 900), "16:9").canvas_size == (1600, 900), "уже нужное — без изменений"
    with pytest.raises(ValueError):
        outpaint.plan_for_ratio((10, 10), "5:7")


@pytest.mark.parametrize("size", [(400, 300), (1600, 1200), (3000, 1000)])
def test_the_gray_canvas_is_the_generation_size_and_the_pipeline_keeps_it(size):
    """Пайплайн приводит исходник к площади ``resolution²`` с округлением до 32 —
    это обязано дать ровно размер холста, иначе вклейка разойдётся с кадром."""
    import math

    image = Image.new("RGB", size, (200, 30, 30))
    job = outpaint.prepare(image, outpaint.plan(size, ["left", "right"], 0.3), outpaint.generation_area(1536))
    width, height = job.size
    assert width % 32 == 0 and height % 32 == 0
    assert 1024 * 1024 * 0.85 < width * height <= 2 * 1024 * 1024 * 1.1
    side = math.sqrt(job.resolution ** 2 * width / height)
    assert (round(side / 32) * 32, round(side / (width / height) / 32) * 32) == (width, height)
    array = np.asarray(job.canvas)
    assert tuple(array[height // 2, 2]) == outpaint.FILL, "новая площадь — ровный серый"
    assert tuple(array[height // 2, width // 2]) == (200, 30, 30), "оригинал — на своём месте"


def test_generation_area_stays_within_what_the_lora_saw():
    assert outpaint.generation_area(768) == outpaint.MIN_AREA, "черновик всё равно считается на 1 Мп"
    assert outpaint.generation_area(2048) == outpaint.MAX_AREA


def test_stitch_returns_the_source_scale_and_keeps_the_original_exactly():
    rng = np.random.default_rng(0)
    original = Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))
    plan = outpaint.plan(original.size, ["left", "right"], 0.5)
    job = outpaint.prepare(original, plan)
    drawn = job.canvas.copy()
    drawn.paste((90, 90, 90), (0, 0, 20, drawn.height))  # модель «нарисовала» что-то на новом месте
    result = outpaint.stitch(original, drawn, job)
    assert result.size == plan.canvas_size
    left, top, right, bottom = plan.paste_box
    inner = np.asarray(result)[top:bottom, left + outpaint.FEATHER:right - outpaint.FEATHER]
    truth = np.asarray(original)[:, outpaint.FEATHER:-outpaint.FEATHER]
    assert np.array_equal(inner, truth), "вне полосы растушёвки — пиксели оригинала бит в бит"


def test_stitch_evens_out_a_tone_shift():
    original = Image.new("RGB", (200, 200), (120, 120, 120))
    plan = outpaint.plan(original.size, ["right"], 0.5)
    job = outpaint.prepare(original, plan)
    drawn = Image.new("RGB", job.size, (140, 140, 140))  # модель сдвинула тон на +20
    result = np.asarray(outpaint.stitch(original, drawn, job), dtype=np.int16)
    assert abs(int(result[100, plan.canvas_size[0] - 5, 0]) - 120) <= 2, "новая площадь — в тон оригиналу"


def test_the_prompt_starts_with_the_lora_trigger():
    assert outpaint.prompt() == outpaint.INSTRUCTION
    text = outpaint.prompt("Photograph of a mug on a table", "a teapot on the right")
    assert text.startswith(outpaint.INSTRUCTION + " Scene: Photograph of a mug")
    assert text.endswith("Also include in the prompt: a teapot on the right")


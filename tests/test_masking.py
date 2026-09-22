"""Построение маски, её уточнение и склейка результата с оригиналом.

Главное свойство, ради которого эти тесты существуют: пиксели вне маски после
склейки должны совпадать с исходными побайтово. Пользователь правит участок
портрета и вправе рассчитывать, что остальной кадр не «поплывёт».
"""

import numpy as np
import pytest
from PIL import Image

from fooocus_qwen.imaging import masking


def solid(size, colour, mode="RGBA"):
    return Image.new(mode, size, colour)


def editor_value(size, painted_box=None):
    """Повторяет структуру, которую отдаёт gr.ImageEditor."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if painted_box is not None:
        layer.paste((255, 0, 0, 255), painted_box)
    return {"background": solid(size, (10, 20, 30, 255)), "layers": [layer], "composite": None}


def test_mask_is_built_from_layer_alpha():
    mask = masking.mask_from_editor(editor_value((64, 64), (16, 16, 32, 32)))
    assert mask.mode == "L"
    assert mask.size == (64, 64)
    array = np.asarray(mask)
    assert array[20, 20] == 255
    assert array[5, 5] == 0


def test_several_layers_are_united():
    size = (32, 32)
    first = Image.new("RGBA", size, (0, 0, 0, 0))
    first.paste((255, 0, 0, 255), (0, 0, 8, 8))
    second = Image.new("RGBA", size, (0, 0, 0, 0))
    second.paste((0, 255, 0, 255), (16, 16, 24, 24))
    value = {"background": solid(size, (0, 0, 0, 255)), "layers": [first, second], "composite": None}

    array = np.asarray(masking.mask_from_editor(value))
    assert array[4, 4] == 255
    assert array[20, 20] == 255
    assert array[12, 12] == 0


def test_empty_mask_is_detected():
    assert masking.is_empty(masking.mask_from_editor(editor_value((32, 32))))
    assert not masking.is_empty(masking.mask_from_editor(editor_value((32, 32), (4, 4, 8, 8))))


def test_condition_mask_is_white_where_the_edit_goes():
    mask = masking.mask_from_editor(editor_value((32, 32), (8, 8, 16, 16)))
    condition = masking.as_condition(mask)
    assert condition.mode == "RGB"
    array = np.asarray(condition)
    assert tuple(array[10, 10]) == (255, 255, 255)
    assert tuple(array[2, 2]) == (0, 0, 0)


def test_the_model_never_sees_a_grey_edge_on_the_mask():
    """Только 0 и 255 — ни одного промежуточного значения.

    Модель читает серый ореол как альфа-матовку и делает область
    прозрачной: на дорисовке полей это давало пурпур вместо продолжения
    сцены (измерения — docs/research/2026-09-22-maska-kak-alfa.md).
    Растушёвка остаётся у склейки, где она и нужна.
    """
    mask = masking.mask_from_editor(editor_value((64, 64), (20, 20, 44, 44)))
    feathered = masking.refine(mask, grow=4, feather=8)
    assert set(np.unique(np.asarray(feathered))) - {0, 255}, (
        "растушёвка не дала полутонов — проверка выродилась бы в пустую"
    )

    condition = np.asarray(masking.as_condition(feathered))
    assert set(np.unique(condition)) <= {0, 255}, (
        f"модели ушли полутона: {sorted(set(np.unique(condition)) - {0, 255})[:5]}"
    )


def test_the_blend_still_gets_the_soft_edge():
    """Бинаризация касается только модели, а не склейки.

    Обе стороны обслуживает одна и та же маска, но представления у них
    разные; если бы бинаризация протекла в склейку, вернулась бы резкая
    граница вклейки, ради устранения которой растушёвка и существует.
    """
    mask = masking.mask_from_editor(editor_value((64, 64), (20, 20, 44, 44)))
    feathered = masking.refine(mask, grow=4, feather=8)
    masking.as_condition(feathered)  # не должна испортить исходную маску

    array = np.asarray(feathered)
    assert set(np.unique(array)) - {0, 255}, "склейка осталась без полутонов"

    base = Image.new("RGB", (64, 64), (0, 0, 0))
    patch = Image.new("RGB", (64, 64), (255, 255, 255))
    blended = np.asarray(masking.blend(base, patch, feathered))[..., 0]
    assert set(np.unique(blended)) - {0, 255}, "склейка дала резкий край"


def test_grow_expands_the_mask():
    mask = masking.mask_from_editor(editor_value((64, 64), (28, 28, 36, 36)))
    grown = np.asarray(masking.refine(mask, grow=6, feather=0))
    assert grown[24, 32] == 255  # шесть пикселей выше исходной границы
    assert np.asarray(mask)[24, 32] == 0


def test_feather_softens_the_edge_without_touching_the_core():
    mask = masking.mask_from_editor(editor_value((64, 64), (16, 16, 48, 48)))
    soft = np.asarray(masking.refine(mask, grow=0, feather=5))
    assert soft[32, 32] == 255
    assert 0 < soft[16, 32] < 255


def test_blend_leaves_pixels_outside_the_mask_byte_identical():
    original = Image.new("RGBA", (64, 64), (17, 89, 200, 255))
    generated = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    mask = masking.mask_from_editor(editor_value((64, 64), (16, 16, 32, 32)))

    result = masking.blend(original, generated, mask)

    before = np.asarray(original)
    after = np.asarray(result)
    outside = np.asarray(mask) == 0
    assert np.array_equal(after[outside], before[outside])
    assert tuple(after[20, 20]) == (255, 0, 0, 255)


def test_blend_keeps_far_pixels_exact_even_with_feather():
    original = Image.new("RGBA", (64, 64), (17, 89, 200, 255))
    generated = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
    mask = masking.refine(masking.mask_from_editor(editor_value((64, 64), (24, 24, 40, 40))), feather=4)

    after = np.asarray(masking.blend(original, generated, mask))
    assert tuple(after[2, 2]) == (17, 89, 200, 255)


def test_blend_resizes_generated_to_the_original():
    # Пайплайн округляет размеры до кратности 32 и может вернуть не тот размер.
    original = Image.new("RGBA", (100, 60), (1, 2, 3, 255))
    generated = Image.new("RGBA", (96, 32), (250, 250, 250, 255))
    mask = Image.new("L", (100, 60), 255)

    assert masking.blend(original, generated, mask).size == (100, 60)


def test_region_box_covers_the_mask_with_padding_and_snaps_to_32():
    mask = masking.mask_from_editor(editor_value((256, 256), (100, 100, 120, 120)))
    left, top, right, bottom = masking.region_box(mask, padding=0.5)

    assert left <= 100 and top <= 100 and right >= 120 and bottom >= 120
    assert (right - left) % 32 == 0 and (bottom - top) % 32 == 0
    assert 0 <= left and 0 <= top and right <= 256 and bottom <= 256


def test_region_box_of_empty_mask_is_none():
    assert masking.region_box(masking.mask_from_editor(editor_value((64, 64)))) is None


def test_region_box_never_leaves_the_canvas():
    mask = masking.mask_from_editor(editor_value((64, 64), (0, 0, 8, 8)))
    left, top, right, bottom = masking.region_box(mask, padding=2.0)
    assert (left, top) == (0, 0)
    assert right <= 64 and bottom <= 64


def test_stitch_puts_the_patch_back_and_keeps_the_rest():
    original = Image.new("RGBA", (128, 128), (10, 10, 10, 255))
    mask = masking.mask_from_editor(editor_value((128, 128), (32, 32, 64, 64)))
    box = masking.region_box(mask, padding=0.25)
    patch = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (200, 100, 50, 255))

    result = masking.stitch(original, patch, box, mask)

    array = np.asarray(result)
    assert result.size == (128, 128)
    assert tuple(array[40, 40]) == (200, 100, 50, 255)
    assert tuple(array[2, 2]) == (10, 10, 10, 255)


def test_mask_from_editor_accepts_plain_image():
    # Пользователь мог передать готовую маску файлом, а не нарисовать её.
    mask = masking.mask_from_editor(Image.new("L", (16, 16), 255))
    assert mask.size == (16, 16)
    assert not masking.is_empty(mask)


def test_mask_from_editor_rejects_value_without_background():
    with pytest.raises(ValueError):
        masking.mask_from_editor({"background": None, "layers": [], "composite": None})


def test_region_box_contains_the_mask_on_a_non_multiple_of_32_canvas():
    # Ширина холста (120) не кратна 32: раньше рассчитанное окно могло стать
    # уже маски, и часть закрашенной пользователем области выпадала из правки.
    mask = Image.new("L", (120, 60), 0)
    mask.paste(255, (5, 10, 115, 20))

    left, top, right, bottom = masking.region_box(mask, padding=0.25)

    assert left <= 5 and right >= 115
    assert top <= 10 and bottom >= 20
    assert 0 <= left and 0 <= top and right <= 120 and bottom <= 60


def test_region_box_against_the_right_edge_of_a_non_multiple_canvas():
    mask = Image.new("L", (100, 64), 0)
    mask.paste(255, (90, 20, 100, 44))

    left, top, right, bottom = masking.region_box(mask, padding=0.25)

    assert left <= 90 and right >= 100
    assert top <= 20 and bottom >= 44
    assert 0 <= left and right <= 100 and 0 <= top and bottom <= 64


def test_blend_resizes_mask_to_the_original():
    original = Image.new("RGBA", (64, 64), (1, 2, 3, 255))
    generated = Image.new("RGBA", (64, 64), (250, 250, 250, 255))
    mask = Image.new("L", (32, 32), 255)  # маска другого размера, чем оригинал

    result = masking.blend(original, generated, mask)

    assert result.size == (64, 64)
    assert tuple(np.asarray(result)[32, 32]) == (250, 250, 250, 255)


def test_mask_from_editor_resizes_layer_to_the_background():
    # Слой мог прийти другого размера, чем фон, — редактор их не всегда
    # выравнивает сам.
    background = solid((64, 64), (10, 20, 30, 255))
    layer = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    layer.paste((255, 0, 0, 255), (8, 8, 16, 16))
    value = {"background": background, "layers": [layer], "composite": None}

    mask = masking.mask_from_editor(value)

    assert mask.size == (64, 64)
    assert not masking.is_empty(mask)


def test_clipped_share_counts_only_what_the_blend_threw_away():
    """Считается изменение вне маски, а не вообще любое изменение.

    Внутри маски модель меняет кадр по просьбе пользователя, и это не
    обрезается. Обрезается то, что она сделала снаружи, — ради чего величина
    и нужна: по ней интерфейс предупреждает о шве.
    """
    size = (64, 64)
    original = Image.new("RGB", size, (10, 10, 10))
    mask = Image.new("L", size, 0)
    mask.paste(255, (0, 0, 64, 16))  # верхняя четверть — область правки

    # Модель изменила только внутри маски: обрезать нечего.
    inside_only = original.copy()
    inside_only.paste((250, 250, 250), (0, 0, 64, 16))
    assert masking.clipped_share(original, inside_only, mask) == 0.0

    # Модель изменила и снаружи: три четверти кадра уйдут под нож склейки.
    everywhere = Image.new("RGB", size, (250, 250, 250))
    share = masking.clipped_share(original, everywhere, mask)
    assert 99.0 <= share <= 100.0, share


def test_clipped_share_ignores_imperceptible_drift():
    """Порог, а не точное равенство — иначе величина была бы всегда стопроцентной.

    Пайплайн перерисовывает кадр целиком, и вне маски он не совпадает с
    оригиналом побайтово никогда. Совпадает лишь то, что выдаёт ``blend``.
    """
    size = (64, 64)
    original = Image.new("RGB", size, (100, 100, 100))
    drifted = Image.new("RGB", size, (104, 104, 104))  # +4 уровня, глазу не видно
    mask = Image.new("L", size, 0)
    mask.paste(255, (0, 0, 64, 8))
    assert masking.clipped_share(original, drifted, mask) == 0.0


def test_clipped_share_is_zero_when_the_mask_covers_everything():
    size = (32, 32)
    original = Image.new("RGB", size, (0, 0, 0))
    produced = Image.new("RGB", size, (255, 255, 255))
    assert masking.clipped_share(original, produced, Image.new("L", size, 255)) == 0.0

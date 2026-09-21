"""Извлечение исходного изображения и маски из значения редактора."""

import numpy as np
import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen.engine import generator as gen
from fooocus_qwen.ui import tab_edit


def editor(size=(64, 64), painted=None, colour=(255, 0, 0, 255)):
    background = Image.new("RGBA", size, (12, 34, 56, 255))
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if painted:
        layer.paste(colour, painted)
    composite = background.copy()
    composite.alpha_composite(layer)
    return {"background": background, "layers": [layer], "composite": composite}


def test_mask_mode_takes_clean_background_and_a_mask():
    source, mask = tab_edit.collect(editor(painted=(8, 8, 24, 24)), gen.MASK_MASK)
    assert np.asarray(source.convert("RGBA"))[2, 2].tolist() == [12, 34, 56, 255]
    assert mask is not None
    assert np.asarray(mask)[16, 16] == 255


def test_annotation_mode_takes_the_composite_and_no_mask():
    # Пометки должны попасть в модель как часть изображения.
    source, mask = tab_edit.collect(editor(painted=(8, 8, 24, 24)), gen.MASK_ANNOTATION)
    assert np.asarray(source.convert("RGBA"))[16, 16].tolist() == [255, 0, 0, 255]
    assert mask is None


def test_region_mode_behaves_like_mask_mode():
    source, mask = tab_edit.collect(editor(painted=(8, 8, 24, 24)), gen.MASK_REGION)
    assert mask is not None
    assert np.asarray(source.convert("RGBA"))[2, 2].tolist() == [12, 34, 56, 255]


def test_no_mask_mode_ignores_the_layers():
    source, mask = tab_edit.collect(editor(painted=(8, 8, 24, 24)), gen.MASK_NONE)
    assert mask is None
    assert np.asarray(source.convert("RGBA"))[16, 16].tolist() == [12, 34, 56, 255]


def test_empty_mask_falls_back_to_no_mask():
    # Пользователь выбрал режим маски, но ничего не нарисовал.
    source, mask = tab_edit.collect(editor(), gen.MASK_MASK)
    assert source is not None
    assert mask is None


def test_missing_value_returns_nothing():
    assert tab_edit.collect(None, gen.MASK_MASK) == (None, None)
    assert tab_edit.collect({"background": None, "layers": []}, gen.MASK_MASK) == (None, None)


def test_annotation_palette_has_the_colours_the_blog_uses():
    # Синий, красный и зелёный — цвета из примера с тремя областями.
    for colour in ("#ff0000", "#0000ff", "#00ff00"):
        assert colour in tab_edit.ANNOTATION_COLOURS


# --- различие режимов не должно стираться в будущих правках ---


def test_mask_mode_never_leaks_the_painted_pixels_into_the_source():
    # Если бы режим маски вдруг стал отдавать сведённую картинку вместо чистого
    # фона, эта проверка провалилась бы первой — испорченный кадр уехал бы в
    # модель как «оригинал», который затем considered untouched вне маски.
    source, _ = tab_edit.collect(editor(painted=(8, 8, 24, 24)), gen.MASK_MASK)
    assert np.asarray(source.convert("RGBA"))[16, 16].tolist() == [12, 34, 56, 255]


def test_annotation_mode_never_produces_a_separate_mask():
    # Если бы аннотация вдруг начала отдавать маску, правка получила бы её
    # вторым условным изображением поверх уже вписанных в кадр пометок —
    # runtime-модель увидела бы область дважды и по-разному.
    _, mask = tab_edit.collect(editor(painted=(8, 8, 24, 24)), gen.MASK_ANNOTATION)
    assert mask is None


def test_region_mode_mask_matches_the_painted_area():
    source, mask = tab_edit.collect(editor(size=(64, 64), painted=(8, 8, 24, 24)), gen.MASK_REGION)
    mask_array = np.asarray(mask)
    assert mask_array[16, 16] == 255
    assert mask_array[2, 2] == 0

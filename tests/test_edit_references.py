"""Референсы на вкладке правки: сетка, теги со сдвигом и путь до модели.

При правке перед референсами стоит исходник (``<image1>``), а в режимах
«маска» и «точная область» ещё и маска (``<image2>``). Подписи ячеек, запрос
к модели и сообщение переписывателю обязаны нумеровать одинаково — иначе
промт, написанный по подписям, ссылался бы на маску вместо референса.
"""

from __future__ import annotations

import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.engine import generator as gen
from fooocus_qwen.prompting import boost
from fooocus_qwen.ui import layout, references, tab_edit
from fooocus_qwen.ui.i18n import Localizer
from fooocus_qwen.ui.state import Studio


class _Recorder:
    def __init__(self) -> None:
        self.captured = None

    def generate(self, request, progress=None):
        self.captured = request
        return []


def _build(studio=None):
    studio = studio or Studio(config.AppConfig())
    with gr.Blocks() as demo:
        components = tab_edit.build(studio, Localizer("ru"))
    handlers = {}
    for block_fn in demo.fns.values():
        if block_fn.fn is not None:
            handlers.setdefault(block_fn.fn.__name__, block_fn.fn)
    return studio, components, handlers


def _images(painted=True):
    background = Image.new("RGBA", (64, 64), (10, 20, 30, 255))
    layer = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    if painted:
        layer.paste((255, 0, 0, 255), (8, 8, 24, 24))
    return background, layer


def _refs(*colours):
    return [Image.new("RGB", (32, 32), colour) if colour else None for colour in colours]


def _quiet(*args, **kwargs):
    return None


@pytest.mark.parametrize(
    ("mode", "first"),
    [
        (gen.MASK_MASK, "<image3>"),
        (gen.MASK_REGION, "<image3>"),
        (gen.MASK_NONE, "<image2>"),
        (gen.MASK_ANNOTATION, "<image2>"),
        (None, "<image1>"),
    ],
)
def test_caption_numbering_follows_the_region_mode(mode, first):
    captions = references._captions(_refs("red", "blue"), "ru", mode)
    assert captions[0] == first
    assert int(captions[1][6:-1]) == int(first[6:-1]) + 1


def test_a_single_reference_is_tagged_when_editing():
    """Исходник плюс референс — уже два изображения: теги обязательны."""
    assert references._captions(_refs("red"), "ru", gen.MASK_NONE) == ["<image2>"]
    assert references._captions(_refs("red"), "ru", None) == ["без тега"]


def test_the_grid_stands_left_of_the_brush():
    _studio, components, _handlers = _build()
    slots = components["reference_slots"]
    assert len(slots) == references.MAX_REFERENCES
    grid_column = slots[0].parent.parent.parent
    row = grid_column.parent
    assert layout.REFS_COL in grid_column.elem_classes and layout.RESULT_ROW in row.elem_classes
    children = list(row.children)
    assert children.index(grid_column) < children.index(components["editor"])
    assert len(components["pose_buttons"]) == len(components["sketch_buttons"]) == references.MAX_REFERENCES


def test_changing_the_region_mode_retags_the_cells():
    _studio, _components, handlers = _build()
    tags = handlers["retag"](_refs(None, "red", "blue"), gen.MASK_NONE, "ru")
    assert [tag["value"] for tag in tags[:3]] == [chr(0xA0), "`<image2>`", "`<image3>`"]
    tags = handlers["retag"](_refs(None, "red", "blue"), gen.MASK_MASK, "ru")
    assert [tag["value"] for tag in tags[1:3]] == ["`<image3>`", "`<image4>`"]


def test_filled_cells_reach_the_model_in_order(painter_value):
    recorder = _Recorder()
    studio = Studio(config.AppConfig())
    studio._generator = recorder
    _studio, _components, handlers = _build(studio)
    grid = _refs(None, "red", None, "blue")
    _paths, status = handlers["run"](
        painter_value(*_images()), "вставь кота с <image3>", False, gen.MASK_MASK,
        config.AppConfig().preset, 8, 12, True, -1, grid, "ru", progress=_quiet,
    )
    request = recorder.captured
    assert [image.getpixel((0, 0)) for image in request.references] == [(255, 0, 0), (0, 0, 255)]
    tags = [slot.tag for slot in gen.build_conditions(request)]
    assert tags == ["<image1>", "<image2>", "<image3>", "<image4>"], "исходник, маска, два референса"
    assert "сдвинулись" not in status


def test_an_empty_mask_says_the_reference_tags_moved(painter_value):
    recorder = _Recorder()
    studio = Studio(config.AppConfig())
    studio._generator = recorder
    _studio, _components, handlers = _build(studio)
    _paths, status = handlers["run"](
        painter_value(*_images(painted=False)), "вставь кота", False, gen.MASK_MASK,
        config.AppConfig().preset, 8, 12, True, -1, _refs("red"), "ru", progress=_quiet,
    )
    assert recorder.captured.mask_mode == gen.MASK_NONE
    assert "сдвинулись" in status and "<image2>" in status


def test_the_rewriter_sees_source_and_references_under_the_model_tags(painter_value, monkeypatch):
    recorder = _Recorder()
    studio = Studio(config.AppConfig())
    studio._generator = recorder
    asked = {}

    def boost_prompt(prompt, mode, lang, images=None, tags=None):
        asked.update(images=images, tags=tags)
        return "rewritten", None, ""

    monkeypatch.setattr(studio, "boost_prompt", boost_prompt)
    _studio, _components, handlers = _build(studio)
    handlers["run"](
        painter_value(*_images()), "вставь кота", True, gen.MASK_MASK,
        config.AppConfig().preset, 8, 12, True, -1, _refs("red"), "ru", progress=_quiet,
    )
    assert asked["tags"] == ["<image1>", "<image3>"], "маску переписыватель не видит, но её номер занят"
    assert len(asked["images"]) == 2
    assert recorder.captured.prompt == "rewritten"


def test_user_message_uses_the_given_tags():
    message = boost.build_user_message("вставь кота", 2, ["<image1>", "<image3>"])
    assert message.startswith("Input images: <image1> <image3>")
    assert boost.build_user_message("вставь кота", 1, ["<image1>"]) == "вставь кота"


def test_the_two_sketch_windows_have_different_painters():
    """Окна эскиза двух вкладок — две кисти на одной странице: имена разные."""
    from fooocus_qwen.ui import reference_tools

    assert tab_edit.SKETCH_PAINTER_ID != reference_tools.SKETCH_PAINTER_ID
    _studio, components, _handlers = _build()
    assert components["sketch"].elem_id == tab_edit.SKETCH_PAINTER_ID

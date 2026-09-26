"""Окна инструментов ячейки: выбор позы, «Добавить позу» и эскиз.

Обработчики вызываются настоящие — из собранной вкладки генерации; каталог
поз и свои позы — во временных каталогах, распознавание позы — подставное
(настоящий DWPose проверяют tests/test_poses.py и tools/ui_check.py).
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.poses import detect, library, skeleton
from fooocus_qwen.ui import layout, reference_tools, tab_generate
from fooocus_qwen.ui.i18n import Localizer
from fooocus_qwen.ui.references import MAX_REFERENCES
from fooocus_qwen.ui.state import Studio

N = MAX_REFERENCES


def _pose() -> skeleton.Pose:
    points = {0: (384, 120), 1: (384, 200), 2: (330, 200), 5: (438, 200), 8: (350, 400), 11: (418, 400)}
    return skeleton.Pose(
        tuple((*points[i], 1.0) if i in points else (0.0, 0.0, 0.0) for i in range(skeleton.POINTS)), 768, 768
    )


@pytest.fixture
def poses(monkeypatch, tmp_path):
    catalog, user = tmp_path / "catalog", tmp_path / "user"
    catalog.mkdir()
    for name in ("dance_01", "standing_01"):
        (catalog / f"{name}.json").write_text(_pose().to_json(), encoding="utf-8")
        skeleton.render(_pose()).save(catalog / f"{name}.png")
        library.set_tile(library.PoseEntry(name, catalog, False), Image.new("RGB", (64, 64), "teal"))
    monkeypatch.setattr(config, "POSE_LIBRARY_DIR", catalog)
    # Свои позы — в каталоге генераций: он и подменяется.
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    user = config.user_pose_dir()
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path)
    return catalog, user


def _handlers(studio=None):
    studio = studio or Studio(config.AppConfig())
    with gr.Blocks() as demo:
        components = tab_generate.build(studio, Localizer("ru"))
    found: dict[str, list] = {}
    for block_fn in demo.fns.values():
        found.setdefault(block_fn.fn.__name__ if block_fn.fn else "js", []).append(block_fn)
    return studio, components, found


def _visible(update):
    """Видимость из ответа: ``gr.Column(visible=…)`` — объект, ``gr.update`` — словарь."""
    return update.get("visible") if isinstance(update, dict) else update.visible


def _untouched(update) -> bool:
    return isinstance(update, dict) and set(update) == {"__type__"}


def _event(index):
    return types.SimpleNamespace(_data={"index": index})


def _split(outputs):
    """Выходы записи в сетку: состояние, десять ячеек, десять тегов, строка."""
    return outputs[0], outputs[1:1 + N], outputs[1 + N:1 + 2 * N], outputs[1 + 2 * N], outputs[2 + 2 * N:]


def test_every_cell_has_a_pose_and_a_sketch_button(poses):
    _studio, components, _found = _handlers()
    assert len(components["pose_buttons"]) == N and len(components["sketch_buttons"]) == N
    for pose_button, sketch_button, slot in zip(components["pose_buttons"], components["sketch_buttons"],
                                                 components["reference_slots"]):
        assert layout.REF_POSE in pose_button.elem_classes and layout.REF_SKETCH in sketch_button.elem_classes
        # Значки — в ячейке своей картинки: строка значков соседствует с полем.
        assert pose_button.parent.parent is slot.parent
        assert sketch_button.parent is pose_button.parent


def test_each_pose_button_opens_the_window_for_its_own_cell(poses):
    _studio, _components, found = _handlers()
    openers = [fn.fn for fn in found["open_pose_window"]]
    assert len(openers) == N
    target, window, add_panel, title, _message = openers[6]("ru")
    assert target == 6 and _visible(window) is True and _visible(add_panel) is False
    assert "7" in title


def test_the_window_lists_the_catalogue_then_add_pose(poses):
    _studio, _components, found = _handlers()
    names, tiles, message = found["load_tiles"][0].fn("ru")
    assert names == ["dance_01", "standing_01"]
    assert len(tiles) == 3 and tiles[-1] == (str(reference_tools.ADD_POSE_TILE), "Добавить позу")
    assert tiles[0][0].endswith("dance_01.thumb.jpg") and message == ""


def test_picking_a_pose_puts_its_skeleton_into_the_target_cell(poses):
    _studio, _components, found = _handlers()
    pick = found["pick_pose"][0].fn
    outputs = pick(3, ["dance_01", "standing_01"], [], None, "ru", _event(1))
    grid, slots, tags, status, (window, add_panel, message) = _split(outputs)
    assert grid[3] is not None and all(image is None for i, image in enumerate(grid) if i != 3)
    assert np.asarray(grid[3]).mean() < 40, "в ячейке — скелет на чёрном"
    assert slots[3]["value"] is grid[3] and all("value" not in slot for i, slot in enumerate(slots) if i != 3)
    assert _visible(window) is False and "4" in status


def test_the_last_tile_opens_the_photo_field_and_keeps_the_grid(poses):
    _studio, _components, found = _handlers()
    outputs = found["pick_pose"][0].fn(0, ["dance_01", "standing_01"], [], None, "ru", _event(2))
    grid, slots, _tags, _status, (window, add_panel, message) = _split(outputs)
    assert _untouched(grid) and all(_untouched(slot) for slot in slots), "ячейки не тронуты"
    assert _untouched(window), "окно остаётся открытым"
    assert _visible(add_panel) is True and "фото" in message.lower()


def test_add_pose_saves_it_places_the_skeleton_and_orders_a_tile(poses, monkeypatch):
    catalog, user = poses
    studio = Studio(config.AppConfig())
    fake = types.SimpleNamespace(detect=lambda image: "найдено")
    monkeypatch.setattr(studio, "pose_detector", lambda: fake)
    monkeypatch.setattr(detect, "to_pose", lambda found: _pose())
    _studio, _components, found = _handlers(studio)

    outputs = found["add_pose"][0].fn(Image.new("RGB", (300, 400)), 5, [], None, "ru")
    grid = outputs[0]
    names, tiles, add_panel, message, new_pose = outputs[1 + 2 * N + 1:]
    assert grid[5] is not None
    assert len(names) == 3 and names[-1] == new_pose and new_pose.startswith(library.CUSTOM_PREFIX)
    assert tiles[-2][0].endswith(f"{new_pose}.png"), "пока плитки нет — в окне скелет"
    assert (user / f"{new_pose}.json").exists() and _visible(add_panel) is False

    asked = []

    def run_generation(request, lang, progress=None):
        asked.append(request)
        return [types.SimpleNamespace(image=Image.new("RGB", (1024, 1024), "orange"))], None

    monkeypatch.setattr(studio, "run_generation", run_generation)
    monkeypatch.setattr(studio, "turbo_weights_present", lambda: False)
    names, tiles, message = found["draw_tile"][0].fn(new_pose, "ru", progress=lambda *a, **k: None)
    assert len(asked) == 1 and len(asked[0].references) == 1
    assert tiles[-2][0].endswith(f"{new_pose}.thumb.jpg") and "готова" in message


def test_no_person_on_the_photo_is_said_in_the_window(poses, monkeypatch):
    studio = Studio(config.AppConfig())

    def refuse(image):
        raise detect.NoPersonFound("нет")

    monkeypatch.setattr(studio, "pose_detector", lambda: types.SimpleNamespace(detect=refuse))
    _studio, _components, found = _handlers(studio)
    outputs = found["add_pose"][0].fn(Image.new("RGB", (300, 400)), 0, [], None, "ru")
    assert "не найден человек" in outputs[-2] and outputs[-1] is None
    assert not list(config.user_pose_dir().glob("*.json")), "нераспознанная поза не сохраняется"


def test_sketch_opens_blank_and_accept_puts_the_drawing_into_the_cell(poses, monkeypatch, tmp_path):
    from fooocus_qwen.ui.painter import payload

    monkeypatch.setattr(payload, "upload_root", lambda: tmp_path)
    _studio, _components, found = _handlers()
    target, window, title, value, _message = found["open_sketch_window"][2].fn("ru")
    assert target == 2 and _visible(window) is True and "3" in title
    blank = payload.decode(value)
    assert blank.background.size == (1024, 1024) and np.asarray(blank.background.convert("L")).min() == 255

    layer = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    layer.paste((0, 0, 0, 255), (400, 400, 600, 600))
    drawn = payload.encode(blank.background.convert("RGB"), layer)
    outputs = found["accept_sketch"][0].fn(drawn, 2, [], None, "ru")
    grid, slots, _tags, status, (window, message) = _split(outputs)
    pixels = np.asarray(grid[2].convert("L"))
    assert pixels[500, 500] == 0 and pixels[50, 50] == 255
    assert _visible(window) is False and "3" in status


def test_sketch_without_a_canvas_keeps_the_window(poses):
    _studio, _components, found = _handlers()
    outputs = found["accept_sketch"][0].fn("", 0, [], None, "ru")
    assert _untouched(outputs[-2]), "окно эскиза остаётся открытым"
    assert "пуст" in outputs[-1]

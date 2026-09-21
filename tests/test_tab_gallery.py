"""Вкладка «Галерея»: обработчики обновления списка, выбора и восстановления.

``restore_fields`` сам по себе проверяется в ``test_restore.py``. Здесь —
обвязка вокруг него: как обработчики находят исходный PNG (по выбору в
галерее или по перетащенному файлу), что происходит при чужом PNG и при
полном отсутствии источника, и что кнопка «Обновить» и «Открыть папку»
действительно читают/создают каталог результатов, не трогая модель.
"""

from __future__ import annotations

import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.imaging import metadata
from fooocus_qwen.storage import gallery
from fooocus_qwen.ui import tab_gallery, tab_generate
from fooocus_qwen.ui.i18n import Localizer
from fooocus_qwen.ui.state import Studio

PARAMS = {
    "prompt": "кот в шляпе",
    "prompt_boosted": "a cat wearing a hat",
    "negative_prompt": "blurry",
    "styles": ["sai-anime"],
    "preset": "MaxQuality",
    "seed": 4242,
    "true_cfg_scale": 2.5,
}


def _build_handlers(monkeypatch, tmp_path, studio=None):
    """Тот же приём, что и в ``test_tab_generate._build_handlers``.

    Обработчики — замыкания внутри ``build`` и не экспортируются в словарь
    компонентов, но Gradio хранит исходную функцию каждого события в
    ``Blocks.fns``, откуда её можно достать по имени и вызвать напрямую.
    """
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    if studio is None:
        studio = Studio(config.AppConfig())

    with gr.Blocks() as demo:
        localizer = Localizer(studio.config.lang)
        generate_components = tab_generate.build(studio, localizer)
        tab_gallery.build(studio, localizer, generate_components)

    handlers = {}
    for block_fn in demo.fns.values():
        handlers.setdefault(block_fn.fn.__name__, block_fn.fn)
    return handlers


# --- restore() ---


def test_restore_reads_parameters_from_the_selected_history_path(monkeypatch, tmp_path):
    handlers = _build_handlers(monkeypatch, tmp_path)
    source = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "a.png", PARAMS)

    *fields, details = handlers["restore"](str(source), None)
    assert fields[0] == "кот в шляпе"
    assert fields[4] == "MaxQuality"
    assert details == PARAMS


def test_restore_reads_parameters_from_a_dropped_file(monkeypatch, tmp_path):
    handlers = _build_handlers(monkeypatch, tmp_path)
    source = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "b.png", PARAMS)

    *fields, details = handlers["restore"](None, str(source))
    assert fields[0] == "кот в шляпе"
    assert details == PARAMS


def test_restore_reports_missing_parameters_for_a_foreign_png_without_raising(monkeypatch, tmp_path):
    handlers = _build_handlers(monkeypatch, tmp_path)
    foreign = tmp_path / "foreign.png"
    Image.new("RGB", (8, 8), "blue").save(foreign)

    *fields, details = handlers["restore"](str(foreign), None)
    assert tuple(fields) == ("", "", "", [], "MiddleQuality", -1, 1.0, "1:1")
    assert isinstance(details, dict)
    assert "не найдены" in next(iter(details.values()))


def test_restore_without_any_source_leaves_the_fields_untouched(monkeypatch, tmp_path):
    handlers = _build_handlers(monkeypatch, tmp_path)

    *fields, details = handlers["restore"](None, None)
    # gr.update() без аргументов сериализуется в {"__type__": "update"} — маркер
    # «не менять это поле», а не конкретное значение.
    assert all(field == {"__type__": "update"} for field in fields)
    assert isinstance(details, dict)


# --- refresh_history() / open_outputs() ---


def test_refresh_history_lists_files_from_the_output_dir(monkeypatch, tmp_path):
    handlers = _build_handlers(monkeypatch, tmp_path)

    from datetime import datetime

    path = gallery.next_path(tmp_path, when=datetime(2026, 9, 21, 12, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4)).save(path)

    listed = handlers["refresh_history"]()
    assert listed == [str(path)]


def test_open_outputs_creates_the_directory_and_reports_it(monkeypatch, tmp_path):
    target = tmp_path / "не_созданный_каталог"
    monkeypatch.setattr(config, "OUTPUT_DIR", target)
    monkeypatch.setattr(tab_gallery, "_open_folder", lambda path: None)  # не открываем реальный проводник

    studio = Studio(config.AppConfig())
    with gr.Blocks() as demo:
        localizer = Localizer(studio.config.lang)
        generate_components = tab_generate.build(studio, localizer)
        tab_gallery.build(studio, localizer, generate_components)

    handlers = {}
    for block_fn in demo.fns.values():
        handlers.setdefault(block_fn.fn.__name__, block_fn.fn)

    report = handlers["open_outputs"]()
    assert target.is_dir()
    assert str(target) in next(iter(report.values()))


# --- on_select() ---


def test_on_select_reads_metadata_of_the_chosen_thumbnail(monkeypatch, tmp_path):
    handlers = _build_handlers(monkeypatch, tmp_path)
    source = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "c.png", PARAMS)

    event = gr.SelectData(target=None, data={"index": 0, "value": {"image": {"path": str(source)}}})
    path, details = handlers["on_select"](event)
    assert path == str(source)
    assert details == PARAMS


def test_on_select_on_a_foreign_png_reports_absence_without_raising(monkeypatch, tmp_path):
    handlers = _build_handlers(monkeypatch, tmp_path)
    foreign = tmp_path / "foreign.png"
    Image.new("RGB", (8, 8), "green").save(foreign)

    event = gr.SelectData(target=None, data={"index": 0, "value": {"image": {"path": str(foreign)}}})
    path, details = handlers["on_select"](event)
    assert path == str(foreign)
    assert isinstance(details, dict) and details

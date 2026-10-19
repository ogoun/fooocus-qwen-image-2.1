"""Вкладка «Галерея»: выбор, карточка, повтор параметров и отправка в редактор.

``restore_fields`` сам по себе проверяется в ``test_restore.py``. Здесь —
обвязка вокруг него: как действия находят исходный PNG (выбор в галерее или
выбранный файл), что происходит при чужом PNG и без источника, как выглядит
карточка вместо прежнего сырого JSON и что «Открыть в редакторе» кладёт
картинку в кисть правки — всё без модели.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.imaging import metadata
from fooocus_qwen.storage import gallery
from fooocus_qwen.ui import layout, painter, tab_edit, tab_gallery, tab_generate
from fooocus_qwen.ui.i18n import Localizer
from fooocus_qwen.ui.state import Studio

PARAMS = {
    "prompt": "кот в *шляпе*",
    "prompt_boosted": "a cat wearing a hat",
    "negative_prompt": "blurry",
    "styles": ["sai-anime"],
    "preset": "MaxQuality",
    "steps": 40,
    "seed": 4242,
    "true_cfg_scale": 2.5,
    "width": 1024,
    "height": 768,
    "seconds": 12.5,
}


def _build(monkeypatch, tmp_path, with_editor=False):
    """Собирает вкладку и достаёт обработчики по имени функции.

    Тот же приём, что и в ``test_tab_generate``: обработчики — замыкания
    внутри ``build``, но Gradio хранит исходную функцию каждого события в
    ``Blocks.fns``.
    """
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path)
    studio = Studio(config.AppConfig())

    with gr.Blocks() as demo:
        localizer = Localizer(studio.config.lang)
        with gr.Tabs() as tabs:
            with gr.Tab("g", id=layout.TAB_GENERATE):
                generate_components = tab_generate.build(studio, localizer)
            edit_components = None
            if with_editor:
                with gr.Tab("e", id=layout.TAB_EDIT):
                    edit_components = tab_edit.build(studio, localizer)
            with gr.Tab("l", id=layout.TAB_GALLERY):
                components = tab_gallery.build(
                    studio, localizer, generate_components,
                    edit_components=edit_components, tabs=tabs if with_editor else None,
                )

    handlers = {}
    for block_fn in demo.fns.values():
        if block_fn.fn is not None:
            handlers.setdefault(block_fn.fn.__name__, block_fn.fn)
    return handlers, components


def _select(path):
    return gr.SelectData(target=None, data={"index": 0, "value": {"image": {"path": str(path)}}})


# --- повтор параметров --------------------------------------------------------


def test_reuse_takes_the_parameters_of_the_selected_picture(monkeypatch, tmp_path):
    handlers, _ = _build(monkeypatch, tmp_path)
    source = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "a.png", PARAMS)

    *fields, message = handlers["restore"](str(source), "ru")
    assert tuple(fields) == tab_gallery.restore_fields(PARAMS)
    assert "a.png" in message


def test_a_chosen_file_restores_the_same_way(monkeypatch, tmp_path):
    handlers, _ = _build(monkeypatch, tmp_path)
    source = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "b.png", PARAMS)

    *fields, _message = handlers["restore_from_file"](str(source), "ru")
    assert tuple(fields) == tab_gallery.restore_fields(PARAMS)


def test_a_foreign_png_changes_nothing_and_says_so(monkeypatch, tmp_path):
    handlers, _ = _build(monkeypatch, tmp_path)
    foreign = tmp_path / "foreign.png"
    Image.new("RGB", (8, 8), "green").save(foreign)

    *fields, message = handlers["restore"](str(foreign), "ru")
    assert all(field == gr.update() for field in fields)
    assert message


def test_nothing_selected_changes_nothing(monkeypatch, tmp_path):
    handlers, _ = _build(monkeypatch, tmp_path, with_editor=True)
    *fields, message, switch = handlers["restore"](None, "ru")
    assert all(field == gr.update() for field in fields)
    assert "Выберите" in message and switch == gr.update()


def test_the_answer_always_matches_the_outputs(monkeypatch, tmp_path):
    """Лишнее значение Gradio отвергает вместе со всем ответом."""
    for with_editor in (False, True):
        handlers, _ = _build(monkeypatch, tmp_path, with_editor=with_editor)
        expected = tab_gallery.RESTORED_FIELDS + 1 + (1 if with_editor else 0)
        assert len(handlers["restore"](None, "ru")) == expected


def test_reuse_leads_to_the_generate_tab(monkeypatch, tmp_path):
    """Параметры легли во вкладку генерации — туда же и человека."""
    handlers, _ = _build(monkeypatch, tmp_path, with_editor=True)
    source = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "c.png", PARAMS)
    *_fields, _message, switch = handlers["restore"](str(source), "ru")
    assert switch.selected == layout.TAB_GENERATE


# --- открыть в редакторе ------------------------------------------------------


def test_open_in_editor_loads_the_picture_into_the_painter(monkeypatch, tmp_path, painter_value):
    handlers, _ = _build(monkeypatch, tmp_path, with_editor=True)
    picture = tmp_path / "d.png"
    Image.new("RGB", (40, 30), (200, 10, 10)).save(picture)

    raw, _message, switch = handlers["send_to_editor"](str(picture), "ru")

    canvas = painter.decode(raw)
    assert canvas.background.size == (40, 30)
    assert canvas.layer is None, "в редактор картинка приходит без чужих пометок"
    assert json.loads(raw)["origin"] == "server"
    assert switch.selected == layout.TAB_EDIT


def test_without_the_editor_the_button_is_hidden(monkeypatch, tmp_path):
    """Вкладку собирают и отдельно (тесты); кнопке без адресата не место."""
    handlers, _ = _build(monkeypatch, tmp_path)
    assert "send_to_editor" not in handlers


# --- карточка -----------------------------------------------------------------


def test_the_card_is_readable_text_not_json(monkeypatch, tmp_path):
    handlers, _ = _build(monkeypatch, tmp_path)
    source = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "e.png", PARAMS)

    path, card = handlers["on_select"](None, "ru", _select(source))
    assert path == str(source)
    assert "{" not in card, "карточка — текст, а не JSON"
    for expected in ("Промт", "4242", "1024×768", "MaxQuality", "40 шагов", "e\\.png"):
        assert expected in card, expected


def test_user_text_in_the_card_is_not_markup(monkeypatch, tmp_path):
    """Звёздочки в промте — часть промта, а не курсив."""
    card = tab_gallery.describe(PARAMS, tmp_path / "f.png", "ru")
    assert "\\*шляпе\\*" in card


def test_a_foreign_png_gets_a_card_that_says_so(monkeypatch, tmp_path):
    handlers, _ = _build(monkeypatch, tmp_path)
    foreign = tmp_path / "foreign.png"
    Image.new("RGB", (8, 8), "green").save(foreign)
    _path, card = handlers["on_select"](None, "ru", _select(foreign))
    assert "foreign" in card and "{" not in card


def test_the_card_speaks_the_current_language(monkeypatch, tmp_path):
    card = tab_gallery.describe(PARAMS, tmp_path / "g.png", "en")
    assert "Prompt" in card and "40 steps" in card and "Промт" not in card


# --- список и папка -----------------------------------------------------------


def test_refresh_lists_files_from_the_output_dir(monkeypatch, tmp_path):
    _handlers, components = _build(monkeypatch, tmp_path)

    from datetime import datetime

    path = gallery.next_path(tmp_path, when=datetime(2026, 9, 21, 12, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4)).save(path)

    assert components["refresh"]() == [str(path)]


def test_open_outputs_creates_the_directory_and_reports_it(monkeypatch, tmp_path, never_open_a_file_manager):
    handlers, _ = _build(monkeypatch, tmp_path)
    target = tmp_path / "не_созданный_каталог"
    monkeypatch.setattr(config, "OUTPUT_DIR", target)

    report = handlers["open_outputs"]("ru")
    assert target.is_dir()
    assert str(target) in report
    assert never_open_a_file_manager == [target]


# --- ролики вкладки Video ---

VIDEO_PARAMS = {
    "kind": "video",
    "prompt": "кот идёт по крыше",
    "mode": "i2v",
    "resolution": "768p",
    "aspect": "9:16",
    "width": 768,
    "height": 1344,
    "seconds": 5.167,
    "requested_seconds": 5.0,
    "frames": 124,
    "seed": 77,
    "quality": "high",
    "seconds_taken": 312.4,
}


def _video(tmp_path, parameters=VIDEO_PARAMS, name="clip.mp4"):
    path = tmp_path / name
    path.write_bytes(b"mp4")
    if parameters is not None:
        path.with_suffix(".json").write_text(json.dumps(parameters, ensure_ascii=False), encoding="utf-8")
    return path


def _build_with_video(monkeypatch, tmp_path):
    """Галерея вместе с вкладками генерации и Video, как в приложении."""
    from fooocus_qwen.ui import tab_video

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path)
    monkeypatch.setattr(config, "VIDEO_LORA_DIR", tmp_path / "loras")
    studio = Studio(config.AppConfig())
    with gr.Blocks() as demo:
        localizer = Localizer(studio.config.lang)
        with gr.Tabs() as tabs:
            with gr.Tab("g", id=layout.TAB_GENERATE):
                generate_components = tab_generate.build(studio, localizer)
            with gr.Tab("v", id=layout.TAB_VIDEO):
                video_components = tab_video.build(studio, localizer)
            with gr.Tab("l", id=layout.TAB_GALLERY):
                tab_gallery.build(studio, localizer, generate_components, tabs=tabs,
                                  video_components=video_components)
    handlers = {}
    for block_fn in demo.fns.values():
        if block_fn.fn is not None:
            handlers.setdefault(block_fn.fn.__name__, block_fn.fn)
    return handlers, video_components


def test_the_card_of_a_video_comes_from_its_json(monkeypatch, tmp_path):
    handlers, _ = _build(monkeypatch, tmp_path)
    video = _video(tmp_path)

    path, card = handlers["on_select"](None, "en", _select(video))
    assert path == str(video)
    assert "{" not in card
    for expected in ("кот идёт по крыше", "i2v", "768×1344", "124 frames", "77", "high", "312\\.4 s", "clip\\.mp4"):
        assert expected in card, expected


def test_a_video_is_found_by_the_select_event_alone(monkeypatch, tmp_path):
    # Событие выбора ролика в галерее несёт {"video": {...}}, а не {"image": {...}}.
    handlers, _ = _build(monkeypatch, tmp_path)
    video = _video(tmp_path)
    event = gr.SelectData(target=None, data={"index": 0, "value": {"video": {"path": str(video)}}})
    path, card = handlers["on_select"](None, "en", event)
    assert path == str(video) and "124 frames" in card


def test_a_video_without_parameters_says_so(monkeypatch, tmp_path):
    video = _video(tmp_path, parameters=None)
    card = tab_gallery.describe(tab_gallery.read_parameters(video), video, "en")
    assert "clip" in card and "{" not in card


def test_restore_video_fields_takes_the_values_of_the_video():
    assert tab_gallery.restore_video_fields(VIDEO_PARAMS) == (
        "кот идёт по крыше", "i2v", "768p", "9:16", 124 / 24, "high", 77,
    )


@pytest.mark.parametrize(
    "garbage",
    [
        {},
        {"mode": "v2v", "resolution": "8k", "aspect": "5:1", "quality": "ultra", "seed": "x", "frames": "many"},
        {"mode": ["i2v"], "resolution": ["768p"], "aspect": {"a": 1}, "quality": ["high"], "seed": None,
         "frames": -48, "prompt": 42},
        {"frames": 0, "requested_seconds": 9999},
    ],
)
def test_restore_video_fields_with_garbage_gives_defaults(garbage):
    from fooocus_qwen.video import geometry, prompt_h3
    from fooocus_qwen.video.request import DEFAULT_QUALITY

    prompt, mode, resolution, aspect, seconds, quality, seed = tab_gallery.restore_video_fields(garbage)
    assert isinstance(prompt, str)
    assert mode == prompt_h3.MODE_T2V
    assert resolution == geometry.DEFAULT_RESOLUTION
    assert aspect == geometry.DEFAULT_ASPECT
    assert quality == DEFAULT_QUALITY
    assert seed == -1
    assert geometry.MIN_SECONDS <= seconds <= geometry.MAX_SECONDS, "ползунок не примет значение вне диапазона"


def test_video_restore_outputs_follow_the_field_order():
    components = {name: f"<{name}>" for name in (*tab_gallery.VIDEO_FIELDS, "extra")}
    assert tab_gallery.video_restore_outputs(components) == [f"<{name}>" for name in tab_gallery.VIDEO_FIELDS]
    assert tab_gallery.video_restore_outputs(None) == []
    assert len(tab_gallery.VIDEO_FIELDS) == len(tab_gallery.restore_video_fields({}))


def test_reuse_of_a_video_fills_the_video_tab(monkeypatch, tmp_path):
    handlers, _video_components = _build_with_video(monkeypatch, tmp_path)
    video = _video(tmp_path)

    answer = handlers["restore"](str(video), "en")
    images = answer[:tab_gallery.RESTORED_FIELDS]
    fields = answer[tab_gallery.RESTORED_FIELDS:-2]
    message, switch = answer[-2:]
    assert all(field == gr.update() for field in images), "поля картинок ролик не трогает"
    assert tuple(fields) == tab_gallery.restore_video_fields(VIDEO_PARAMS)
    assert "clip.mp4" in message
    assert switch.selected == layout.TAB_VIDEO


def test_reuse_of_a_picture_leaves_the_video_tab_alone(monkeypatch, tmp_path):
    handlers, _ = _build_with_video(monkeypatch, tmp_path)
    source = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "p.png", PARAMS)

    answer = handlers["restore"](str(source), "en")
    video_fields = answer[tab_gallery.RESTORED_FIELDS:tab_gallery.RESTORED_FIELDS + len(tab_gallery.VIDEO_FIELDS)]
    assert tuple(answer[:tab_gallery.RESTORED_FIELDS]) == tab_gallery.restore_fields(PARAMS)
    assert all(field == gr.update() for field in video_fields)
    assert answer[-1].selected == layout.TAB_GENERATE


def test_every_answer_with_the_video_tab_matches_the_outputs(monkeypatch, tmp_path):
    handlers, _ = _build_with_video(monkeypatch, tmp_path)
    expected = tab_gallery.RESTORED_FIELDS + len(tab_gallery.VIDEO_FIELDS) + 2
    broken = _video(tmp_path, parameters=None, name="bare.mp4")
    for source in (None, str(broken), str(_video(tmp_path))):
        assert len(handlers["restore"](source, "en")) == expected, source


def test_a_video_without_the_video_tab_restores_nothing(monkeypatch, tmp_path):
    handlers, _ = _build(monkeypatch, tmp_path)
    *fields, message = handlers["restore"](str(_video(tmp_path)), "en")
    assert len(fields) == tab_gallery.RESTORED_FIELDS
    assert all(field == gr.update() for field in fields)
    assert message

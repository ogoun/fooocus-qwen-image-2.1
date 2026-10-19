"""Вкладка Video: сборка, видимость входов, референсы, подписи и «В видео».

Движок видео не запускается: ``Studio`` создаётся лениво, как в
``test_tab_generate``, а ``run_video`` и ``video_status`` там, где они нужны,
подменяются записью вызова. Обработчики — замыкания внутри ``build``; их
достают из ``Blocks.fns`` по имени функции.
"""

from __future__ import annotations

import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.ui import layout, tab_video
from fooocus_qwen.ui.i18n import Localizer
from fooocus_qwen.ui.state import Studio
from fooocus_qwen.video import prompt_h3
from fooocus_qwen.video.engine import EngineStatus
from fooocus_qwen.video.request import Reference


def _no_progress(*_args, **_kwargs):
    pass


@pytest.fixture
def built(monkeypatch, tmp_path):
    """Собранная вкладка: (studio, обработчики по имени, компоненты)."""
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path / "prompts")
    monkeypatch.setattr(config, "VIDEO_LORA_DIR", tmp_path / "loras")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "VIDEO_RUNS_DIR", tmp_path / "runs")
    studio = Studio(config.AppConfig())

    with gr.Blocks() as demo:
        localizer = Localizer(studio.config.lang)
        components = tab_video.build(studio, localizer)

    handlers = {}
    for block_fn in demo.fns.values():
        if block_fn.fn is not None:
            handlers.setdefault(block_fn.fn.__name__, block_fn.fn)
    return studio, handlers, components


# --- сборка ---


def test_the_tab_builds_without_the_engine(built):
    studio, handlers, components = built
    assert studio._video is None, "сборка вкладки не запускает драйвер"
    for name in ("prompt", "mode", "resolution", "aspect", "seconds", "quality", "seed", "first", "last",
                 "references", "result", "status", "engine_note", "install_button"):
        assert name in components, name
    for name in ("run", "stop", "update_size", "change_mode", "reference_note", "save", "load"):
        assert name in handlers, name


def test_engine_state_shows_the_install_button_only_when_needed(built, monkeypatch):
    studio, _handlers, components = built
    monkeypatch.setattr(studio, "video_status", lambda: EngineStatus(True, False, "FreeVideo setup is incomplete"))
    note, button = components["engine_state"]("en")
    assert note["visible"] is True and "FreeVideo setup is incomplete" in note["value"]
    assert button["visible"] is True

    monkeypatch.setattr(studio, "video_status", lambda: EngineStatus(True, True))
    note, button = components["engine_state"]("en")
    assert note["visible"] is False and button["visible"] is False


def test_run_without_a_prompt_does_not_start_anything(built, monkeypatch):
    studio, handlers, _components = built
    monkeypatch.setattr(studio, "run_video", lambda *_a, **_k: pytest.fail("nothing to film"))
    answer = handlers["run"]("  ", "", False, prompt_h3.MODE_T2V, "480p", "16:9", 5.0, -1, "light",
                             None, None, None, [], 1.0, "en", progress=_no_progress)
    assert "Describe the video" in answer[1]


def test_run_hands_the_request_to_the_studio(built, monkeypatch, tmp_path):
    studio, handlers, _components = built
    asked = []
    video = tmp_path / "v.mp4"

    def run_video(request, lang, on_event=None):
        asked.append(request)
        return video, {"seed": 3, "seconds": 5.17, "width": 864, "height": 480, "seconds_taken": 60.0,
                       "quality": "draft"}, None

    monkeypatch.setattr(studio, "run_video", run_video)
    result, line, last_result, finish, _boosted = handlers["run"](
        "a cat walks", "", False, prompt_h3.MODE_T2V, "768p", "9:16", 8.0, 3, "draft",
        None, None, None, [], 1.0, "en", progress=_no_progress,
    )
    (request,) = asked
    assert (request.prompt, request.mode, request.resolution, request.aspect) == ("a cat walks", "t2v", "768p", "9:16")
    assert (request.seconds, request.seed, request.quality) == (8.0, 3, "draft")
    assert request.references == ()
    assert result == last_result == str(video)
    assert "seed 3" in line
    assert finish["visible"] is True, "черновик можно довести до финала"


def test_a_failure_is_a_status_line(built, monkeypatch):
    studio, handlers, _components = built
    monkeypatch.setattr(studio, "run_video", lambda *_a, **_k: (None, {}, "Video stopped."))
    result, line, last_result, _finish, _boosted = handlers["run"](
        "a cat", "", False, prompt_h3.MODE_T2V, "480p", "16:9", 5.0, -1, "light",
        None, None, None, [], 1.0, "en", progress=_no_progress,
    )
    assert "Video stopped." in line
    assert result == gr.update() and last_result == gr.update()


def test_save_and_load_keep_the_video_settings(built):
    _studio, handlers, _components = built
    handlers["save"]("ролик", "a cat", prompt_h3.MODE_I2V, "768p", "9:16", 8.0, "high", 11, "ru")
    text, mode, resolution, aspect, seconds, quality, seed, message = handlers["load"]("ролик", "ru")
    assert (text, mode, resolution, aspect, seconds, quality, seed) == (
        "a cat", "i2v", "768p", "9:16", 8.0, "high", 11)
    assert "ролик" in message


# --- видимость и референсы ---


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (prompt_h3.MODE_T2V, (False, False, False, False)),
        (prompt_h3.MODE_I2V, (True, False, False, False)),
        (prompt_h3.MODE_FLF2V, (True, True, False, False)),
        (prompt_h3.MODE_REF, (False, False, True, True)),
    ],
)
def test_only_the_inputs_of_the_mode_are_visible(mode, expected):
    assert tuple(update["visible"] for update in tab_video.visibility(mode)) == expected


class _Upload:
    """Файл поля ``gr.File``: путь — в атрибуте, как у объектов Gradio."""

    def __init__(self, name=None, path=None) -> None:
        if name is not None:
            self.name = name
        if path is not None:
            self.path = path


def test_references_take_their_kind_from_the_extension():
    files = ["face.PNG", _Upload(name="walk.mp4"), _Upload(path="voice.wav"), "notes.txt", None, _Upload(), ""]
    assert tab_video.references_from(files) == (
        Reference("image", "face.PNG"), Reference("video", "walk.mp4"), Reference("audio", "voice.wav"),
    )
    assert tab_video.references_from(None) == ()


def test_reference_note_numbers_in_engine_order(monkeypatch):
    monkeypatch.setattr(tab_video, "_has_audio", lambda path: path.endswith("talk.mp4"))
    note = tab_video.reference_note(["voice.wav", "talk.mp4", "a.png", "silent.mp4", "b.jpg"], "en")
    assert note.startswith("Labels for the script:")
    expected = [
        "`<Picture 1>` — a.png", "`<Picture 2>` — b.jpg",
        "`<Video 1>` + `<Audio 1>` — talk.mp4", "`<Video 2>` — silent.mp4", "`<Audio 2>` — voice.wav",
    ]
    positions = [note.index(item) for item in expected]
    assert positions == sorted(positions)


def test_reference_note_agrees_with_the_engine_labels(monkeypatch):
    # Подпись в интерфейсе и метки в промте — одна нумерация.
    from fooocus_qwen.video.request import engine_references

    files = ["voice.wav", "talk.mp4", "a.png"]
    monkeypatch.setattr(tab_video, "_has_audio", lambda _path: True)
    note = tab_video.reference_note(files, "en")
    ordered = engine_references(tab_video.references_from(files))
    labels = prompt_h3.reference_labels([item.kind for item in ordered], [True])
    for label in labels:
        assert f"`{label}`" in note


def test_an_unreadable_video_gets_no_audio_label(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    note = tab_video.reference_note([str(broken)], "en")
    assert "`<Video 1>` — broken.mp4" in note and "Audio" not in note


def test_no_references_explain_the_limits():
    assert "Up to 9 images" in tab_video.reference_note([], "en")


# --- подпись размера ---


def test_size_note_shows_the_canvas_frames_and_actual_seconds():
    note = tab_video.describe_size(prompt_h3.MODE_T2V, "480p", "16:9", 5.0, None, None, "en")
    assert "864×480" in note and "124" in note and "5.17" in note
    assert "parts" not in note


def test_size_note_follows_the_frame_of_the_mode():
    tall = Image.new("RGB", (900, 1600))
    following = tab_video.describe_size(prompt_h3.MODE_I2V, "480p", "image", 5.0, tall, None, "en")
    assert "480×864" in following
    # Кадр от другого режима холст не задаёт.
    ignored = tab_video.describe_size(prompt_h3.MODE_T2V, "480p", "image", 5.0, tall, None, "en")
    assert "864×480" in ignored
    only_last = tab_video.describe_size(prompt_h3.MODE_FLF2V, "480p", "image", 5.0, None, tall, "en")
    assert "480×864" in only_last


def test_a_long_video_says_it_comes_in_parts():
    note = tab_video.describe_size(prompt_h3.MODE_T2V, "480p", "16:9", 20.0, None, None, "en")
    assert "parts" in note


def test_an_empty_slider_does_not_break_the_note():
    assert "39" in tab_video.describe_size(prompt_h3.MODE_T2V, "480p", "16:9", None, None, None, "en")


# --- «В видео» с вкладок изображений ---


@pytest.fixture
def picture(tmp_path):
    path = tmp_path / "result.png"
    Image.new("RGB", (40, 30), (10, 20, 200)).save(path)
    return str(path)


def test_from_text_mode_the_picture_becomes_the_first_frame(picture):
    mode, first, last, references, message = tab_video.send_to_video(
        [(picture, None)], None, prompt_h3.MODE_T2V, None, None, "en")
    assert mode == prompt_h3.MODE_I2V
    assert first["visible"] is True and first["value"].size == (40, 30)
    assert last == references == gr.update()
    assert "first frame" in message


def test_in_image_mode_the_mode_stays(picture):
    mode, first, *_rest = tab_video.send_to_video([picture], picture, prompt_h3.MODE_I2V, None, None, "en")
    assert mode == prompt_h3.MODE_I2V and first["visible"] is True


def test_with_a_first_frame_the_picture_becomes_the_last(picture):
    mode, first, last, references, message = tab_video.send_to_video(
        [picture], None, prompt_h3.MODE_FLF2V, Image.new("RGB", (8, 8)), None, "en")
    assert mode == first == references == gr.update()
    assert last["visible"] is True and last["value"].size == (40, 30)
    assert "last frame" in message


def test_first_and_last_mode_without_a_first_frame_fills_the_first(picture):
    mode, first, last, _references, _message = tab_video.send_to_video(
        [picture], None, prompt_h3.MODE_FLF2V, None, None, "en")
    assert mode == prompt_h3.MODE_FLF2V
    assert first["visible"] is True and last == gr.update()


def test_in_reference_mode_the_picture_is_added_to_the_list(picture):
    mode, first, last, references, message = tab_video.send_to_video(
        [picture], None, prompt_h3.MODE_REF, None, ["a.png", _Upload(path="b.mp4"), None], "en")
    assert mode == first == last == gr.update()
    assert references["value"] == ["a.png", "b.mp4", picture]
    assert references["visible"] is True
    assert "references" in message


def test_nothing_to_send(picture):
    answer = tab_video.send_to_video([], None, prompt_h3.MODE_T2V, None, None, "en")
    assert answer[:4] == (gr.update(),) * 4
    assert answer[4] == "Nothing to send"


def test_open_video_tab_switches_to_the_video_tab():
    assert tab_video.open_video_tab().selected == layout.TAB_VIDEO

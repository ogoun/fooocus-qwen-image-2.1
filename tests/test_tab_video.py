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
from fooocus_qwen.ui import layout, tab_video, video_refs
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
    for name in ("run", "stop", "update_size", "change_mode", "refresh", "source_loaded", "save", "load"):
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
                             None, None, None, [], 1.0, False, "multiplier", 2.0, 1920, 1088, 2.0, "en", progress=_no_progress)
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
        None, None, None, [], 1.0, False, "multiplier", 2.0, 1920, 1088, 2.0, "en", progress=_no_progress,
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
        None, None, None, [], 1.0, False, "multiplier", 2.0, 1920, 1088, 2.0, "en", progress=_no_progress,
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
        (prompt_h3.MODE_T2V, (False, False)),
        (prompt_h3.MODE_I2V, (True, False)),
        (prompt_h3.MODE_FLF2V, (True, True)),
        (prompt_h3.MODE_REF, (False, False)),
        (prompt_h3.MODE_V2V, (False, False)),
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


def _tiles(images=(), videos=(), audios=()):
    """Значения плиток по порядку: 9 картинок, 3 видео, 3 звука."""
    def pad(values, count):
        return [*values, *[None] * (count - len(values))]

    return pad(list(images), 9) + pad(list(videos), 3) + pad(list(audios), 3)


def test_tiles_are_labelled_in_engine_order():
    values = _tiles(["a.png", None, "b.jpg"], ["talk.mp4", "silent.mp4"], ["voice.wav"])
    tags, source = video_refs.labels(prompt_h3.MODE_REF, None, values, has_audio=lambda path: path == "talk.mp4")
    assert tags["image"][:3] == ["<Picture 1>", "", "<Picture 2>"]
    assert tags["video"][:2] == ["<Audio 1> + <Video 1>", "<Video 2>"]
    assert tags["audio"][0] == "<Audio 2>"
    assert source == ""


def test_tile_labels_agree_with_the_engine_labels():
    # Подпись плитки и метки в промте — одна нумерация.
    from fooocus_qwen.video.request import engine_references

    values = _tiles(["a.png"], ["talk.mp4"], ["voice.wav"])
    tags, _source = video_refs.labels(prompt_h3.MODE_REF, None, values, has_audio=lambda _path: True)
    ordered = engine_references(tab_video.references_from(values))
    labels = prompt_h3.reference_labels([item.kind for item in ordered], [True])
    shown = " ".join(tag for kind in tags.values() for tag in kind)
    for label in labels:
        assert label in shown


def test_in_video_to_video_the_source_is_video_1():
    values = _tiles(["fox.png"], ["ignored.mp4"], ["voice.wav"])
    tags, source = video_refs.labels(prompt_h3.MODE_V2V, "source.mp4", values, has_audio=lambda _path: True)
    assert source == "<Audio 1> + <Video 1>"
    assert tags["video"] == ["", "", ""], "плитки видео в правке ролика не участвуют"
    assert tags["image"][0] == "<Picture 1>" and tags["audio"][0] == "<Audio 2>"


def test_one_empty_tile_per_kind_is_offered():
    updates = video_refs.view(prompt_h3.MODE_REF, None, _tiles(["a.png"]), "en", has_audio=lambda _path: False)
    state, *rest, source, note = updates
    tiles, cells = rest[:video_refs.SLOT_COUNT], rest[video_refs.SLOT_COUNT:]
    assert state[0] == "a.png"
    visible = [cell["visible"] is True for cell in cells]
    assert visible[:9] == [True, True] + [False] * 7
    assert visible[9:] == [True, False, False, True, False, False]
    assert tiles[0]["label"] == "<Picture 1>" and tiles[1]["label"] == "Image"
    assert source["visible"] is False and video_refs.frame(prompt_h3.MODE_REF)["visible"] is True
    assert "Up to 9 images" in note


def test_video_to_video_hides_the_video_tiles_and_shows_the_source():
    updates = video_refs.view(prompt_h3.MODE_V2V, "source.mp4", _tiles(), "en", has_audio=lambda _path: False)
    _state, *rest, source, _note = updates
    cells = rest[video_refs.SLOT_COUNT:]
    assert [cell["visible"] for cell in cells[9:12]] == [False] * 3
    assert source["visible"] is True and source["label"] == "<Video 1>"
    assert video_refs.frame(prompt_h3.MODE_V2V)["visible"] is True


def test_text_mode_offers_the_tiles_and_frame_modes_do_not():
    *_rest, source, note = video_refs.view(prompt_h3.MODE_T2V, None, _tiles(), "en")
    assert video_refs.frame(prompt_h3.MODE_T2V)["visible"] is True and source["visible"] is False
    assert "switches the mode" in note
    for mode in (prompt_h3.MODE_I2V, prompt_h3.MODE_FLF2V):
        assert video_refs.frame(mode)["visible"] is False


def test_a_file_in_text_mode_switches_to_references():
    mode, message = video_refs.promote(prompt_h3.MODE_T2V, "en", _tiles(["a.png"]))
    assert mode == prompt_h3.MODE_REF and "references" in message
    assert video_refs.promote(prompt_h3.MODE_T2V, "en", _tiles()) == (gr.update(), gr.update())
    assert video_refs.promote(prompt_h3.MODE_V2V, "en", _tiles(["a.png"])) == (gr.update(), gr.update())


def test_an_unreadable_video_gets_no_audio_label(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    tags, _source = video_refs.labels(prompt_h3.MODE_REF, None, _tiles(videos=[str(broken)]))
    assert tags["video"][0] == "<Video 1>"


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
    mode, first, last, *tiles, message = tab_video.send_to_video(
        [(picture, None)], None, prompt_h3.MODE_T2V, None, None, "en")
    assert mode == prompt_h3.MODE_I2V
    assert first["visible"] is True and first["value"].size == (40, 30)
    assert last == gr.update() and tiles == [gr.update()] * 9
    assert "first frame" in message


def test_in_image_mode_the_mode_stays(picture):
    mode, first, *_rest = tab_video.send_to_video([picture], picture, prompt_h3.MODE_I2V, None, None, "en")
    assert mode == prompt_h3.MODE_I2V and first["visible"] is True


def test_with_a_first_frame_the_picture_becomes_the_last(picture):
    mode, first, last, *tiles, message = tab_video.send_to_video(
        [picture], None, prompt_h3.MODE_FLF2V, Image.new("RGB", (8, 8)), None, "en")
    assert mode == first == gr.update() and tiles == [gr.update()] * 9
    assert last["visible"] is True and last["value"].size == (40, 30)
    assert "last frame" in message


def test_first_and_last_mode_without_a_first_frame_fills_the_first(picture):
    mode, first, last, *_rest = tab_video.send_to_video(
        [picture], None, prompt_h3.MODE_FLF2V, None, None, "en")
    assert mode == prompt_h3.MODE_FLF2V
    assert first["visible"] is True and last == gr.update()


@pytest.mark.parametrize("mode", [prompt_h3.MODE_REF, prompt_h3.MODE_V2V])
def test_in_reference_modes_the_picture_takes_the_first_free_image_tile(picture, mode):
    state = _tiles(["a.png", None, "c.png"], ["b.mp4"])
    answer_mode, first, last, *tiles, message = tab_video.send_to_video([picture], None, mode, None, state, "en")
    assert answer_mode == first == last == gr.update()
    assert tiles[1] == gr.update(value=picture)
    assert all(tile == gr.update() for index, tile in enumerate(tiles) if index != 1)
    assert "references" in message


def test_full_image_tiles_say_so(picture):
    state = _tiles([f"{index}.png" for index in range(9)])
    *_rest, message = tab_video.send_to_video([picture], None, prompt_h3.MODE_REF, None, state, "en")
    assert "taken" in message


def test_nothing_to_send(picture):
    answer = tab_video.send_to_video([], None, prompt_h3.MODE_T2V, None, None, "en")
    assert answer[:-1] == (gr.update(),) * 12
    assert answer[-1] == "Nothing to send"


def test_run_in_video_to_video_needs_a_source(built, monkeypatch):
    studio, handlers, _components = built
    monkeypatch.setattr(studio, "run_video", lambda *_a, **_k: pytest.fail("no source to edit"))
    answer = handlers["run"]("make it night", "", False, prompt_h3.MODE_V2V, "480p", "16:9", 5.0, -1, "light",
                             None, None, _tiles(), [], 1.0, False, "multiplier", 2.0, 1920, 1088, 2.0, "en", None,
                             progress=_no_progress)
    assert "source video" in answer[1]


def test_run_in_video_to_video_sends_the_source(built, monkeypatch, tmp_path):
    studio, handlers, _components = built
    asked = []
    source = tmp_path / "source.mp4"
    source.write_bytes(b"x")
    monkeypatch.setattr(studio, "run_video", lambda request, lang, on_event=None: asked.append(request) or (
        tmp_path / "v.mp4", {"seed": 1, "seconds": 4.0, "width": 864, "height": 480, "quality": "light"}, None))
    handlers["run"]("make it night", "", False, prompt_h3.MODE_V2V, "480p", "16:9", 5.0, -1, "light",
                    None, None, _tiles(["fox.png"], ["other.mp4"]), [], 1.0, False, "multiplier", 2.0, 1920, 1088,
                    2.0, "en", str(source), progress=_no_progress)
    (request,) = asked
    assert request.mode == prompt_h3.MODE_V2V and request.source_video == str(source)
    assert Reference("image", "fox.png") in request.references


def test_open_video_tab_switches_to_the_video_tab():
    assert tab_video.open_video_tab().selected == layout.TAB_VIDEO


def test_run_passes_the_upscale_settings(built, monkeypatch, tmp_path):
    from fooocus_qwen.video import geometry

    studio, handlers, _components = built
    asked = []
    monkeypatch.setattr(studio, "run_video", lambda request, lang, on_event=None: asked.append(request) or (
        tmp_path / "v.mp4", {"seed": 1, "seconds": 5.17, "width": 864, "height": 480, "quality": "light"}, None))
    handlers["run"]("a cat", "", False, prompt_h3.MODE_T2V, "480p", "16:9", 5.0, -1, "light",
                    None, None, None, [], 1.0, True, "megapixels", 2.0, 1920, 1088, 3.0, "en",
                    progress=_no_progress)
    (request,) = asked
    assert request.upscale == geometry.Upscale(mode="megapixels", factor=2.0, width=1920, height=1088, megapixels=3.0)


def test_upscale_fields_turn_into_settings_only_when_checked():
    from fooocus_qwen.ui import tab_video
    from fooocus_qwen.video import geometry

    assert tab_video.upscale_from(False, "multiplier", 2.0, 1920, 1088, 2.0) is None
    assert tab_video.upscale_from(True, "nonsense", None, "x", 1088, 2.0) == geometry.Upscale(
        mode="multiplier", factor=2.0, width=1920, height=1088, megapixels=2.0)
    box, factor, dims, megapixels = tab_video.upscale_visibility(True, "dimensions")
    assert box["visible"] and dims["visible"] and not factor["visible"] and not megapixels["visible"]
    assert tab_video.upscale_visibility(False, "multiplier")[0]["visible"] is False


def test_the_size_line_shows_the_upscaled_frame():
    from fooocus_qwen.ui import tab_video
    from fooocus_qwen.video import geometry

    line = tab_video.describe_size("t2v", "480p", "16:9", 5.0, None, None, "en", upscale=geometry.Upscale())
    assert "864×480" in line and "1728×960" in line and "×2.00" in line
    draft = tab_video.describe_size("t2v", "480p", "16:9", 5.0, None, None, "en", upscale=geometry.Upscale(),
                                    quality="draft")
    assert "not upscaled" in draft
    too_big = tab_video.describe_size("t2v", "480p", "16:9", 5.0, None, None, "en",
                                      upscale=geometry.Upscale(mode="dimensions", width=8192, height=4096))
    assert "more than 4" in too_big

"""Запрос на видео: задание драйвера из полей вкладки и ролик обратно в галерею.

Всё здесь — чистая логика без видеокарты: картинки синтетические (PIL),
видеореференсы — файлы-пустышки с расширением ``.mp4`` (``build_job`` их только
копирует, а наличие звука спрашивает у переданной функции).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from fooocus_qwen.video import geometry, prompt_h3
from fooocus_qwen.video import request as video_request
from fooocus_qwen.video.request import Reference, VideoRequest


def _image(path: Path, size=(64, 48), color=(200, 30, 30)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def _file(path: Path, data: bytes = b"fake media") -> Path:
    path.write_bytes(data)
    return path


def _build(tmp_path, request, video_has_audio=None):
    return video_request.build_job(
        request, tmp_path / "run", tmp_path / "runs", video_has_audio=video_has_audio,
    )


# --- кадры ---


def test_text_to_video_has_no_media(tmp_path):
    job, parameters = _build(tmp_path, VideoRequest(prompt="a cat on a windowsill", seed=7))
    assert job["media"] is None
    assert job["continuation_media"] is None
    assert (job["width"], job["height"]) == (864, 480)
    assert job["prompts"][0].endswith(prompt_h3.NEGATIVE_LINE)
    assert job["output_dir"] == str(tmp_path / "runs")
    assert parameters["mode"] == prompt_h3.MODE_T2V


def test_the_first_frame_is_cropped_exactly_into_the_canvas(tmp_path):
    first = Image.new("RGB", (1000, 1000), (10, 200, 10))
    job, _ = _build(tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_I2V, first=first))
    with Image.open(job["media"]["first"]) as saved:
        assert saved.size == (job["width"], job["height"]) == (864, 480)
    assert "last" not in job["media"]


def test_follow_image_takes_the_canvas_from_the_first_frame(tmp_path):
    first = Image.new("RGB", (900, 1600))
    job, parameters = _build(
        tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_I2V, aspect=geometry.FOLLOW_IMAGE, first=first),
    )
    assert (job["width"], job["height"]) == geometry.canvas("480p", "9:16")
    assert (parameters["width"], parameters["height"]) == (job["width"], job["height"])


def test_both_frames_are_cropped_into_the_canvas(tmp_path):
    request = VideoRequest(
        prompt="x", mode=prompt_h3.MODE_FLF2V, resolution="768p",
        first=Image.new("RGB", (300, 200)), last=Image.new("RGB", (200, 300)),
    )
    job, _ = _build(tmp_path, request)
    for key in ("first", "last"):
        with Image.open(job["media"][key]) as saved:
            assert saved.size == (1344, 768), key


def test_image_to_video_without_a_frame_is_refused(tmp_path):
    with pytest.raises(ValueError, match="first frame"):
        _build(tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_I2V))


def test_first_and_last_mode_needs_some_frame(tmp_path):
    with pytest.raises(ValueError):
        _build(tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_FLF2V))
    # Одного последнего кадра достаточно (L2VA).
    job, _ = _build(tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_FLF2V, last=Image.new("RGB", (8, 8))))
    assert "last" in job["media"] and "first" not in job["media"]


def test_an_unknown_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unknown video mode"):
        _build(tmp_path, VideoRequest(prompt="x", mode="x2v"))


def test_frames_of_other_modes_are_ignored(tmp_path):
    # Поле первого кадра скрыто, но значение в нём могло остаться.
    job, _ = _build(tmp_path, VideoRequest(prompt="x", first=Image.new("RGB", (8, 8))))
    assert job["media"] is None


# --- референсы ---


def test_references_go_to_the_engine_as_pictures_then_videos_then_sounds(tmp_path):
    references = (
        Reference("audio", str(_file(tmp_path / "voice.wav"))),
        Reference("video", str(_file(tmp_path / "walk.mp4"))),
        Reference("image", str(_image(tmp_path / "face.png"))),
    )
    job, parameters = _build(
        tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_REF, references=references),
        video_has_audio=lambda _path: False,
    )
    rows = job["media"]["references"]
    assert [row["kind"] for row in rows] == ["image", "video", "audio"]
    assert [Path(row["path"]).name for row in rows] == ["ref01.png", "ref02.mp4", "ref03.wav"]
    assert all(Path(row["path"]).is_file() for row in rows)
    assert [item["name"] for item in parameters["references"]] == ["face.png", "walk.mp4", "voice.wav"]
    assert parameters["labels"] == ["<Picture 1>", "<Video 1>", "<Audio 1>"]


def test_the_sound_of_a_video_reference_shifts_the_audio_labels(tmp_path):
    references = (
        Reference("audio", str(_file(tmp_path / "voice.wav"))),
        Reference("video", str(_file(tmp_path / "walk.mp4"))),
    )
    asked = []
    _job, parameters = _build(
        tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_REF, references=references),
        video_has_audio=lambda path: asked.append(path) or True,
    )
    assert asked == [str(tmp_path / "walk.mp4")]
    assert parameters["labels"] == ["<Audio 1>", "<Video 1>", "<Audio 2>"]


@pytest.mark.parametrize(
    ("kind", "count"),
    [("image", video_request.MAX_IMAGES + 1), ("video", video_request.MAX_VIDEOS + 1),
     ("audio", video_request.MAX_AUDIOS + 1)],
)
def test_too_many_references_are_refused(tmp_path, kind, count):
    references = tuple(Reference(kind, str(tmp_path / f"{kind}{index}")) for index in range(count))
    if kind == "audio":
        references += (Reference("image", str(tmp_path / "face.png")),)
    with pytest.raises(ValueError, match="Too many references"):
        _build(tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_REF, references=references))


def test_the_limits_themselves_are_allowed(tmp_path):
    images = tuple(Reference("image", str(_image(tmp_path / f"i{index}.png"))) for index in range(9))
    videos = tuple(Reference("video", str(_file(tmp_path / f"v{index}.mp4"))) for index in range(3))
    sounds = tuple(Reference("audio", str(_file(tmp_path / f"a{index}.wav"))) for index in range(3))
    job, _ = _build(
        tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_REF, references=images + videos + sounds),
        video_has_audio=lambda _path: False,
    )
    assert len(job["media"]["references"]) == 15


def test_a_sound_alone_is_refused(tmp_path):
    references = (Reference("audio", str(_file(tmp_path / "voice.wav"))),)
    with pytest.raises(ValueError, match="audio reference needs"):
        _build(tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_REF, references=references))


def test_reference_mode_without_references_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least one reference"):
        _build(tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_REF))


def test_a_large_picture_is_shrunk_to_1536_on_the_short_side(tmp_path):
    big = _image(tmp_path / "big.png", size=(1600, 2400))
    small = _image(tmp_path / "small.png", size=(300, 200))
    job, _ = _build(
        tmp_path,
        VideoRequest(prompt="x", mode=prompt_h3.MODE_REF,
                     references=(Reference("image", str(big)), Reference("image", str(small)))),
    )
    first, second = (row["path"] for row in job["media"]["references"])
    with Image.open(first) as saved:
        assert saved.size == (1536, 2304)
    with Image.open(second) as saved:
        assert saved.size == (300, 200), "мелкий снимок не растягивается"


def test_continuation_takes_at_most_three_pictures_and_nothing_else(tmp_path):
    images = tuple(Reference("image", str(_image(tmp_path / f"i{index}.png"))) for index in range(5))
    video = (Reference("video", str(_file(tmp_path / "v.mp4"))),)
    job, _ = _build(
        tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_REF, references=video + images,
                               loras=(("style.safetensors", 0.8),)),
        video_has_audio=lambda _path: False,
    )
    continuation = job["continuation_media"]
    assert len(continuation["references"]) == video_request.CONTINUATION_IMAGES
    assert {row["kind"] for row in continuation["references"]} == {"image"}
    assert continuation["loras"] == [{"path": "style.safetensors", "strength": 0.8}]


def test_loras_alone_make_media_for_both_parts(tmp_path):
    job, parameters = _build(tmp_path, VideoRequest(prompt="x", loras=((str(tmp_path / "a.safetensors"), 1),)))
    assert job["media"]["loras"] == [{"path": str(tmp_path / "a.safetensors"), "strength": 1.0}]
    assert job["continuation_media"]["references"] == []
    assert parameters["loras"] == [{"name": "a.safetensors", "strength": 1.0}]


# --- качество, сид, параметры ---


def test_draft_is_a_preview_and_the_flag_is_not_sent_as_quality(tmp_path):
    job, parameters = _build(tmp_path, VideoRequest(prompt="x", quality="draft"))
    assert job["preview"] is True
    assert "preview" not in job["quality"]
    assert parameters["quality"] == "draft"


@pytest.mark.parametrize("quality", ["light", "medium", "high", "max"])
def test_other_qualities_are_not_previews(tmp_path, quality):
    job, _ = _build(tmp_path, VideoRequest(prompt="x", quality=quality))
    assert job["preview"] is False
    assert job["quality"] == video_request.QUALITY[quality]


def test_an_unknown_quality_falls_back_to_the_default(tmp_path):
    job, parameters = _build(tmp_path, VideoRequest(prompt="x", quality="ultra"))
    assert job["quality"] == video_request.QUALITY[video_request.DEFAULT_QUALITY]
    assert parameters["quality"] == video_request.DEFAULT_QUALITY


def test_parameters_carry_frames_and_the_resolved_seed(tmp_path):
    job, parameters = _build(tmp_path, VideoRequest(prompt="x", seconds=5, seed=-1))
    assert parameters["frames"] == 124
    assert parameters["fps"] == 24
    assert parameters["seconds"] == round(124 / 24, 3)
    assert parameters["requested_seconds"] == 5.0
    assert 0 <= parameters["seed"] < 2**31
    assert job["seed"] == parameters["seed"], "в файл пишется тот сид, с которым снимали"
    assert job["seconds"] == pytest.approx(124 / 24)


def test_a_fixed_seed_is_kept(tmp_path):
    job, parameters = _build(tmp_path, VideoRequest(prompt="x", seed=42))
    assert job["seed"] == parameters["seed"] == 42


def test_the_original_text_is_stored_next_to_the_engine_prompt(tmp_path):
    job, parameters = _build(
        tmp_path, VideoRequest(prompt="boosted text", prompt_original="кот", extra={"storyboard_shot": 2}),
    )
    assert parameters["prompt"] == "кот"
    assert parameters["prompt_h3"] == job["prompts"][0]
    assert parameters["storyboard_shot"] == 2


def test_continue_and_refine_are_passed_through(tmp_path):
    job, parameters = _build(
        tmp_path, VideoRequest(prompt="x", continue_from="a.mp4", refine_from="draft/b.mp4"),
    )
    assert job["continue_from"] == "a.mp4" and parameters["continue_from"] == "a.mp4"
    assert job["refine_from"] == "draft/b.mp4"


def test_the_whole_job_is_plain_json(tmp_path):
    # Задание уходит драйверу строкой JSON — ничего непредставимого в нём быть не должно.
    references = (Reference("image", str(_image(tmp_path / "face.png"))),)
    job, parameters = _build(tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_REF, references=references))
    json.dumps(job)
    json.dumps(parameters)


def test_kind_of_reads_the_extension():
    assert video_request.kind_of("A.PNG") == "image"
    assert video_request.kind_of("b.webm") == "video"
    assert video_request.kind_of("c.flac") == "audio"
    assert video_request.kind_of("d.txt") is None


# --- результат ---


def test_save_result_puts_the_video_and_its_json_side_by_side(tmp_path):
    runs = tmp_path / "runs"
    produced = runs / "2026-10-19" / "job" / "final.mp4"
    produced.parent.mkdir(parents=True)
    produced.write_bytes(b"mp4 bytes")
    outputs = tmp_path / "outputs"

    saved = video_request.save_result(produced, outputs, {"seed": 5, "prompt": "кот"}, runs)

    assert saved.suffix == ".mp4" and saved.parent.parent == outputs
    assert saved.read_bytes() == b"mp4 bytes"
    assert produced.is_file(), "копия, а не перенос: кэш движка остаётся на месте"
    stored = json.loads(saved.with_suffix(".json").read_text(encoding="utf-8"))
    assert stored["engine_output"] == "2026-10-19/job/final.mp4"
    assert stored["prompt"] == "кот"
    assert video_request.read_parameters(saved) == stored


def test_an_output_outside_the_runs_keeps_its_full_path(tmp_path):
    produced = _file(tmp_path / "elsewhere.mp4")
    saved = video_request.save_result(produced, tmp_path / "outputs", {}, tmp_path / "runs")
    assert video_request.read_parameters(saved)["engine_output"] == str(produced)


def test_two_videos_in_the_same_second_do_not_share_a_name(tmp_path):
    produced = _file(tmp_path / "a.mp4")
    first = video_request.save_result(produced, tmp_path / "outputs", {"n": 1}, tmp_path)
    second = video_request.save_result(produced, tmp_path / "outputs", {"n": 2}, tmp_path)
    assert first != second
    assert video_request.read_parameters(first)["n"] == 1
    assert video_request.read_parameters(second)["n"] == 2


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", ""])
def test_broken_parameters_read_as_empty(tmp_path, content):
    video = _file(tmp_path / "v.mp4")
    video.with_suffix(".json").write_text(content, encoding="utf-8")
    assert video_request.read_parameters(video) == {}


def test_missing_parameters_read_as_empty(tmp_path):
    assert video_request.read_parameters(tmp_path / "none.mp4") == {}



# --- финальный апскейл ---


def test_the_upscaled_size_goes_to_the_engine_and_the_parameters(tmp_path):
    job, parameters = _build(tmp_path, VideoRequest(prompt="a cat", upscale=geometry.Upscale(factor=2.0)))
    assert job["final_upscale"] == [1728, 960]
    assert (job["width"], job["height"]) == (864, 480), "модель считает холст, а не выход"
    assert (parameters["output_width"], parameters["output_height"]) == (1728, 960)
    assert parameters["upscale"] == geometry.Upscale(factor=2.0).as_dict()
    json.dumps(job)


def test_a_draft_is_not_upscaled_but_keeps_the_settings_for_finishing(tmp_path):
    job, parameters = _build(tmp_path, VideoRequest(prompt="a cat", quality="draft", upscale=geometry.Upscale()))
    assert job["final_upscale"] is None
    assert parameters["upscale"] == geometry.Upscale().as_dict()
    assert (parameters["output_width"], parameters["output_height"]) == (864, 480)


def test_without_upscale_the_output_is_the_canvas(tmp_path):
    job, parameters = _build(tmp_path, VideoRequest(prompt="a cat"))
    assert job["final_upscale"] is None and parameters["upscale"] is None
    assert (parameters["output_width"], parameters["output_height"]) == (864, 480)


def test_an_impossible_upscale_is_refused_before_the_engine(tmp_path):
    request = VideoRequest(prompt="a cat", upscale=geometry.Upscale(mode=geometry.UPSCALE_DIMENSIONS,
                                                                      width=8192, height=4608))
    with pytest.raises(ValueError, match="more than 4"):
        _build(tmp_path, request)


# --- видео → видео ---


class _Info:
    def __init__(self, width=1280, height=720, seconds=6.0, has_audio=True):
        self.width, self.height, self.seconds, self.has_audio = width, height, seconds, has_audio


def _edit(tmp_path, info=None, **fields):
    source = _file(tmp_path / "source.mp4")
    request = VideoRequest(prompt="turn the dog into a fox", mode=prompt_h3.MODE_V2V, source_video=str(source),
                           **fields)
    info = info or _Info()
    return video_request.build_job(request, tmp_path / "run", tmp_path / "runs",
                                   video_has_audio=lambda path: info.has_audio, probe_video=lambda path: info)


def test_the_source_video_is_video_1_with_its_sound(tmp_path):
    face = Reference("image", str(_image(tmp_path / "fox.png")))
    job, parameters = _edit(tmp_path, references=(face,))
    assert [row["kind"] for row in job["media"]["references"]] == ["image", "video"]
    assert parameters["labels"] == ["<Picture 1>", "<Audio 1>", "<Video 1>"]
    prompt = job["prompts"][0]
    assert "[video editing + audio reuse] The target video is an edited version of <Video 1>" in prompt
    assert parameters["source_video"] == "source.mp4"


def test_other_videos_are_dropped_in_video_to_video(tmp_path):
    other = Reference("video", str(_file(tmp_path / "other.mp4")))
    job, parameters = _edit(tmp_path, references=(other,))
    assert [row["kind"] for row in job["media"]["references"]] == ["video"]
    assert Path(job["media"]["references"][0]["path"]).name.endswith(".mp4")


def test_the_edit_lasts_no_longer_than_its_source(tmp_path):
    job, parameters = _edit(tmp_path, _Info(seconds=4.0), seconds=10.0)
    assert parameters["seconds"] <= 4.0 + 1 / geometry.FPS
    job, _ = _edit(tmp_path, _Info(seconds=40.0), seconds=40.0)
    assert job["seconds"] <= geometry.SINGLE_MAX_SECONDS


def test_the_edit_keeps_the_source_proportion(tmp_path):
    job, _ = _edit(tmp_path, _Info(width=720, height=1280), aspect="16:9")
    assert job["width"] < job["height"], "вертикальный исходник — вертикальный холст, выбор 16:9 не действует"


def test_video_to_video_needs_a_source(tmp_path):
    with pytest.raises(ValueError, match="source video"):
        _build(tmp_path, VideoRequest(prompt="x", mode=prompt_h3.MODE_V2V))


def test_the_source_sound_comes_from_the_copied_source(tmp_path):
    job, parameters = _edit(tmp_path, audio_from="ignored, the copy is used")
    copied = [row["path"] for row in job["media"]["references"] if row["kind"] == "video"]
    assert parameters["audio_from"] == copied[0]
    _job, plain = _edit(tmp_path)
    assert plain["audio_from"] is None, "без галочки — звук модели"


def test_finishing_a_draft_keeps_the_source_sound(tmp_path):
    from fooocus_qwen.video import board

    video = tmp_path / "draft.mp4"
    video.write_bytes(b"x")
    video_request.sidecar(video).write_text(json.dumps({"engine_output": "drafts/a.mp4", "frames": 124,
                                                       "audio_from": "C:/runs/source.mp4"}), encoding="utf-8")
    assert board.follow_up_request(video, board.FINISH, "light").audio_from == "C:/runs/source.mp4"
    assert board.follow_up_request(video, board.EXTEND, "light").audio_from is None

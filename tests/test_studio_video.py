"""``Studio`` и видео: ролик в галерею, сбои строкой состояния, карта для FreeVideo.

Движок видео подменяется фальшивкой с методом ``generate``: она лишь кладёт
готовый «ролик» (байты) в каталог заданий и возвращает событие ``done``, как
настоящий драйвер. Каталоги результатов и заданий — во временном каталоге
теста. Модель Qwen не загружается: ``Studio`` ленивая (``test_studio_lazy``),
а «загруженный генератор» — пустышка, выгрузка которой подменена записью.
"""

from __future__ import annotations

import json

import pytest

from fooocus_qwen import config
from fooocus_qwen.ui.state import Studio
from fooocus_qwen.video import prompt_h3
from fooocus_qwen.video.engine import VideoCancelled, VideoEngineError
from fooocus_qwen.video.request import VideoRequest


class FakeVideo:
    """Движок видео: ``generate`` по сценарию, ``release``/``cancel`` — записью."""

    def __init__(self, runs, outcome=None, busy=False) -> None:
        self.runs = runs
        self.outcome = outcome
        self.busy = busy
        self.jobs: list[dict] = []
        self.released = 0
        self.cancelled = 0

    def generate(self, job, on_event=None):
        self.jobs.append(job)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        if on_event is not None:
            on_event({"event": "progress", "id": job["id"] if "id" in job else None, "value": {}})
        output = self.runs / "2026-10-19" / "job" / "final.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake mp4")
        return {"event": "done", "output": str(output), "parts": [str(output), str(output)]}

    def release(self):
        self.released += 1

    def cancel(self):
        self.cancelled += 1


@pytest.fixture
def studio(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "VIDEO_RUNS_DIR", tmp_path / "runs")
    studio = Studio(config.AppConfig())
    studio._video = FakeVideo(tmp_path / "runs")
    unloaded = []
    monkeypatch.setattr(studio, "unload", lambda: unloaded.append(True))
    studio.unloaded = unloaded
    return studio


def test_the_video_goes_to_the_gallery_with_its_parameters(studio, tmp_path):
    events = []
    path, parameters, failure = studio.run_video(VideoRequest(prompt="a cat", seed=9), "en", events.append)

    assert failure is None
    assert path.parent.parent == tmp_path / "outputs" and path.suffix == ".mp4"
    assert path.read_bytes() == b"fake mp4"
    stored = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    # Путь в кэше движка — только в файле: «Довести» и «Продлить» читают его оттуда.
    assert stored == {**parameters, "engine_output": stored["engine_output"]}
    assert stored["seed"] == 9 and stored["frames"] == 124 and stored["parts"] == 2
    assert stored["engine_output"] == "2026-10-19/job/final.mp4"
    assert isinstance(stored["seconds_taken"], float)
    assert [event["event"] for event in events] == ["progress"], "прогресс доходит до вкладки"
    (job,) = studio._video.jobs
    assert job["output_dir"] == str(tmp_path / "runs")


def test_job_inputs_stay_inside_the_runs_directory(studio, tmp_path):
    from PIL import Image

    request = VideoRequest(prompt="x", mode=prompt_h3.MODE_I2V, first=Image.new("RGB", (32, 32)))
    path, _parameters, failure = studio.run_video(request, "en")
    assert failure is None and path is not None
    first = studio._video.jobs[0]["media"]["first"]
    assert (tmp_path / "runs" / "inputs") in type(path)(first).parents


def test_a_stopped_video_is_a_message(studio):
    studio._video.outcome = VideoCancelled("Video generation was stopped")
    path, parameters, failure = studio.run_video(VideoRequest(prompt="x"), "en")
    assert (path, parameters) == (None, {})
    assert failure == "Video stopped."


def test_an_impossible_request_is_a_message_and_reaches_no_engine(studio):
    path, _parameters, failure = studio.run_video(VideoRequest(prompt="x", mode=prompt_h3.MODE_I2V), "en")
    assert path is None
    assert failure.startswith("Cannot do that:") and "first frame" in failure
    assert studio._video.jobs == []
    assert studio.unloaded == [], "ради заведомо невозможного ролика Qwen не выгружается"


def test_an_engine_failure_is_a_message(studio):
    studio._video.outcome = VideoEngineError("FreeVideo driver exited with code 3")
    _path, _parameters, failure = studio.run_video(VideoRequest(prompt="x"), "en")
    assert "The video engine failed" in failure and "code 3" in failure


def test_any_other_failure_is_a_message_too(studio):
    studio._video.outcome = OSError("disk full")
    _path, _parameters, failure = studio.run_video(VideoRequest(prompt="x"), "en")
    assert failure and "disk full" in failure


def test_nothing_is_written_to_the_gallery_on_failure(studio, tmp_path):
    studio._video.outcome = VideoCancelled("stop")
    studio.run_video(VideoRequest(prompt="x"), "en")
    outputs = tmp_path / "outputs"
    assert not outputs.exists() or not any(path.is_file() for path in outputs.rglob("*"))


# --- видеопамять ---


def test_claim_unloads_qwen_only_when_it_is_loaded(studio):
    studio.claim_gpu_for_video()
    assert studio.unloaded == []

    studio._generator = object()
    studio.claim_gpu_for_video()
    assert studio.unloaded == [True]


def test_a_video_frees_the_gpu_from_qwen_first(studio):
    studio._generator = object()
    studio.run_video(VideoRequest(prompt="x"), "en")
    assert studio.unloaded == [True]


def test_release_video_skips_a_busy_or_absent_engine(studio):
    studio.release_video()
    assert studio._video.released == 1

    studio._video.busy = True
    studio.release_video()
    assert studio._video.released == 1, "идущий ролик не прерывается отпусканием"

    studio._video = None
    studio.release_video()


def test_stop_video_cancels_only_an_existing_engine(studio):
    studio.stop_video()
    assert studio._video.cancelled == 1
    studio._video = None
    studio.stop_video()
    assert studio._video is None, "кнопка «Прервать» не запускает драйвер"

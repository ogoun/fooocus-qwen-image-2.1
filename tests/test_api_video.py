"""HTTP API видео: ``/video``, буст, раскадровка, статус, галерея с MP4 — на фальшивом движке.

Ни видеокарты, ни FreeVideo: ``Studio`` получает движок-заглушку с тем же
договором, что у ``video.engine.VideoEngine`` (``generate(job, on_event)``
→ событие ``done``, ``cancel``, ``busy``). Настоящий путь проходит всё
остальное: схемы, ``Studio.run_video``, ``request.build_job`` (подготовка
кадров и референсов), сохранение ролика с соседним JSON, очередь, отмена.
Склейка и кадры роликов (ffmpeg/PyAV) подменены там, где ролик — заглушка.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.api import server  # noqa: E402
from fooocus_qwen.ui.state import Studio  # noqa: E402
from fooocus_qwen.video import assemble, engine  # noqa: E402

TOKEN = "video-token"
FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42fake-video"


class FakeVideoEngine:
    """Движок-заглушка: пишет «ролик» в каталог заданий и шлёт событие прогресса."""

    def __init__(self, runs: Path) -> None:
        self.runs = runs
        self.jobs: list[dict] = []
        self.gate: threading.Event | None = None
        self.cancelled = False
        self.fail: str | None = None
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def generate(self, job, on_event=None):
        self._busy = True
        try:
            self.jobs.append(job)
            if on_event is not None:
                on_event({"event": "progress", "value": {
                    "label": "sampling", "done": 3, "total": 8, "timing_phase": "sampling",
                    "overall": {"fraction": 0.5, "remaining_seconds": 12.0}}})
            if self.gate is not None:
                self.gate.wait(10)
            if self.cancelled:
                self.cancelled = False
                raise engine.VideoCancelled("Video generation was stopped")
            if self.fail:
                raise engine.VideoEngineError(self.fail)
            output = self.runs / "out" / f"job{len(self.jobs)}.mp4"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(FAKE_MP4)
            return {"event": "done", "output": str(output), "parts": [str(output)]}
        finally:
            self._busy = False

    def cancel(self):
        if self._busy:
            self.cancelled = True
            if self.gate is not None:
                self.gate.set()

    def release(self, timeout=30.0):
        pass

    def close(self):
        pass


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path / "prompts")
    monkeypatch.setattr(config, "ENDPOINT_FILE", tmp_path / "llm_endpoint.txt")
    monkeypatch.setattr(config, "VIDEO_RUNS_DIR", tmp_path / "video_runs")
    monkeypatch.setattr(config, "VIDEO_LORA_DIR", tmp_path / "loras" / "video")
    config.OUTPUT_DIR.mkdir()
    config.PROMPT_DIR.mkdir()
    config.VIDEO_LORA_DIR.mkdir(parents=True)
    studio = Studio(config.AppConfig(preload=False, lora_dir=tmp_path / "loras"))
    # Генератора картинок нет: ``claim_gpu_for_video`` не выгружает модель
    # (выгрузка трогала бы torch.cuda).
    studio._generator = None
    fake = FakeVideoEngine(config.VIDEO_RUNS_DIR)
    studio._video = fake
    monkeypatch.setattr(studio, "video_status", lambda: engine.EngineStatus(True, True))
    client = TestClient(server.create_app(studio, TOKEN))
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    client.engine = fake
    client.studio = studio
    return client


def png_base64(size=(320, 180), colour=(10, 120, 200)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def make_video(api, **fields) -> dict:
    answer = api.post("/api/v1/video", json={"prompt": "A fox runs through snow.", "seed": 7, **fields})
    assert answer.status_code == 200, answer.text
    return answer.json()


def wait_for(api, job_id: str) -> dict:
    """Задача, дождавшаяся конца (готова, упала или снята)."""
    for _ in range(300):
        job = api.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    pytest.fail(f"the job is still {job['status']}")


# --- ролик -------------------------------------------------------------------------


def test_text_to_video_saves_mp4_with_parameters_and_serves_it(api):
    body = make_video(api, seconds=5, resolution="480p", aspect="16:9")
    video = body["video"]
    saved = config.OUTPUT_DIR / video["path"]
    assert saved.suffix == ".mp4" and saved.read_bytes() == FAKE_MP4
    assert saved.with_suffix(".json").is_file(), "параметры ролика — соседним JSON"
    assert video["seed"] == 7 and (video["width"], video["height"]) == (864, 480)
    assert video["parameters"]["frames"] == 124 and video["parameters"]["mode"] == "t2v"
    assert "base64" not in video
    job = api.engine.jobs[-1]
    assert job["seed"] == 7 and job["prompts"][0] == body["prompt"]

    served = api.get(video["url"])
    assert served.status_code == 200 and served.content == FAKE_MP4
    assert served.headers["content-type"] == "video/mp4"
    part = api.get(video["url"], headers={"Range": "bytes=0-3"})
    assert part.status_code == 206 and part.content == FAKE_MP4[:4], "плеер перематывает по Range"

    listed = api.get("/api/v1/gallery").json()
    assert listed[0]["path"] == video["path"] and listed[0]["kind"] == "video"
    assert listed[0]["parameters"]["seed"] == 7, "параметры ролика — из JSON, не из PNG"


def test_video_job_reports_engine_progress_and_returns_base64(api):
    answer = api.post("/api/v1/video?wait=false", json={"prompt": "A fox.", "return_base64": True})
    assert answer.status_code == 202 and answer.json()["kind"] == "video"
    job = wait_for(api, answer.json()["id"])
    assert job["status"] == "done", job
    progress = job["progress"]
    assert progress["fraction"] == 0.5 and progress["remaining_seconds"] == 12.0
    assert (progress["step"], progress["steps"]) == (3, 8) and progress["stage"] == "sampling"
    assert base64.b64decode(job["result"]["video"]["base64"]) == FAKE_MP4


def test_frames_follow_the_mode(api):
    refused = api.post("/api/v1/video", json={"prompt": "x", "mode": "i2v"})
    assert refused.status_code == 400 and "first frame" in refused.json()["detail"]
    assert api.post("/api/v1/video", json={"prompt": "x", "mode": "flf2v"}).status_code == 400
    body = make_video(api, mode="flf2v", first=png_base64(), last=png_base64(colour=(200, 0, 0)), aspect="image")
    media = api.engine.jobs[-1]["media"]
    assert Path(media["first"]).is_file() and Path(media["last"]).is_file()
    with Image.open(media["first"]) as first:
        assert first.size == (api.engine.jobs[-1]["width"], api.engine.jobs[-1]["height"])
    ignored = make_video(api, mode="t2v", first=png_base64())
    assert api.engine.jobs[-1]["media"] is None
    assert any("first frame is ignored" in line for line in ignored["messages"])
    assert body["video"]["parameters"]["mode"] == "flf2v"


def test_a_gallery_video_as_the_first_frame_gives_its_last_frame(api, monkeypatch):
    previous = make_video(api)["video"]["path"]
    seen = []

    def last_frame(path, dest_png=None):
        seen.append(Path(path).name)
        return Image.new("RGB", (640, 360), (1, 2, 3))

    monkeypatch.setattr(assemble, "last_frame", last_frame)
    make_video(api, mode="i2v", first={"gallery": previous})
    assert seen == [Path(previous).name]
    with Image.open(api.engine.jobs[-1]["media"]["first"]) as first:
        assert first.getpixel((10, 10)) == (1, 2, 3)


def test_references_from_base64_and_gallery_get_labels_and_limits(api):
    gallery_png = config.OUTPUT_DIR / "2026-10-19" / "10-00-00.png"
    gallery_png.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (0, 200, 0)).save(gallery_png)
    voice = base64.b64encode(b"RIFF....WAVEfmt fake").decode()
    body = make_video(api, mode="ref", prompt="<Picture 1> talks to <Picture 2> in the voice <Audio 1>.",
                      references=[{"base64": png_base64(), "kind": "image"},
                                  {"gallery": "2026-10-19/10-00-00.png"},
                                  {"base64": voice, "name": "voice.mp3"}])
    rows = api.engine.jobs[-1]["media"]["references"]
    assert [row["kind"] for row in rows] == ["image", "image", "audio"]
    assert rows[2]["path"].endswith(".mp3")
    assert body["video"]["parameters"]["labels"] == ["<Picture 1>", "<Picture 2>", "<Audio 1>"]
    uploads = config.VIDEO_RUNS_DIR / "api_uploads"
    assert not any(path.is_file() for path in uploads.rglob("*")), "загрузки удаляются после работы"

    unknown = api.post("/api/v1/video", json={"prompt": "x", "mode": "ref", "references": [{"base64": voice}]})
    assert unknown.status_code == 400 and "references[0]" in unknown.json()["detail"]
    lonely = api.post("/api/v1/video", json={"prompt": "x", "mode": "ref",
                                             "references": [{"base64": voice, "kind": "audio"}]})
    assert lonely.status_code == 400 and "audio" in lonely.json()["detail"].lower()
    too_many = [{"base64": png_base64((32, 32)), "kind": "image"}] * 10
    assert api.post("/api/v1/video", json={"prompt": "x", "mode": "ref", "references": too_many}).status_code == 400
    assert api.post("/api/v1/video", json={"prompt": "x", "mode": "ref"}).status_code == 400
    both = api.post("/api/v1/video", json={"prompt": "x", "mode": "ref",
                                           "references": [{"base64": voice, "gallery": "a.png"}]})
    assert both.status_code == 422
    outside = api.post("/api/v1/video", json={"prompt": "x", "mode": "ref",
                                              "references": [{"gallery": "../secret.png"}]})
    assert outside.status_code in (400, 404)


def test_video_loras_come_from_the_video_lora_folder(api):
    (config.VIDEO_LORA_DIR / "style.safetensors").write_bytes(b"x")
    assert api.get("/api/v1/video/loras").json() == ["style.safetensors"]
    make_video(api, loras=[{"name": "style", "strength": 0.7}])
    assert api.engine.jobs[-1]["media"]["loras"] == [
        {"path": str(config.VIDEO_LORA_DIR / "style.safetensors"), "strength": 0.7}]
    assert api.post("/api/v1/video", json={"prompt": "x", "loras": [{"name": "nope"}]}).status_code == 404
    assert api.post("/api/v1/video", json={"prompt": "x", "loras": [{"name": "../style"}]}).status_code == 400
    assert api.post("/api/v1/video", json={"prompt": "x", "loras": [{"name": "style", "strength": 5}]}) \
        .status_code == 422


def test_bad_requests_are_refused_before_queueing(api):
    assert api.post("/api/v1/video", json={"prompt": "  "}).status_code == 400
    assert api.post("/api/v1/video", json={"prompt": "x", "quality": "ultra"}).status_code == 422
    assert api.post("/api/v1/video", json={"prompt": "x", "seconds": 90}).status_code == 422
    assert api.post("/api/v1/video", json={"prompt": "x", "frames": 10}).status_code == 422
    assert api.engine.jobs == []


def test_engine_not_installed_is_503_and_status_says_how_to_install(api, monkeypatch):
    missing = engine.EngineStatus(False, False, "FreeVideo is not installed")
    monkeypatch.setattr(api.studio, "video_status", lambda: missing)
    refused = api.post("/api/v1/video", json={"prompt": "x"})
    assert refused.status_code == 503 and "--setup-video" in refused.json()["detail"]
    status = api.get("/api/v1/video/status").json()
    assert status["ready"] is False and status["installed"] is False and "--setup-video" in status["install"]
    assert api.get("/api/v1/info").json()["video"]["qualities"][0] == "draft"


def test_an_engine_failure_is_a_500_with_the_reason(api):
    api.engine.fail = "CUDA out of memory"
    answer = api.post("/api/v1/video", json={"prompt": "x"})
    assert answer.status_code == 500 and "out of memory" in answer.json()["detail"]


def test_cancelling_a_running_video_stops_the_engine(api):
    api.engine.gate = threading.Event()
    started = api.post("/api/v1/video?wait=false", json={"prompt": "x"}).json()
    for _ in range(200):
        if api.engine.busy:
            break
        time.sleep(0.02)
    assert api.delete(f"/api/v1/jobs/{started['id']}").status_code == 200
    job = wait_for(api, started["id"])
    assert job["status"] == "cancelled" and "stopped" in job["error"].lower()


# --- продление и доводка -----------------------------------------------------------


def test_refine_from_finishes_the_draft_and_continue_from_extends(api):
    draft = make_video(api, quality="draft", seconds=5)["video"]
    assert api.engine.jobs[-1]["preview"] is True
    engine_output = draft["parameters"]["engine_output"]

    make_video(api, prompt="", refine_from=draft["path"])
    job = api.engine.jobs[-1]
    assert job["refine_from"] == engine_output and job["preview"] is False and job["seed"] == 7

    extended = api.post("/api/v1/video", json={"continue_from": draft["path"], "prompt": "The fox jumps.",
                                               "quality": "draft"})
    assert extended.status_code == 200, extended.text
    job = api.engine.jobs[-1]
    assert job["continue_from"] == str((config.VIDEO_RUNS_DIR / engine_output).resolve())
    assert "The fox jumps." in job["prompts"][0] and job["preview"] is False

    assert api.post("/api/v1/video", json={"continue_from": "x.mp4", "refine_from": "y.mp4"}).status_code == 422
    picture = config.OUTPUT_DIR / "2026-10-19" / "11-00-00.png"
    picture.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(picture)
    not_video = api.post("/api/v1/video", json={"continue_from": "2026-10-19/11-00-00.png"})
    assert not_video.status_code == 400
    (config.OUTPUT_DIR / draft["path"]).with_suffix(".json").write_text("{}", encoding="utf-8")
    gone = api.post("/api/v1/video", json={"refine_from": draft["path"]})
    assert gone.status_code == 400 and "engine files" in gone.json()["detail"]


# --- языковая модель -----------------------------------------------------------------


def test_video_boost_and_boost_inside_the_job(api, monkeypatch):
    calls = []

    def boost_video(text, mode, duration, references, first, last, aspect, lang):
        calls.append((text, mode, duration.frames, first is not None, aspect))
        return "Rewritten H3 prompt.", "Prompt rewritten."

    monkeypatch.setattr(api.studio, "boost_video", boost_video)
    answer = api.post("/api/v1/video/boost", json={"prompt": "лиса бежит", "mode": "i2v", "first": png_base64(),
                                                   "seconds": 5})
    assert answer.status_code == 200 and answer.json()["prompt"] == "Rewritten H3 prompt."
    assert calls[-1] == ("лиса бежит", "i2v", 124, True, "16:9")

    body = make_video(api, prompt="лиса бежит", boost=True)
    assert "Rewritten H3 prompt." in api.engine.jobs[-1]["prompts"][0]
    assert body["video"]["parameters"]["prompt"] == "лиса бежит", "в параметрах — исходный текст"
    assert "Prompt rewritten." in body["messages"]

    monkeypatch.setattr(api.studio, "boost_video", lambda *args: ("", "AI boost failed: offline"))
    assert api.post("/api/v1/video/boost", json={"prompt": "x"}).status_code == 502


def test_storyboard_plan_returns_shot_objects(api, monkeypatch):
    rows = [[1, 6.375, "cut", "Opening", "A fox wakes up."], [2, 5.667, "dissolve", "Run", "The fox runs."]]
    seen = []

    def plan(brief, total, structure, aspect, lang):
        seen.append((brief, total, structure, aspect))
        return rows, "2 shots"

    monkeypatch.setattr(api.studio, "plan_storyboard", plan)
    answer = api.post("/api/v1/video/storyboard", json={"brief": "A fox's morning", "seconds": 12,
                                                        "structure": "story", "aspect": "9:16"})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    assert body["shots"][1] == {"seconds": 5.667, "transition": "dissolve", "title": "Run",
                                "prompt": "The fox runs."}
    assert seen == [("A fox's morning", 12.0, "story", "9:16")]
    assert api.post("/api/v1/video/storyboard", json={"brief": "x", "structure": "nope"}).status_code == 422
    monkeypatch.setattr(api.studio, "plan_storyboard", lambda *args: (None, "Planning failed: offline"))
    assert api.post("/api/v1/video/storyboard", json={"brief": "x"}).status_code == 502


def test_storyboard_run_chains_frames_and_joins_the_shots(api, monkeypatch):
    frames = []

    def last_frame(path, dest_png=None):
        frames.append(Path(path).name)
        return Image.new("RGB", (320, 180), (9, 9, 9))

    def concat(paths, out, transitions=()):
        Path(out).write_bytes(FAKE_MP4 + b"joined")
        concat.seen = (list(paths), list(transitions))
        return Path(out)

    def contact_sheet(paths, dest, **_kwargs):
        Image.new("RGB", (16, 16)).save(dest)
        return Path(dest)

    monkeypatch.setattr(assemble, "last_frame", last_frame)
    monkeypatch.setattr(assemble, "concat", concat)
    monkeypatch.setattr(assemble, "contact_sheet", contact_sheet)
    monkeypatch.setattr(assemble, "probe", lambda path: assemble.MediaInfo(864, 480, 24.0, 290, 12.08, True))
    answer = api.post("/api/v1/video/storyboard/run", json={
        "shots": [{"prompt": "Shot one.", "seconds": 6}, {"prompt": "Shot two.", "seconds": 6,
                                                         "transition": "cross dissolve"}],
        "seed": 100, "first": png_base64(),
    })
    assert answer.status_code == 200, answer.text
    body = answer.json()
    first_job, second_job = api.engine.jobs[-2:]
    assert first_job["seed"] == 100 and second_job["seed"] == 101
    assert first_job["media"]["first"] and second_job["media"]["first"], "цепочка кадров"
    assert len(frames) == 2 and concat.seen[1] == ["dissolve"]
    assert len(body["shots"]) == 2 and body["seconds"] == 12.08
    assert (config.OUTPUT_DIR / body["video"]["path"]).read_bytes().endswith(b"joined")
    assert api.get(body["contact_sheet"]["url"]).headers["content-type"] == "image/png"
    assert body["video"]["parameters"]["prompt"] == "Storyboard of 2 shots"
    assert api.post("/api/v1/video/storyboard/run", json={"shots": []}).status_code == 422


def test_interrupt_also_stops_the_video(api, monkeypatch):
    stopped = []
    monkeypatch.setattr(api.studio, "stop_video", lambda: stopped.append(True))
    assert api.post("/api/v1/interrupt").status_code == 200
    assert stopped == [True]



def test_upscale_settings_reach_the_request_and_bad_sizes_are_refused(api):
    body = make_video(api, upscale={"mode": "multiplier", "factor": 2.0})
    assert api.engine.jobs[-1]["final_upscale"] == [1728, 960]
    assert body["video"]["parameters"]["upscale"]["factor"] == 2.0
    refused = api.post("/api/v1/video", json={"prompt": "A fox.", "upscale": {"mode": "dimensions",
                                                                              "width": 8192, "height": 4608}})
    assert refused.status_code == 400 and "more than 4" in refused.text
    too_much = api.post("/api/v1/video", json={"prompt": "A fox.", "upscale": {"factor": 5}})
    assert too_much.status_code == 422


def test_video_to_video_edits_the_source_as_video_1(api, monkeypatch):
    previous = make_video(api)["video"]["path"]
    monkeypatch.setattr(assemble, "probe", lambda path: assemble.MediaInfo(720, 1280, 24.0, 96, 4.0, True))
    body = make_video(api, mode="v2v", prompt="Turn the fox into a wolf.", seconds=10.0,
                      source={"gallery": previous},
                      references=[{"base64": png_base64(), "kind": "image"}])
    job = api.engine.jobs[-1]
    assert [row["kind"] for row in job["media"]["references"]] == ["image", "video"]
    assert job["width"] < job["height"], "холст — в пропорции исходника"
    assert job["seconds"] <= 4.0
    assert body["video"]["parameters"]["labels"] == ["<Picture 1>", "<Audio 1>", "<Video 1>"]
    assert "edited version of <Video 1>" in body["prompt"]

    missing = api.post("/api/v1/video", json={"prompt": "x", "mode": "v2v"})
    assert missing.status_code == 400 and "source" in missing.json()["detail"]
    picture = api.post("/api/v1/video", json={"prompt": "x", "mode": "v2v",
                                              "source": {"base64": png_base64(), "kind": "image"}})
    assert picture.status_code == 400 and "video is needed" in picture.json()["detail"]

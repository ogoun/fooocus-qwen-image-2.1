"""HTTP API (``--api``): маршруты, токен, очередь задач — на подставном генераторе.

Модель не грузится: ``Studio`` получает генератор-заглушку, которая
запоминает запросы и отдаёт картинки нужного размера. Проверяется то, что
делает сам API: разбор картинок и масок, сборка ``GenerationRequest`` (та
же, что у вкладок), сохранение PNG, ответы и коды ошибок, Bearer-токен.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from contextlib import contextmanager

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.api import server  # noqa: E402
from fooocus_qwen.engine import lora  # noqa: E402
from fooocus_qwen.engine.generator import MASK_MASK, MASK_NONE, GeneratedImage, resolve_seed  # noqa: E402
from fooocus_qwen.imaging import aspect, metadata  # noqa: E402
from fooocus_qwen.ui.state import Studio  # noqa: E402

TOKEN = "s3cret-token"


class FakeGenerator:
    def __init__(self) -> None:
        self.requests = []
        self.interrupted = False
        self.gate: threading.Event | None = None
        self.fail: Exception | None = None
        self.pipe = None

    def generate(self, request, progress=None):
        self.requests.append(request)
        if self.gate is not None:
            self.gate.wait(10)
        if self.fail is not None:
            raise self.fail
        if self.interrupted:
            self.interrupted = False
            return []
        if request.size:
            size = request.size
        elif request.source is not None:
            size = request.source.size
        else:
            size = (96, 64)
        produced = []
        for index in range(request.image_number):
            if progress:
                progress(index, 1, 2)
            seed = resolve_seed(request.seed) + index
            parameters = {"prompt": request.prompt, "seed": seed, "width": size[0], "height": size[1],
                          "preset": request.preset.name}
            produced.append(GeneratedImage(image=Image.new("RGB", size, (200, 30, 30)), seed=seed,
                                           parameters=parameters))
        return produced

    def interrupt(self):
        self.interrupted = True
        if self.gate is not None:
            self.gate.set()

    @contextmanager
    def exclusive(self):
        yield

    def recover(self):
        pass


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path / "prompts")
    monkeypatch.setattr(config, "ENDPOINT_FILE", tmp_path / "llm_endpoint.txt")
    config.OUTPUT_DIR.mkdir()
    config.PROMPT_DIR.mkdir()
    studio = Studio(config.AppConfig(preload=False, lora_dir=tmp_path / "loras"))
    fake = FakeGenerator()
    studio._generator = fake
    outpaint_file = tmp_path / "outpaint.safetensors"
    monkeypatch.setattr(studio, "outpaint_lora", lambda lang, progress=None: (
        lora.ResolvedLora("outpaint", 1.0, outpaint_file, "abc"), None))
    client = TestClient(server.create_app(studio, TOKEN))
    client.headers["Authorization"] = f"Bearer {TOKEN}"
    client.fake = fake
    client.studio = studio
    return client


def png_base64(size=(80, 60), colour=(10, 120, 200), mode="RGB") -> str:
    buffer = io.BytesIO()
    Image.new(mode, size, colour).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode(text: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(text)))


# --- доступ ------------------------------------------------------------------------


def test_health_is_open_and_everything_else_needs_the_token(api):
    bare = TestClient(api.app)
    assert bare.get("/api/v1/health").json() == {"status": "ok"}
    assert bare.get("/api/v1/info").status_code == 401
    assert bare.get("/api/v1/info", headers={"Authorization": "Bearer wrong"}).status_code == 401
    refused = bare.get("/api/v1/info")
    assert refused.headers["www-authenticate"] == "Bearer"
    assert api.get("/api/v1/info").status_code == 200


def test_without_a_token_the_api_is_open(api):
    open_client = TestClient(server.create_app(api.studio, None))
    assert open_client.get("/api/v1/presets").status_code == 200


def test_info_lists_the_catalogues(api):
    info = api.get("/api/v1/info").json()
    assert "QwenTurbo" in info["presets"] and info["default_preset"] == "MiddleQuality"
    assert "reference" in info["aspect_ratios"] and aspect.FOLLOW_REFERENCE not in info["aspect_ratios"]
    presets = {item["name"]: item for item in api.get("/api/v1/presets").json()}
    assert presets["QwenTurbo2K"]["resolution"] == 2048 and presets["QwenTurbo2K"]["steps"] == 8
    assert len(api.get("/api/v1/styles").json()) > 10


# --- генерация ---------------------------------------------------------------------


def test_generate_waits_saves_png_and_returns_base64(api):
    answer = api.post("/api/v1/generate", json={"prompt": "a red fox", "preset": "LowQuality", "seed": 5,
                                                 "image_number": 2, "aspect": "16:9"})
    assert answer.status_code == 200, answer.text
    body = answer.json()
    request = api.fake.requests[-1]
    assert request.prompt == "a red fox" and request.aspect == "16:9" and request.image_number == 2
    assert request.preset.name == "LowQuality"
    assert [item["seed"] for item in body["images"]] == [5, 6]
    first = body["images"][0]
    assert decode(first["base64"]).size == (96, 64)
    saved = config.OUTPUT_DIR / first["path"]
    assert saved.is_file() and metadata.read_png(saved)["seed"] == 5
    assert api.get(first["url"]).content == saved.read_bytes()


def test_generate_without_wait_returns_a_job_to_poll(api):
    answer = api.post("/api/v1/generate?wait=false", json={"prompt": "cat", "preset": "LowQuality",
                                                            "return_base64": False})
    assert answer.status_code == 202
    location = answer.headers["location"]
    for _ in range(100):
        job = api.get(location).json()
        if job["status"] == "done":
            break
        time.sleep(0.02)
    assert job["status"] == "done" and job["kind"] == "generate"
    assert "base64" not in job["result"]["images"][0]
    assert job["progress"]["steps"] == 2


def test_bad_requests_are_refused_before_queueing(api):
    assert api.post("/api/v1/generate", json={"prompt": "x", "preset": "Fastest"}).status_code == 422
    assert api.post("/api/v1/generate", json={"prompt": "x", "colour": "red"}).status_code == 422
    assert api.post("/api/v1/generate", json={"prompt": "   "}).status_code == 400
    styles = api.post("/api/v1/generate", json={"prompt": "x", "styles": ["No Such Style"]})
    assert styles.status_code == 400 and "No Such Style" in styles.json()["detail"]
    broken = api.post("/api/v1/generate", json={"prompt": "x", "references": ["not base64 at all"]})
    assert broken.status_code == 400 and "references[0]" in broken.json()["detail"]
    follow = api.post("/api/v1/generate", json={"prompt": "x", "aspect": "reference"})
    assert follow.status_code == 400
    assert api.fake.requests == []


def test_references_take_base64_poses_and_gallery_images(api):
    pose = api.get("/api/v1/poses").json()[0]
    assert api.get(pose["skeleton_url"]).status_code == 200
    first = api.post("/api/v1/generate", json={"prompt": "cat", "preset": "LowQuality"}).json()
    answer = api.post("/api/v1/generate", json={
        "prompt": "<image1> in the pose of <image2>, background from <image3>", "preset": "LowQuality",
        "aspect": "reference",
        "references": [png_base64(), {"pose": pose["name"]}, {"gallery": first["images"][0]["path"]}],
    })
    assert answer.status_code == 200, answer.text
    request = api.fake.requests[-1]
    assert len(request.references) == 3 and request.aspect == aspect.FOLLOW_REFERENCE
    assert [item["tag"] for item in answer.json()["conditions"]] == ["<image1>", "<image2>", "<image3>"]
    assert api.post("/api/v1/generate", json={"prompt": "x", "references": [{"pose": "nope"}]}).status_code == 404


def test_a_generation_failure_is_a_500_with_the_reason(api):
    api.fake.fail = RuntimeError("CUDA error: device lost")
    answer = api.post("/api/v1/generate", json={"prompt": "cat", "preset": "LowQuality"})
    assert answer.status_code == 500
    assert "device lost" in answer.json()["detail"] and answer.json()["status"] == "failed"


class _CudaFault(RuntimeError):
    """Как ``torch.AcceleratorError``: код ошибки CUDA в ``error_code``."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.error_code = code


def test_a_cuda_fault_makes_health_unhealthy_and_later_jobs_fail_without_the_gpu(api):
    api.fake.fail = _CudaFault("CUDA error: shared object initialization failed", 302)
    first = api.post("/api/v1/generate", json={"prompt": "cat", "preset": "LowQuality"})
    assert first.status_code == 500 and "Restart the studio" in first.json()["detail"]
    health = TestClient(api.app).get("/api/v1/health")
    assert health.status_code == 503 and health.json()["status"] == "gpu_fault"
    assert "shared object" in health.json()["detail"]
    calls = len(api.fake.requests)
    second = api.post("/api/v1/generate", json={"prompt": "dog", "preset": "LowQuality"})
    assert second.status_code == 500 and "Restart the studio" in second.json()["detail"]
    assert len(api.fake.requests) == calls  # к видеокарте второй раз не ходили


def test_a_queued_job_can_be_cancelled_and_a_running_one_interrupted(api):
    api.fake.gate = threading.Event()
    running = api.post("/api/v1/generate?wait=false", json={"prompt": "one", "preset": "LowQuality"}).json()
    queued = api.post("/api/v1/generate?wait=false", json={"prompt": "two", "preset": "LowQuality"}).json()
    assert queued["queue_position"] == 0
    assert api.delete(f"/api/v1/jobs/{queued['id']}").json()["status"] == "cancelled"
    for _ in range(100):
        if api.fake.requests:
            break
        time.sleep(0.02)
    api.delete(f"/api/v1/jobs/{running['id']}")
    for _ in range(100):
        job = api.get(f"/api/v1/jobs/{running['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.02)
    assert job["status"] == "cancelled"
    assert [request.prompt for request in api.fake.requests] == ["one"], "снятая задача не запускалась"
    assert api.delete(f"/api/v1/jobs/{queued['id']}").status_code == 409


# --- правка и расширение кадра -----------------------------------------------------


def test_edit_with_a_transparent_stroke_layer_as_the_mask(api):
    layer = Image.new("RGBA", (80, 60), (0, 0, 0, 0))
    layer.paste((255, 255, 255, 255), (10, 10, 30, 30))
    buffer = io.BytesIO()
    layer.save(buffer, format="PNG")
    answer = api.post("/api/v1/edit", json={
        "image": png_base64(), "mask": base64.b64encode(buffer.getvalue()).decode(), "prompt": "add a cat",
        "preset": "LowQuality", "seed": 3,
    })
    assert answer.status_code == 200, answer.text
    request = api.fake.requests[-1]
    assert request.mask_mode == MASK_MASK and request.source.size == (80, 60)
    assert request.mask.getpixel((15, 15)) == 255 and request.mask.getpixel((50, 50)) == 0
    assert [item["role"] for item in answer.json()["conditions"]] == ["source", "mask"]


def test_edit_crop_cuts_image_and_mask_together(api):
    mask = png_base64((80, 60), (255, 255, 255))
    answer = api.post("/api/v1/edit", json={"image": png_base64(), "mask": mask, "prompt": "x",
                                             "preset": "LowQuality", "crop": [10, 5, 50, 45]})
    assert answer.status_code == 200, answer.text
    request = api.fake.requests[-1]
    assert request.source.size == (40, 40) and request.mask.size == (40, 40)
    outside = api.post("/api/v1/edit", json={"image": png_base64(), "prompt": "x", "crop": [0, 0, 500, 10]})
    assert outside.status_code == 400


def test_edit_with_an_empty_mask_edits_the_whole_frame_and_says_so(api):
    answer = api.post("/api/v1/edit", json={"image": png_base64(), "mask": png_base64(colour=(0, 0, 0)),
                                             "prompt": "x", "preset": "LowQuality"})
    assert api.fake.requests[-1].mask_mode == MASK_NONE
    assert any("empty" in line for line in answer.json()["messages"])
    refused = api.post("/api/v1/edit", json={"image": png_base64(), "prompt": "x", "mask_mode": "region"})
    assert refused.status_code == 400


def test_outpaint_to_a_ratio_returns_the_wider_frame(api):
    answer = api.post("/api/v1/outpaint", json={"image": png_base64((300, 300)), "ratio": "16:9",
                                                 "preset": "LowQuality"})
    assert answer.status_code == 200, answer.text
    image = answer.json()["images"][0]
    assert decode(image["base64"]).size == (534, 300)
    assert image["parameters"]["outpaint"]["ratio"] == "16:9"
    request = api.fake.requests[-1]
    assert request.size == request.source.size and request.loras[0].name == "outpaint"
    assert api.post("/api/v1/outpaint", json={"image": png_base64((300, 300))}).status_code == 422
    same = api.post("/api/v1/outpaint", json={"image": png_base64((300, 300)), "ratio": "1:1"})
    assert same.status_code == 400


# --- прочее ------------------------------------------------------------------------


def test_gallery_lists_outputs_and_files_stay_inside_the_outputs_folder(api):
    api.post("/api/v1/generate", json={"prompt": "cat", "preset": "LowQuality", "seed": 9})
    listed = api.get("/api/v1/gallery").json()
    assert listed[0]["parameters"]["seed"] == 9
    (config.OUTPUT_DIR.parent / "secret.png").write_bytes(b"x")
    assert api.get("/api/v1/files/../secret.png").status_code in (400, 404)
    assert api.get("/api/v1/files/..%2Fsecret.png").status_code in (400, 404)
    assert api.post("/api/v1/generate", json={"prompt": "x", "references": [{"gallery": "../secret.png"}]}) \
        .status_code == 400


def test_saved_prompts_round_trip(api):
    saved = api.put("/api/v1/prompts/My fox", json={"prompt": "a fox", "preset": "QwenTurbo", "seed": 4})
    assert saved.status_code == 200 and saved.json()["preset"] == "QwenTurbo"
    assert "My fox" in api.get("/api/v1/prompts").json()
    assert api.get("/api/v1/prompts/My fox").json()["prompt"] == "a fox"
    assert api.delete("/api/v1/prompts/My fox").status_code == 200
    assert api.get("/api/v1/prompts/My fox").status_code == 404


def test_language_model_failures_are_502_and_the_token_never_leaves(api):
    assert api.post("/api/v1/boost", json={"prompt": "cat"}).status_code == 502
    assert api.get("/api/v1/llm").json() == {"configured": False}
    stored = api.put("/api/v1/llm", json={"address": "http://127.0.0.1:8000/v1", "token": "llm-key"}).json()
    assert stored["token_set"] is True and "llm-key" not in str(stored)
    assert "llm-key" not in api.get("/api/v1/llm").text
    assert api.put("/api/v1/llm", json={"address": "http://127.0.0.1:8001/v1"}).json()["token_set"] is True
    assert api.put("/api/v1/llm", json={"address": "http://127.0.0.1:8001/v1", "forget_token": True}) \
        .json()["token_set"] is False
    assert api.get("/api/v1/system-prompts/../../secret.txt").status_code == 404


def test_settings_change_goes_through_the_queue(api):
    api.studio._generator = None  # модель не загружена: выбор только сохраняется
    state = api.get("/api/v1/settings").json()
    assert state["precision"] == "bf16" and "Q4_K_M" in state["precisions"]
    changed = api.patch("/api/v1/settings", json={"sage_attention": True})
    assert changed.status_code == 200 and changed.json()["sage_attention"] is True
    assert api.patch("/api/v1/settings", json={"precision": "fp4"}).status_code == 422


# --- запуск ------------------------------------------------------------------------


def test_api_flags_and_secrets_from_the_environment(monkeypatch):
    monkeypatch.delenv(config.ENV_PASSWORD, raising=False)
    monkeypatch.delenv(config.ENV_API_TOKEN, raising=False)
    plain = config.parse_args([])
    assert plain.api is False and plain.auth is None and plain.api_token is None
    assert config.parse_args(["-api"]).api and config.parse_args(["--api"]).api
    cfg = config.parse_args(["--username", "anna", "--password", "pw", "--api-token", "t"])
    assert cfg.auth == ("anna", "pw") and cfg.api_token == "t"
    assert "pw" not in repr(cfg) and "'t'" not in repr(cfg), "секреты не попадают в журнал"
    with pytest.raises(SystemExit):
        config.parse_args(["--username", "anna"])
    monkeypatch.setenv(config.ENV_PASSWORD, "from-env")
    monkeypatch.setenv(config.ENV_API_TOKEN, "env-token")
    cfg = config.parse_args(["--username", "anna"])
    assert cfg.auth == ("anna", "from-env") and cfg.api_token == "env-token"


def test_main_runs_only_the_api_with_api(monkeypatch):
    from fooocus_qwen import __main__ as entry
    from fooocus_qwen.ui import app

    started = []
    monkeypatch.setattr(server, "run", lambda cfg: started.append(cfg))
    monkeypatch.setattr(app, "launch", lambda cfg: pytest.fail("the UI must not start with --api"))
    assert entry.main(["-api", "--api-token", "t", "--port", "9000"]) == 0
    assert started[0].api and started[0].port == 9000 and started[0].api_token == "t"


def test_the_ui_launch_passes_the_login(monkeypatch):
    from fooocus_qwen.ui import app

    seen = {}

    class Demo:
        def queue(self, **_kwargs):
            pass

        def launch(self, **kwargs):
            seen.update(kwargs)
            # Приложение Gradio появляется в launch — в него встаёт маршрут перевода форматов.
            self.app = FastAPI()

    demo = Demo()
    monkeypatch.setattr(app, "build", lambda cfg, return_studio=False: (demo, None))
    app.start(config.AppConfig(username="anna", password="pw"))
    assert seen["auth"] == ("anna", "pw") and seen["auth_message"]
    assert any(getattr(route, "path", "") == "/fooocus/convert-image" for route in demo.app.router.routes)
    app.start(config.AppConfig())
    assert seen["auth"] is None

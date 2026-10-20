"""Движок видео: установка FreeVideo по ``machine.json`` и разговор с драйвером.

Ни FreeVideo, ни его интерпретатор здесь не запускаются. Процесс драйвера —
фальшивка с тем же лицом, что у ``subprocess.Popen``: stdin принимает
JSON-команды, stdout отдаёт строки событий с префиксом, а что отвечать на
команду, решает сценарий теста. Так проверяется весь путь
``VideoEngine`` — запуск, ожидание ``ready``, отбор событий своего задания,
отмена, выход процесса — без видеокарты.
"""

from __future__ import annotations

import json
import queue
import sys
from pathlib import Path

import pytest

from fooocus_qwen.video import engine
from fooocus_qwen.video.engine import VideoCancelled, VideoEngine, VideoEngineError


def _install(base: Path, ready=True, python: Path | None | str = "make") -> Path:
    """Каталог «установленного» FreeVideo: пакет и ``machine.json``."""
    (base / "freevideo_engine").mkdir(parents=True, exist_ok=True)
    if python == "make":
        python = base / "envs" / "python.exe"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"")
    machine = {"ready": ready, "python": str(python) if python else None}
    engine.machine_file(base).write_text(json.dumps(machine), encoding="utf-8")
    return base


# --- состояние установки ---


def test_status_without_a_clone(tmp_path):
    found = engine.status(tmp_path)
    assert (found.cloned, found.ready) == (False, False)
    assert "not installed" in found.reason


def test_status_without_machine_json(tmp_path):
    (tmp_path / "freevideo_engine").mkdir()
    found = engine.status(tmp_path)
    assert (found.cloned, found.ready) == (True, False)
    assert "has not finished" in found.reason


def test_status_with_a_broken_machine_json(tmp_path):
    (tmp_path / "freevideo_engine").mkdir()
    engine.machine_file(tmp_path).write_text("{oops", encoding="utf-8")
    found = engine.status(tmp_path)
    assert found.cloned and not found.ready
    assert "unreadable" in found.reason


def test_status_when_setup_is_not_ready(tmp_path):
    found = engine.status(_install(tmp_path, ready=False))
    assert found.cloned and not found.ready
    assert "incomplete" in found.reason


@pytest.mark.parametrize("python", [None, "missing"])
def test_status_without_the_python_environment(tmp_path, python):
    if python == "missing":
        python = tmp_path / "envs" / "nowhere.exe"
    found = engine.status(_install(tmp_path, python=python))
    assert found.cloned and not found.ready
    assert "Python environment is missing" in found.reason


def test_status_when_everything_is_in_place(tmp_path):
    assert engine.status(_install(tmp_path)) == engine.EngineStatus(True, True)


# --- окружение и установка ---


def test_environment_keeps_temporary_files_inside_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "our-venv"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "our-home"))
    env = engine.environment(tmp_path)

    for name in ("TEMP", "TMP", "TMPDIR"):
        assert Path(env[name]) == tmp_path / "tmp"
    assert (tmp_path / "tmp").is_dir()
    assert Path(env["HF_HOME"]).is_relative_to(tmp_path)
    assert Path(env["FREEVIDEO_DOCUMENTS_HOME"]).is_relative_to(tmp_path)
    assert env["FREEVIDEO_HOME"] == env["PYTHONPATH"] == str(tmp_path)
    assert "VIRTUAL_ENV" not in env and "PYTHONHOME" not in env


def test_setup_command_uses_official_sources(tmp_path):
    command = engine.setup_command(tmp_path)
    expected = "setup.cmd" if sys.platform == "win32" else "setup.sh"
    assert Path(command[0]) == tmp_path / expected
    assert command[command.index("--network") + 1] == "official"
    assert command[command.index("--root") + 1] == str(tmp_path)
    assert {"--yes", "--accept-model-license", "--plain"} <= set(command)
    assert "--reuse-models" not in command


def test_setup_command_can_reuse_downloaded_weights(tmp_path):
    command = engine.setup_command(tmp_path, reuse_models=tmp_path / "comfy" / "models")
    assert command[command.index("--reuse-models") + 1] == str(tmp_path / "comfy" / "models")


# --- прогресс ---


def test_progress_takes_the_overall_fraction():
    view = engine.progress_view({
        "event": "progress", "part": 0, "parts": 1,
        "value": {"label": "Sampling", "overall": {"fraction": 0.25, "remaining_seconds": 90}},
    })
    assert view.fraction == pytest.approx(0.25)
    assert view.label == "Sampling"
    assert view.remaining_seconds == 90.0
    assert view.step is None


def test_progress_of_a_part_is_a_share_of_the_whole_video():
    view = engine.progress_view({
        "event": "progress", "part": 1, "parts": 4, "value": {"overall": {"fraction": 0.5}},
    })
    assert view.fraction == pytest.approx((1 + 0.5) / 4)


def test_progress_never_leaves_zero_to_one():
    over = engine.progress_view({"event": "progress", "part": 3, "parts": 2, "value": {"overall": {"fraction": 7}}})
    under = engine.progress_view({"event": "progress", "value": {"overall": {"fraction": -1}}})
    assert over.fraction == 1.0
    assert under.fraction == 0.0


def test_sampling_steps_come_from_done_and_total():
    view = engine.progress_view({
        "event": "progress", "value": {"done": 3, "total": 8, "timing_phase": "sampling", "label": "Base pass"},
    })
    assert view.step == (3, 8)
    assert view.fraction == pytest.approx(3 / 8), "без overall доля — по шагам"
    assert view.remaining_seconds is None


def test_steps_of_other_phases_are_not_sampling_steps():
    view = engine.progress_view({"event": "progress", "value": {"done": 1, "total": 2, "timing_phase": "decode"}})
    assert view.step is None


def test_a_progress_without_numbers_has_no_fraction():
    view = engine.progress_view({"event": "progress", "value": {"label": "Joining parts", "phase": "join"}})
    assert view.fraction is None and view.label == "Joining parts"


@pytest.mark.parametrize("event", [{"event": "plan"}, {"event": "done"}, {}])
def test_other_events_are_not_progress(event):
    assert engine.progress_view(event) is None


# --- драйвер ---


class _Stdin:
    def __init__(self, process: FakeDriver) -> None:
        self._process = process
        self.closed = False

    def write(self, text: str) -> None:
        for line in text.splitlines():
            if line.strip():
                self._process.receive(json.loads(line))

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True
        self._process.finish(0)


class FakeDriver:
    """Процесс драйвера: отвечает событиями на команды по сценарию теста.

    ``script(command) -> list`` — что ответить на команду: словари событий,
    сырые строки (как печать библиотек) или ``("exit", код)`` — процесс
    умирает. stdout блокируется, пока ответа нет, как настоящий канал.
    """

    def __init__(self, script, greeting=True, exit_code=0) -> None:
        self.script = script
        self.commands: list[dict] = []
        self.args = None
        self.kwargs = None
        self.returncode = None
        self.exit_code = exit_code
        self._lines: queue.Queue = queue.Queue()
        self.stdin = _Stdin(self)
        self.stdout = self._read()
        self.killed = False
        if greeting:
            self._lines.put(engine.EVENT_PREFIX + json.dumps({"event": "ready", "python": "3.12"}) + "\n")

    def __call__(self, args, **kwargs):
        # Сам объект — и popen, и процесс: так тест видит, с чем его запустили.
        self.args, self.kwargs = args, kwargs
        return self

    def receive(self, command: dict) -> None:
        self.commands.append(command)
        for answer in self.script(command):
            if isinstance(answer, tuple):
                self.finish(answer[1])
            elif isinstance(answer, str):
                self._lines.put(answer + "\n")
            else:
                self._lines.put(engine.EVENT_PREFIX + json.dumps(answer) + "\n")

    def finish(self, code: int) -> None:
        self.exit_code = code
        self._lines.put(None)

    def _read(self):
        while True:
            line = self._lines.get(timeout=10)
            if line is None:
                self.returncode = self.exit_code
                return
            yield line

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.exit_code

    def kill(self):
        self.killed = True


def _engine(tmp_path, script, **driver) -> tuple[VideoEngine, FakeDriver]:
    fake = FakeDriver(script, **driver)
    return VideoEngine(base=_install(tmp_path), popen=fake), fake


def _generating(*events):
    """Сценарий: на ``generate`` — перечисленные события с id задания."""
    def script(command):
        if command.get("cmd") != "generate":
            return []
        return [dict(event, id=command["id"]) if isinstance(event, dict) else event for event in events]
    return script


JOB = {"prompts": ["x"], "width": 864, "height": 480, "seconds": 5.0, "seed": 1}


def test_generate_runs_the_driver_and_returns_done(tmp_path):
    progress = {"event": "progress", "part": 0, "parts": 1, "value": {"overall": {"fraction": 0.5}}}
    done = {"event": "done", "output": "out.mp4", "parts": ["out.mp4"]}
    video, fake = _engine(tmp_path, _generating(progress, "torch noise on stdout", done))
    seen = []

    result = video.generate(JOB, seen.append)

    assert result["output"] == "out.mp4" and result["parts"] == ["out.mp4"]
    assert [event["event"] for event in seen] == ["progress"]
    command = fake.commands[0]
    assert command["cmd"] == "generate" and command["id"] == "job1"
    assert {key: command[key] for key in JOB} == JOB
    assert video.busy is False
    # Запуск — интерпретатором FreeVideo, с его окружением и в его каталоге.
    machine = json.loads(engine.machine_file(tmp_path).read_text(encoding="utf-8"))
    assert fake.args[0] == machine["python"] and fake.args[-1] == str(engine.DRIVER)
    assert fake.kwargs["cwd"] == str(tmp_path)
    assert Path(fake.kwargs["env"]["TEMP"]) == tmp_path / "tmp"


def test_the_process_lives_between_jobs(tmp_path):
    done = {"event": "done", "output": "out.mp4", "parts": []}
    video, fake = _engine(tmp_path, _generating(done))
    starts = []
    video._popen = lambda *args, **kwargs: starts.append(1) or fake(*args, **kwargs)

    video.generate(JOB)
    video.generate(JOB)

    assert len(starts) == 1
    assert [command["id"] for command in fake.commands] == ["job1", "job2"]


def test_events_of_another_job_are_skipped(tmp_path):
    def script(command):
        return [
            {"event": "done", "id": "stale", "output": "old.mp4"},
            {"event": "progress", "id": "stale", "value": {}},
            {"event": "done", "id": command["id"], "output": "new.mp4"},
        ]

    video, _fake = _engine(tmp_path, script)
    seen = []
    assert video.generate(JOB, seen.append)["output"] == "new.mp4"
    assert seen == []


def test_an_error_event_becomes_an_engine_error(tmp_path):
    video, _fake = _engine(tmp_path, _generating({"event": "error", "message": "CUDA is not available"}))
    with pytest.raises(VideoEngineError, match="CUDA is not available"):
        video.generate(JOB)
    assert video.busy is False


def test_an_error_without_text_still_says_something(tmp_path):
    video, _fake = _engine(tmp_path, _generating({"event": "error"}))
    with pytest.raises(VideoEngineError, match="FreeVideo failed"):
        video.generate(JOB)


def test_cancel_sends_the_command_and_ends_with_video_cancelled(tmp_path):
    def script(command):
        if command["cmd"] == "generate":
            script.job = command["id"]
            return [{"event": "progress", "id": command["id"], "value": {}}]
        if command["cmd"] == "cancel":
            return [{"event": "cancelled", "id": script.job}]
        return []

    video, fake = _engine(tmp_path, script)
    with pytest.raises(VideoCancelled):
        video.generate(JOB, lambda _event: video.cancel())
    assert fake.commands[-1] == {"cmd": "cancel"}
    assert isinstance(VideoCancelled("x"), VideoEngineError), "отмена ловится и как сбой движка"


def test_cancel_without_a_job_sends_nothing(tmp_path):
    video, fake = _engine(tmp_path, _generating({"event": "done", "output": "a.mp4"}))
    video.cancel()
    video.generate(JOB)
    video.cancel()
    assert [command["cmd"] for command in fake.commands] == ["generate"]


def test_a_driver_that_exits_is_an_engine_error_and_restarts_next_time(tmp_path):
    video, fake = _engine(tmp_path, _generating(("exit", 3)))
    with pytest.raises(VideoEngineError, match="exited with code 3"):
        video.generate(JOB)
    assert video._process is None


def test_a_driver_that_does_not_greet_is_an_engine_error(tmp_path):
    video, fake = _engine(tmp_path, lambda _command: [], greeting=False)
    fake._lines.put(engine.EVENT_PREFIX + json.dumps({"event": "error", "message": "no module"}) + "\n")
    with pytest.raises(VideoEngineError, match="did not start"):
        video.generate(JOB)


def test_generate_without_an_installed_engine_says_why(tmp_path):
    video = VideoEngine(base=tmp_path, popen=lambda *_a, **_k: pytest.fail("the driver must not start"))
    with pytest.raises(VideoEngineError, match="not installed"):
        video.generate(JOB)


def test_release_asks_the_driver_to_free_the_gpu(tmp_path):
    def script(command):
        if command["cmd"] == "release":
            return [{"event": "released", "id": command["id"]}]
        return [{"event": "done", "id": command["id"], "output": "a.mp4"}]

    video, fake = _engine(tmp_path, script)
    video.generate(JOB)
    video.release(timeout=5)
    assert fake.commands[-1] == {"cmd": "release", "id": "release"}


def test_release_without_a_process_does_nothing(tmp_path):
    video, fake = _engine(tmp_path, lambda _command: [])
    video.release()
    assert fake.commands == [] and fake.args is None, "ради отпускания драйвер не запускается"


def test_close_closes_stdin_of_the_driver(tmp_path):
    video, fake = _engine(tmp_path, _generating({"event": "done", "output": "a.mp4"}))
    video.generate(JOB)
    video.close()
    assert fake.stdin.closed and video._process is None
    assert not fake.killed



# --- правки исходников FreeVideo ---


def _git(*args, cwd):
    import subprocess

    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _clone_with(tmp_path, text):
    base = tmp_path / "FreeVideo"
    base.mkdir()
    _git("init", "-q", cwd=base)
    _git("config", "core.autocrlf", "false", cwd=base)
    (base / "module.py").write_bytes(text)
    _git("add", ".", cwd=base)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base", cwd=base)
    return base


def _patches(tmp_path, monkeypatch):
    directory = tmp_path / "patches"
    directory.mkdir()
    (directory / "0001-change.patch").write_bytes(
        b"--- a/module.py\n+++ b/module.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n")
    monkeypatch.setattr(engine, "PATCHES", directory)


def test_patches_apply_once_and_then_count_as_applied(tmp_path, monkeypatch):
    _patches(tmp_path, monkeypatch)
    base = _clone_with(tmp_path, b"value = 1\n")
    said = []
    assert engine.apply_patches(base, said.append) == ["0001-change.patch"]
    assert (base / "module.py").read_bytes() == b"value = 2\n"
    assert engine.apply_patches(base, said.append) == [], "наложенный патч не накладывается второй раз"
    assert said == ["Applying FreeVideo patch 0001-change.patch"]


def test_a_patch_that_does_not_fit_is_refused(tmp_path, monkeypatch):
    _patches(tmp_path, monkeypatch)
    base = _clone_with(tmp_path, b"value = 7\n")
    with pytest.raises(engine.VideoEngineError, match="does not apply"):
        engine.apply_patches(base)
    assert (base / "module.py").read_bytes() == b"value = 7\n"


def test_the_shipped_patches_are_byte_exact_lf():
    """CRLF сломал бы git apply: .gitattributes держит *.patch как есть."""
    patches = sorted(engine.PATCHES.glob("*.patch"))
    assert patches, "патч финального апскейла на месте"
    for patch in patches:
        assert b"\r\n" not in patch.read_bytes(), patch.name

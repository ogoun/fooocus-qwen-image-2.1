"""Драйвер FreeVideo: план частей, цепочка продолжений, отмена и протокол.

Драйвер исполняется в окружении FreeVideo, но это обычный модуль без
импортов приложения — его можно импортировать и здесь. Пакет
``freevideo_engine`` подменяется фальшивками (``types.ModuleType``): план
частей задаёт тест, «генерация» лишь записывает, с чем её позвали, и
возвращает имя файла. Видеокарта и сам FreeVideo не нужны.
"""

from __future__ import annotations

import io
import json
import sys
import types

import pytest

from fooocus_qwen.video import engine
from fooocus_qwen.video import freevideo_driver as driver

QUALITY = {"two_pass": True, "base_steps": 8, "refine_steps": 3, "step_cache": None, "upscale": 2}


class FakeEngine:
    """Пакет ``freevideo_engine`` с планом, оркестратором и склейкой."""

    def __init__(self, blocks: list[int]) -> None:
        self.blocks = blocks
        self.calls: list[dict] = []
        self.joined: list[tuple] = []
        self.planned: list[tuple] = []
        self.released: list[str] = []
        self.leads: list = []

    def plan(self, seconds, continued=False):
        self.planned.append((seconds, continued))
        frames = sum(self.blocks)
        return {"seconds": seconds, "frames": frames, "single": len(self.blocks) == 1,
                "blocks": [{"frames": count, "seconds": count / 24} for count in self.blocks]}

    def generate(self, prompt, width, height, seconds, seed, output_directory, **options):
        self.calls.append(dict(prompt=prompt, width=width, height=height, seconds=seconds, seed=seed,
                               output_directory=output_directory, **options))
        options["progress"]({"label": "Sampling", "done": 1, "total": 8, "frame_preview": object()})
        return f"{output_directory}/part{len(self.calls)}.mp4"

    def join_files(self, sources, target, cancelled=None, frames=None, leads=None):
        self.joined.append((list(sources), target, frames, cancelled))
        self.leads.append(leads)
        return {"frames": 999}

    def install(self, monkeypatch) -> None:
        package = types.ModuleType("freevideo_engine")
        chain_plan = types.ModuleType("freevideo_engine.chain_plan")
        chain_plan.plan = self.plan
        comfy_bridge = types.ModuleType("freevideo_engine.comfy_bridge")
        comfy_bridge.generate = self.generate
        chain_join = types.ModuleType("freevideo_engine.chain_join")
        chain_join.join_files = self.join_files
        resident = types.ModuleType("freevideo_engine.resident_process")
        resident.OWNER = types.SimpleNamespace(release=self.released.append)
        package.chain_plan, package.comfy_bridge = chain_plan, comfy_bridge
        package.chain_join, package.resident_process = chain_join, resident
        for name, module in (("freevideo_engine", package), ("freevideo_engine.chain_plan", chain_plan),
                             ("freevideo_engine.comfy_bridge", comfy_bridge),
                             ("freevideo_engine.chain_join", chain_join),
                             ("freevideo_engine.resident_process", resident)):
            monkeypatch.setitem(sys.modules, name, module)


class _Captured:
    """Вывод драйвера за тест: ``capsys`` копится, читать можно много раз.

    Подменить ``sys.stdout`` в фикстуре нельзя — pytest ставит свой перехват
    уже после фикстур, на время самого теста.
    """

    def __init__(self, capsys) -> None:
        self._capsys = capsys
        self._text = ""

    def getvalue(self) -> str:
        self._text += self._capsys.readouterr().out
        return self._text


@pytest.fixture
def stdout(capsys):
    return _Captured(capsys)


@pytest.fixture(autouse=True)
def clean_cancel():
    driver._cancel.clear()
    yield
    driver._cancel.clear()


def _events(buffer) -> list[dict]:
    lines = buffer.getvalue().splitlines()
    assert all(line.startswith(driver.EVENT_PREFIX) for line in lines), lines
    return [json.loads(line[len(driver.EVENT_PREFIX):]) for line in lines]


def _fake(monkeypatch, blocks) -> FakeEngine:
    fake = FakeEngine(blocks)
    fake.install(monkeypatch)
    return fake


def _job(**changes) -> dict:
    job = {"id": "job1", "prompts": ["shot"], "width": 864, "height": 480, "seconds": 5.0, "seed": 10,
           "quality": dict(QUALITY), "preview": False, "media": None, "continuation_media": None,
           "continue_from": None, "refine_from": None, "output_dir": "runs"}
    job.update(changes)
    return job


# --- протокол ---


def test_the_prefix_is_the_same_on_both_sides():
    assert driver.EVENT_PREFIX == engine.EVENT_PREFIX


def test_emit_writes_one_prefixed_json_line(stdout):
    driver.emit("progress", id="job1", value={"label": "Кадры"})
    assert stdout.getvalue().count("\n") == 1 and stdout.getvalue().endswith("\n")
    assert _events(stdout) == [{"event": "progress", "id": "job1", "value": {"label": "Кадры"}}]


def test_plain_drops_previews_and_non_json_values():
    value = driver._plain({"frame_preview": b"pixels", "eta": float("nan"), "path": object(), "list": tuple(range(40))})
    assert "frame_preview" not in value
    assert value["eta"] is None
    assert isinstance(value["path"], str)
    assert len(value["list"]) == 32
    json.dumps(value)


def test_interrupted_raises_only_after_cancel():
    driver.interrupted()
    driver._cancel.set()
    with pytest.raises(KeyboardInterrupt):
        driver.interrupted()


# --- задание ---


def test_a_single_part(monkeypatch, stdout):
    fake = _fake(monkeypatch, [124])
    media = {"version": 1, "first": "first.png"}
    driver.run_job(_job(media=media))

    (call,) = fake.calls
    assert call["frames"] == 124 and call["seconds"] == pytest.approx(124 / 24)
    assert call["seed"] == 10 and call["continue_from"] is None
    assert call["first_frame_reference"] is False
    assert call["export_inputs"](None, None) == {"media": media}
    assert call["interrupted"] is driver.interrupted
    assert fake.joined == []
    events = _events(stdout)
    assert [event["event"] for event in events] == ["plan", "progress", "part", "done"]
    assert "frame_preview" not in events[1]["value"]
    assert events[-1] == {"event": "done", "id": "job1", "output": "runs/part1.mp4", "parts": ["runs/part1.mp4"],
                          "last": "runs/part1.mp4", "frames": None}


def test_without_media_no_inputs_are_exported(monkeypatch, stdout):
    fake = _fake(monkeypatch, [124])
    driver.run_job(_job())
    assert fake.calls[0]["export_inputs"] is None


def test_several_parts_chain_and_join(monkeypatch, stdout):
    fake = _fake(monkeypatch, [361, 233, 233])
    continuation = {"version": 1, "references": [], "loras": []}
    driver.run_job(_job(seconds=33.0, prompts=["first", "second"], media={"first": "f.png"},
                        continuation_media=continuation))

    assert [call["seed"] for call in fake.calls] == [10, 11, 12]
    assert [call["continue_from"] for call in fake.calls] == [None, "runs/part1.mp4", "runs/part2.mp4"]
    assert [call["first_frame_reference"] for call in fake.calls] == [False, True, True]
    # Последний промт продолжает и все следующие части.
    assert [call["prompt"] for call in fake.calls] == ["first", "second", "second"]
    assert fake.calls[0]["export_inputs"](None, None) == {"media": {"first": "f.png"}}
    assert fake.calls[1]["export_inputs"](None, None) == {"media": continuation}
    ((sources, target, frames, cancelled),) = fake.joined
    assert sources == ["runs/part1.mp4", "runs/part2.mp4", "runs/part3.mp4"]
    assert target.replace("\\", "/") == "runs/joined.mp4"
    assert frames == 361 + 233 + 233
    assert cancelled() is False
    done = _events(stdout)[-1]
    assert done["event"] == "done" and done["output"] == target and done["parts"] == sources
    assert done["last"] == "runs/part3.mp4"
    progress = [event for event in _events(stdout) if event["event"] == "progress"]
    assert [event["part"] for event in progress] == [0, 1, 2, 3]
    assert {event["parts"] for event in progress} == {3}


def test_continuing_a_video_starts_from_it(monkeypatch, stdout):
    fake = _fake(monkeypatch, [141])
    driver.run_job(_job(continue_from="old.mp4"))
    assert fake.planned == [(5.0, True)]
    assert fake.calls[0]["continue_from"] == "old.mp4"
    assert fake.calls[0]["first_frame_reference"] is True


def test_a_draft_is_one_part(monkeypatch, stdout):
    fake = _fake(monkeypatch, [124])
    driver.run_job(_job(preview=True))
    assert len(fake.calls) == 1 and fake.calls[0]["preview"] is True


def test_a_long_draft_is_refused_before_any_generation(monkeypatch, stdout):
    fake = _fake(monkeypatch, [361, 233])
    with pytest.raises(ValueError, match="draft is limited to one part"):
        driver.run_job(_job(seconds=25.0, preview=True))
    assert fake.calls == []


def test_finishing_a_draft_is_one_part_too(monkeypatch, stdout):
    fake = _fake(monkeypatch, [124])
    driver.run_job(_job(refine_from="draft/a.mp4"))
    assert fake.calls[0]["refine_from"] == "draft/a.mp4"
    with pytest.raises(ValueError):
        _fake(monkeypatch, [361, 233])
        driver.run_job(_job(refine_from="draft/a.mp4"))


def test_quality_reaches_the_orchestrator(monkeypatch, stdout):
    fake = _fake(monkeypatch, [124])
    quality = {"two_pass": False, "base_steps": 20, "refine_steps": 3, "step_cache": "easycache-av", "upscale": 1.5}
    driver.run_job(_job(quality=quality))
    call = fake.calls[0]
    assert (call["two_pass"], call["base_steps"], call["refine_steps"]) == (False, 20, 3)
    assert call["step_cache"] == "easycache-av" and call["upscale_factor"] == 1.5


def test_a_new_job_forgets_an_old_cancel(monkeypatch, stdout):
    fake = _fake(monkeypatch, [124])
    driver._cancel.set()
    driver.run_job(_job())
    assert len(fake.calls) == 1


# --- главный цикл ---


def _main(monkeypatch, commands: list) -> tuple[int, list[dict]]:
    lines = [command if isinstance(command, str) else json.dumps(command) for command in commands]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(lines) + "\n"))
    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    code = driver.main()
    return code, _events(buffer)


def test_main_greets_runs_the_job_and_quits(monkeypatch):
    fake = _fake(monkeypatch, [124])
    code, events = _main(monkeypatch, [dict(_job(), cmd="generate")])
    assert code == 0
    assert events[0]["event"] == "ready"
    assert events[-1]["event"] == "done" and events[-1]["id"] == "job1"
    assert len(fake.calls) == 1
    assert fake.released == ["application closed"], "закрытый stdin отпускает модель"


def test_main_reports_an_unknown_command_and_lives_on(monkeypatch):
    _fake(monkeypatch, [124])
    code, events = _main(monkeypatch, [{"cmd": "dance", "id": "x"}, "not json", "",
                                       {"cmd": "release", "id": "r1", "reason": "qwen"}])
    assert code == 0
    kinds = [(event["event"], event.get("id")) for event in events]
    assert ("error", "x") in kinds and ("error", None) in kinds and ("released", "r1") in kinds
    unknown = next(event for event in events if event.get("id") == "x")
    assert "Unknown command: dance" in unknown["message"]


def test_main_turns_a_failure_into_an_error_event(monkeypatch):
    fake = _fake(monkeypatch, [124])

    def broken(*_args, **_kwargs):
        raise RuntimeError("Out of memory in block 3")

    sys.modules["freevideo_engine.comfy_bridge"].generate = broken
    _code, events = _main(monkeypatch, [dict(_job(), cmd="generate")])
    error = events[-1]
    assert error["event"] == "error" and error["id"] == "job1"
    assert error["message"] == "Out of memory in block 3" and error["kind"] == "RuntimeError"
    assert fake.calls == []


def test_main_turns_an_interrupt_into_cancelled(monkeypatch):
    _fake(monkeypatch, [124])

    def stopped(*_args, **_kwargs):
        raise KeyboardInterrupt

    sys.modules["freevideo_engine.comfy_bridge"].generate = stopped
    _code, events = _main(monkeypatch, [dict(_job(), cmd="generate")])
    assert events[-1] == {"event": "cancelled", "id": "job1"}


def test_main_quits_on_command(monkeypatch):
    fake = _fake(monkeypatch, [124])
    code, events = _main(monkeypatch, [{"cmd": "quit"}, dict(_job(), cmd="generate")])
    assert code == 0
    assert [event["event"] for event in events] == ["ready"]
    assert fake.calls == []


def test_a_memory_budget_reaches_the_engine_only_for_its_job(monkeypatch):
    """Бюджет задания подмешивается к настройкам ресурсов, которые оркестратор
    превращает в ключи CLI, — и снимается после задания."""
    settings = types.ModuleType("freevideo_engine.resource_settings")
    settings.read = lambda root, *, backend="cuda": {"gpu_reserve_gib": None}
    package = types.ModuleType("freevideo_engine")
    package.resource_settings = settings
    monkeypatch.setitem(sys.modules, "freevideo_engine", package)
    monkeypatch.setitem(sys.modules, "freevideo_engine.resource_settings", settings)
    original = settings.read
    seen = []
    with driver.budget(8, 16):
        seen.append(settings.read("root"))
    assert seen == [{"gpu_reserve_gib": None, "vram_gib": 8.0, "ram_gib": 16.0}]
    assert settings.read is original
    with driver.budget():
        assert settings.read is original


def test_command_lines_reads_a_plain_stream():
    assert list(driver.command_lines(io.StringIO('{"cmd": "quit"}\n'))) == ['{"cmd": "quit"}\n']


CHILD = """
import importlib.util, sys, threading, time
spec = importlib.util.spec_from_file_location("driver", sys.argv[1])
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
threading.Thread(target=lambda: [print("line", line.strip(), flush=True) for line in driver.command_lines()],
                 daemon=True).start()
time.sleep(0.5)
started = time.perf_counter()
import numpy
print("imported", round(time.perf_counter() - started, 2), flush=True)
time.sleep(10)
"""


@pytest.mark.skipif(sys.platform != "win32", reason="the loader lock on stdin is a Windows trait")
def test_waiting_for_a_command_does_not_block_loading_a_dll(tmp_path):
    """Ожидание команды не держит загрузку DLL: так зависало продление ролика.

    Ребёнок ждёт команд в соседнем потоке и импортирует numpy; при чтении
    stdin «в лоб» импорт стоял бы до первой строки — а её здесь не шлют,
    пока импорт не отчитается.
    """
    import subprocess

    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    child = subprocess.Popen([sys.executable, "-I", str(script), driver.__file__], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
    try:
        first = child.stdout.readline()
        child.stdin.write("hello\n")
        child.stdin.flush()
        second = child.stdout.readline()
    finally:
        child.kill()
        child.wait(timeout=30)
    assert first.startswith("imported"), first
    assert second.strip() == "line hello"


def test_an_extension_is_joined_to_its_source(monkeypatch, stdout):
    """«Продлить» отдаёт ролик целиком: исходник и новая часть одной склейкой.

    Обрезки до плана нет — длина исходника в план продолжения не входит.
    Следующее продление пойдёт от новой части: запись латентов лежит у неё.
    """
    fake = _fake(monkeypatch, [136])
    driver.run_job(_job(continue_from="old.mp4"))
    ((sources, target, frames, _cancelled),) = fake.joined
    assert sources == ["old.mp4", "runs/part1.mp4"] and frames is None
    done = _events(stdout)[-1]
    assert done["output"] == target and done["parts"] == ["runs/part1.mp4"]
    assert done["last"] == "runs/part1.mp4" and done["frames"] == 999


def test_a_joined_video_is_extended_from_its_last_part_and_joined_whole(monkeypatch, stdout, tmp_path):
    """Продлённый ролик: движок продолжает последнюю часть, склеивается целое.

    Звук стыка движок сверяет по записи новой части, а она ссылается на
    последнюю часть, не на склейку, — ``leads`` подменяет предыдущий файл.
    """
    fake = _fake(monkeypatch, [136])
    lead_calls = []
    continuation = types.ModuleType("freevideo_engine.comfy_continuation")
    continuation.audio_lead_in = lambda video, previous=None: lead_calls.append((video, previous)) or "lead"
    monkeypatch.setitem(sys.modules, "freevideo_engine.comfy_continuation", continuation)
    whole, last = str(tmp_path / "joined.mp4"), str(tmp_path / "video.mp4")

    driver.run_job(_job(continue_from=last, continue_whole=whole))
    assert fake.calls[0]["continue_from"] == last
    ((sources, _target, frames, _cancelled),) = fake.joined
    assert sources == [whole, "runs/part1.mp4"] and frames is None
    (leads,) = fake.leads
    assert leads("runs/part1.mp4", whole) == "lead"
    assert lead_calls == [("runs/part1.mp4", last)]
    assert leads("runs/part2.mp4", "runs/part1.mp4") is None, "другие стыки движок читает сам"


def test_a_plain_extension_needs_no_seam_substitute(monkeypatch, stdout):
    fake = _fake(monkeypatch, [136])
    driver.run_job(_job(continue_from="old.mp4"))
    assert fake.leads == [None]



def test_the_final_upscale_reaches_the_orchestrator(monkeypatch, stdout):
    fake = _fake(monkeypatch, [124])
    driver.run_job(_job(final_upscale=[1728, 960]))
    assert fake.calls[0]["final_upscale"] == (1728, 960)


def test_without_upscale_the_orchestrator_gets_no_flag(monkeypatch, stdout):
    """Без апскейла ключа нет вовсе: оркестратор без нашей правки его бы не принял."""
    fake = _fake(monkeypatch, [124])
    driver.run_job(_job())
    assert "final_upscale" not in fake.calls[0]

"""Движок видео: установка FreeVideo и управление его драйвером.

FreeVideo (https://github.com/FlashML-org/FreeVideo, Apache-2.0) — движок
инференса MiniMax H3 под малую память. Он работает от 8 ГБ VRAM и 16 ГБ RAM
за счёт int8 ConvRot, поблочной асинхронной подкачки трансформера, таблиц
AdaLN вместо проекций и тайлового VAE (docs/research/2026-10-19-video-freevideo.md).
Окружение у него своё: Python 3.12, torch 2.13 cu130 и пропатченный
diffusers. С нашим окружением оно несовместимо, поэтому движок живёт отдельно
в ``config.FREEVIDEO_DIR`` и запускается подпроцессом.

Устройство:

* ``install()`` — клон на закреплённом теге плюс headless-установка его
  собственным ``setup``. Всё, что установка пишет, остаётся внутри каталога:
  bootstrap Python, окружение, кэши HF, uv, Triton, временные файлы.
* ``VideoEngine`` — один долгоживущий процесс драйвера
  (``freevideo_driver.py``) в окружении FreeVideo. Команды уходят в stdin,
  события приходят из stdout. Генерации идут строго по одной: замок здесь,
  как у генератора изображений.

Видеопамять делится с Qwen. Перед заданием её освобождает вызывающий
(``Studio.claim_gpu``); FreeVideo сам планирует размещение по тому, что
реально свободно, — так получается его заявленный расход, а не «24 минус
Qwen».
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import config

LOGGER = logging.getLogger(__name__)

REPOSITORY = "https://github.com/FlashML-org/FreeVideo.git"
# Закреплённый выпуск: и код оркестратора, и форматы кэша весов меняются от
# версии к версии, а драйвер опирается на ``comfy_bridge.generate``.
TAG = "v0.4.15"
COMMIT = "71129a7123292e0612f3b7f3ff19f053dd2f72a1"

DRIVER = Path(__file__).with_name("freevideo_driver.py")
EVENT_PREFIX = "@@fooocus-video@@ "

# Сколько ждать готовности драйвера: импорт оркестратора без torch — секунды.
_START_TIMEOUT = 120.0

EventCallback = Callable[[dict], None]


class VideoEngineError(RuntimeError):
    """Сбой движка видео — текст для строки состояния, по-английски."""


class VideoCancelled(VideoEngineError):
    """Генерацию прервал пользователь."""


@dataclass(frozen=True)
class EngineStatus:
    """Состояние установки: есть ли клон, окружение и готова ли машина."""

    cloned: bool
    ready: bool
    reason: str = ""


def root() -> Path:
    return config.FREEVIDEO_DIR


def machine_file(base: Path | None = None) -> Path:
    return (base or root()) / "machine.json"


def status(base: Path | None = None) -> EngineStatus:
    """Готов ли движок — по ``machine.json``, который пишет конец установки FreeVideo."""
    base = base or root()
    if not (base / "freevideo_engine").is_dir():
        return EngineStatus(False, False, "FreeVideo is not installed")
    try:
        machine = json.loads(machine_file(base).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return EngineStatus(True, False, "FreeVideo setup has not finished")
    except (OSError, ValueError) as error:
        return EngineStatus(True, False, f"FreeVideo machine.json is unreadable: {error}")
    if not machine.get("ready"):
        return EngineStatus(True, False, "FreeVideo setup is incomplete")
    python = Path(machine.get("python") or "")
    if not python.is_file():
        return EngineStatus(True, False, "The FreeVideo Python environment is missing")
    return EngineStatus(True, True)


def environment(base: Path | None = None) -> dict[str, str]:
    """Окружение процессов FreeVideo: всё временное и кэши — внутри его каталога.

    Сама установка FreeVideo уводит в свой корень кэши HF, uv и Triton, но
    ``TEMP``/``TMP`` подменяет лишь тогда, когда системный путь «небезопасен»
    (непредставим в кодовой странице). Иначе временные файлы шли бы в
    ``%TEMP%``, а правило проекта — ничего вне проекта. Настройки лаунчера
    по умолчанию пишутся в «Документы»; ``FREEVIDEO_DOCUMENTS_HOME`` держит их
    рядом.
    """
    base = base or root()
    temporary = base / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        FREEVIDEO_HOME=str(base),
        PYTHONPATH=str(base),
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PYTHONUNBUFFERED="1",
        NO_COLOR="1",
        TEMP=str(temporary),
        TMP=str(temporary),
        TMPDIR=str(temporary),
        FREEVIDEO_DOCUMENTS_HOME=str(base / "documents"),
        HF_HOME=str(base / "downloads" / "huggingface"),
        HF_HUB_DISABLE_TELEMETRY="1",
    )
    # Наш venv не должен просочиться в чужой интерпретатор.
    for name in ("VIRTUAL_ENV", "PYTHONHOME"):
        env.pop(name, None)
    return env


def _creation_flags() -> int:
    if sys.platform != "win32":
        return 0
    return subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP


def setup_command(base: Path, reuse_models: Path | None = None) -> list[str]:
    """Команда headless-установки FreeVideo в ``base``.

    ``--network official`` — только официальные источники (GitHub, PyPI,
    Hugging Face): без него установка пробует и зеркала для Китая.
    ``--reuse-models`` — каталог, где уже лежат веса H3 (например, установка
    ComfyUI): файлы сверяются по sha256 и берутся жёсткой ссылкой, а не
    качаются заново — энкодер весит 15.7 ГБ.
    """
    if sys.platform == "win32":
        command = [str(base / "setup.cmd")]
    else:
        command = [str(base / "setup.sh")]
    command += [
        "--root", str(base), "--yes", "--accept-model-license", "--plain", "--network", "official",
    ]
    if reuse_models is not None:
        command += ["--reuse-models", str(reuse_models)]
    return command


def install(
    on_line: Callable[[str], None] | None = None,
    reuse_models: Path | None = None,
    base: Path | None = None,
) -> None:
    """Ставит FreeVideo: клон на закреплённом теге и его собственная установка.

    Повторный вызов безопасен — установка FreeVideo докачивает недостающее и
    проверяет готовое. Строки её вывода уходят в ``on_line`` (журнал или
    строка состояния интерфейса).
    """
    base = base or root()
    say = on_line or (lambda line: LOGGER.info("%s", line))
    if not (base / "freevideo_engine").is_dir():
        say(f"Cloning FreeVideo {TAG} into {base}")
        _run(["git", "clone", "--quiet", REPOSITORY, str(base)], say)
    _run(["git", "-C", str(base), "fetch", "--quiet", "--tags"], say, check=False)
    _run(["git", "-C", str(base), "checkout", "--quiet", TAG], say)
    head = subprocess.run(
        ["git", "-C", str(base), "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    if head != COMMIT:
        raise VideoEngineError(f"FreeVideo {TAG} resolves to {head}, expected {COMMIT}")
    say("Running FreeVideo setup (environment and model weights, ~40-55 GB)")
    _run(setup_command(base, reuse_models), say, env=environment(base))
    found = status(base)
    if not found.ready:
        raise VideoEngineError(found.reason)


def _run(command: list[str], say: Callable[[str], None], env: dict | None = None, check: bool = True) -> None:
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        env=env, text=True, encoding="utf-8", errors="replace", creationflags=_creation_flags(),
    )
    last = ""
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        # Установка рисует прогресс одной и той же строкой много раз подряд.
        if line and line != last:
            say(line)
            last = line
    code = process.wait()
    if check and code:
        raise VideoEngineError(f"{Path(command[0]).name} exited with code {code}")


@dataclass(frozen=True)
class ProgressView:
    """Событие драйвера для полосы прогресса: доля всего задания, подпись, остаток."""

    fraction: float | None
    label: str
    remaining_seconds: float | None = None
    step: tuple[int, int] | None = None


def progress_view(event: dict) -> ProgressView | None:
    """Доля и подпись из события ``progress`` драйвера; None для прочих событий.

    Оркестратор FreeVideo сам считает долю всего ролика с учётом загрузки,
    кодирования и декодирования (``overall.fraction``, сглаженная по истории
    этой машины) — её и показываем; шаги сэмплинга идут подписью. Длинный
    ролик — несколько частей: доля части пересчитывается в долю целого.
    """
    if event.get("event") != "progress":
        return None
    value = event.get("value") or {}
    overall = value.get("overall") or {}
    fraction = overall.get("fraction")
    if fraction is None and value.get("total"):
        fraction = (value.get("done") or 0) / value["total"]
    parts = max(1, int(event.get("parts") or 1))
    part = min(int(event.get("part") or 0), parts)
    if fraction is not None:
        fraction = min(1.0, (part + max(0.0, min(1.0, float(fraction)))) / parts)
    step = None
    if value.get("total") and value.get("timing_phase") == "sampling":
        step = (int(value.get("done") or 0), int(value["total"]))
    remaining = overall.get("remaining_seconds")
    return ProgressView(
        fraction=fraction,
        label=str(value.get("label") or ""),
        remaining_seconds=float(remaining) if isinstance(remaining, (int, float)) else None,
        step=step,
    )


class VideoEngine:
    """Процесс драйвера FreeVideo и очередь его событий.

    Процесс запускается при первом задании и живёт до закрытия приложения
    (``close``): держатель модели FreeVideo принадлежит ему, и части
    длинного ролика и раскадровки идут без повторной загрузки весов.
    """

    def __init__(self, base: Path | None = None, popen=subprocess.Popen) -> None:
        self._base = base or root()
        self._popen = popen
        self._process: subprocess.Popen | None = None
        self._events: queue.Queue[dict] = queue.Queue()
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._sequence = 0
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        found = status(self._base)
        if not found.ready:
            raise VideoEngineError(found.reason)
        machine = json.loads(machine_file(self._base).read_text(encoding="utf-8"))
        command = [machine["python"], "-B", "-X", "utf8", str(DRIVER)]
        LOGGER.info("Starting the FreeVideo driver")
        self._events = queue.Queue()
        self._process = self._popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=environment(self._base), cwd=str(self._base), text=True, encoding="utf-8",
            errors="replace", bufsize=1, creationflags=_creation_flags(),
        )
        threading.Thread(target=self._read, args=(self._process,), name="freevideo-events", daemon=True).start()
        event = self._next_event(timeout=_START_TIMEOUT)
        if event.get("event") != "ready":
            raise VideoEngineError(f"FreeVideo driver did not start: {event}")

    def _read(self, process) -> None:
        for line in process.stdout:
            line = line.rstrip("\n")
            if line.startswith(EVENT_PREFIX):
                try:
                    self._events.put(json.loads(line[len(EVENT_PREFIX):]))
                except ValueError:
                    LOGGER.warning("Malformed FreeVideo event: %s", line[:200])
            elif line.strip():
                LOGGER.debug("[FreeVideo] %s", line)
        self._events.put({"event": "exit", "code": process.wait()})

    def _next_event(self, timeout: float | None = None) -> dict:
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            raise VideoEngineError("FreeVideo driver did not respond") from None

    def _send(self, command: dict) -> None:
        with self._io_lock:
            if self._process is None or self._process.poll() is not None or self._process.stdin is None:
                raise VideoEngineError("FreeVideo driver is not running")
            self._process.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
            self._process.stdin.flush()

    def generate(self, job: dict, on_event: EventCallback | None = None) -> dict:
        """Выполняет задание драйвера; возвращает событие ``done`` (путь ролика и частей).

        Блокирует до конца задания. События прогресса уходят в ``on_event``.
        """
        with self._lock:
            self._busy = True
            try:
                self._start()
                self._sequence += 1
                job = dict(job, cmd="generate", id=f"job{self._sequence}")
                self._send(job)
                while True:
                    event = self._next_event()
                    kind = event.get("event")
                    if kind == "exit":
                        self._process = None
                        raise VideoEngineError(f"FreeVideo driver exited with code {event.get('code')}")
                    if event.get("id") not in (None, job["id"]):
                        continue
                    if kind == "done":
                        return event
                    if kind == "cancelled":
                        raise VideoCancelled("Video generation was stopped")
                    if kind == "error":
                        raise VideoEngineError(event.get("message") or "FreeVideo failed")
                    if on_event is not None:
                        on_event(event)
            finally:
                self._busy = False

    def cancel(self) -> None:
        """Просит прервать текущее задание. Замок не берётся: он занят именно им."""
        if self._busy:
            try:
                self._send({"cmd": "cancel"})
            except (VideoEngineError, OSError) as error:
                LOGGER.warning("Could not cancel the video job: %s", error)

    def release(self, timeout: float = 30.0) -> None:
        """Отпускает видеопамять, которую держит процесс модели между заданиями.

        Зовётся перед загрузкой Qwen: модель FreeVideo живёт в отдельном
        процессе, и без этого две модели делили бы одну карту.
        """
        if self._process is None or self._process.poll() is not None:
            return
        with self._lock:
            try:
                self._send({"cmd": "release", "id": "release"})
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    event = self._next_event(timeout=max(0.1, deadline - time.monotonic()))
                    if event.get("event") in ("released", "exit"):
                        return
            except VideoEngineError as error:
                LOGGER.warning("Could not release the video model: %s", error)

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()

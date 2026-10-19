"""Очередь работ видеокарты: по одной, по порядку, с прогрессом и отменой.

Генерация длится от десяти секунд до нескольких минут — держать на ней
HTTP-соединение можно (так работает ``wait=true``), но клиенту, который
хочет видеть ход работы или поставить несколько работ подряд, нужна
задача с номером. Обе дороги идут через эту очередь: один рабочий поток —
та же «одна работа с видеокартой за раз», что у общей группы очереди
Gradio в интерфейсе (``ui.state.GPU_CONCURRENCY_ID``); настоящий
инвариант по-прежнему держит замок генератора.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
FINISHED = (DONE, FAILED, CANCELLED)

# Сколько законченных задач помнить: результат забирают сразу, а память
# процесса не должна расти с каждой генерацией.
KEEP_FINISHED = 200


class JobFailed(RuntimeError):
    """Работа не состоялась по причине, которую стоит показать клиенту как есть."""


@dataclass
class Progress:
    stage: str = ""
    image: int = 0
    images: int = 0
    step: int = 0
    steps: int = 0
    # Доля всей работы и оценка остатка — их сообщает движок видео; у
    # картинок прогресс считается шагами, и здесь пусто.
    fraction: float | None = None
    remaining_seconds: float | None = None


@dataclass
class Job:
    id: str
    kind: str
    work: Callable[[Job], dict]
    order: int
    status: str = QUEUED
    progress: Progress = field(default_factory=Progress)
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    done: threading.Event = field(default_factory=threading.Event)
    cancel_requested: bool = False
    # Своё прерывание у работ не генератора картинок (видео — движок FreeVideo);
    # ``None`` — прерывание очереди по умолчанию.
    interrupt: Callable[[], None] | None = None

    def report(self, index: int, step: int, total: int) -> None:
        """Обратный вызов генератора: (номер картинки, шаг, всего шагов)."""
        self.progress.image, self.progress.step, self.progress.steps = index + 1, step, total
        self.progress.stage = "generating"

    def stage(self, text: str) -> None:
        self.progress.stage = text

    def describe(self, position: int | None = None) -> dict[str, Any]:
        described: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": vars(self.progress).copy(),
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
        }
        if position is not None:
            described["queue_position"] = position
        if self.status == DONE:
            described["result"] = self.result
        if self.error:
            described["error"] = self.error
        return described


class JobQueue:
    """Один рабочий поток; ``interrupt`` прерывает работу, которая идёт сейчас."""

    def __init__(self, interrupt: Callable[[], None]) -> None:
        self._interrupt = interrupt
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._pending: list[Job] = []
        self._condition = threading.Condition()
        self._counter = itertools.count()
        self._worker = threading.Thread(target=self._run, name="api-jobs", daemon=True)
        self._worker.start()

    def submit(self, kind: str, work: Callable[[Job], dict], interrupt: Callable[[], None] | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind, work=work, order=next(self._counter), interrupt=interrupt)
        with self._condition:
            self._jobs[job.id] = job
            self._pending.append(job)
            self._forget_old()
            self._condition.notify()
        LOGGER.info("API job %s (%s) queued", job.id, kind)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._condition:
            return self._jobs.get(job_id)

    def position(self, job: Job) -> int | None:
        """Место в очереди с нуля (ноль — следующая); у начатой — ``None``."""
        with self._condition:
            return self._pending.index(job) if job in self._pending else None

    def list(self) -> list[Job]:
        with self._condition:
            return list(self._jobs.values())

    def cancel(self, job: Job) -> bool:
        """Снимает задачу из очереди или прерывает идущую. ``False`` — уже закончена."""
        with self._condition:
            if job.status in FINISHED:
                return False
            job.cancel_requested = True
            if job in self._pending:
                self._pending.remove(job)
                self._finish(job, CANCELLED)
                return True
        # Идёт сейчас: генератор остановится после текущего шага, движок
        # видео — по своей команде отмены.
        (job.interrupt or self._interrupt)()
        return True

    def _finish(self, job: Job, status: str, result: dict | None = None, error: str | None = None) -> None:
        job.status, job.result, job.error, job.finished = status, result, error, time.time()
        job.done.set()

    def _forget_old(self) -> None:
        finished = [job_id for job_id, job in self._jobs.items() if job.status in FINISHED]
        for job_id in finished[: max(0, len(finished) - KEEP_FINISHED)]:
            del self._jobs[job_id]

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending:
                    self._condition.wait()
                job = self._pending.pop(0)
                job.status, job.started = RUNNING, time.time()
            try:
                result = job.work(job)
            except JobFailed as error:
                self._finish(job, CANCELLED if job.cancel_requested else FAILED, error=str(error))
            except Exception as error:  # noqa: BLE001 — рабочий поток не должен умереть
                LOGGER.exception("API job %s failed", job.id)
                self._finish(job, FAILED, error=f"{type(error).__name__}: {error}")
            else:
                self._finish(job, CANCELLED if job.cancel_requested else DONE, result=result)
            LOGGER.info("API job %s (%s) %s", job.id, job.kind, job.status)

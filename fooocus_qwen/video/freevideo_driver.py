"""Драйвер FreeVideo: исполняется В ОКРУЖЕНИИ FreeVideo, а не приложения.

Приложение (``engine.py``) запускает этот файл интерпретатором
``FreeVideo/envs/unified`` и говорит с ним JSON-строками: команды — в stdin,
события — в stdout с префиксом ``EVENT_PREFIX`` (прочий вывод библиотек
событием не считается и уходит в журнал как есть).

Почему не CLI ``freevideo generate`` напрямую. Узел ComfyUI самого FreeVideo
зовёт CLI через оркестратор ``comfy_bridge.generate`` — и вся забота о
памяти и надёжности живёт там: подготовка таблиц сэмплинга под число шагов,
проверка отчёта движка, прогресс по всему ролику с оценкой остатка,
удержание процесса модели между частями, понятный текст сбоя. Оркестратор
не импортирует torch и не требует ComfyUI — ему нужны только каталог вывода
и два обратных вызова. Повторять его значило бы получить вторую, хуже
проверенную копию; поэтому драйвер — тонкая обёртка над ним.

Процесс живёт между заданиями: держатель модели (``resident_process.OWNER``)
принадлежит ему, и части длинного ролика идут без повторной загрузки весов.

Импортов из ``fooocus_qwen`` здесь нет и быть не может: другой интерпретатор,
другой torch. Комментарии — по-русски, сообщения — по-английски, как во всём
проекте.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import traceback
from contextlib import contextmanager
from pathlib import Path

# Каталог этого файла в sys.path[0] — каталог пакета приложения; его модули
# (geometry.py и др.) не должны заслонять одноимённые где-либо ещё.
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    del sys.path[0]

EVENT_PREFIX = "@@fooocus-video@@ "

_stdout_lock = threading.Lock()
_cancel = threading.Event()


def emit(event: str, **values) -> None:
    line = EVENT_PREFIX + json.dumps({"event": event, **values}, ensure_ascii=False, default=str)
    with _stdout_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def _plain(value, depth: int = 0):
    """Сообщение прогресса — в то, что переживёт JSON: числа, строки, словари."""
    if depth > 4:
        return None
    if isinstance(value, dict):
        return {str(key): _plain(item, depth + 1) for key, item in value.items()
                if key != "frame_preview"}
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth + 1) for item in value[:32]]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def interrupted() -> None:
    """Оркестратор зовёт это между опросами движка; отмена — исключением.

    ``KeyboardInterrupt`` он распознаёт как отмену: останавливает CLI,
    отпускает держателя модели и пишет в отчёт «cancelled», а не «failed».
    """
    if _cancel.is_set():
        raise KeyboardInterrupt("cancelled by the user")


def _generate_part(bridge, job: dict, part: dict, index: int, total: int, previous: str | None):
    media = job.get("media") if index == 0 else job.get("continuation_media")

    def export_inputs(_run, _canvas):
        return {"media": media} if media else {}

    def progress(message):
        emit("progress", id=job["id"], part=index, parts=total, value=_plain(message))

    quality = job["quality"]
    return bridge.generate(
        job["prompts"][min(index, len(job["prompts"]) - 1)],
        job["width"],
        job["height"],
        part["frames"] / 24,
        int(job["seed"]) + index,
        job["output_dir"],
        progress=progress,
        interrupted=interrupted,
        export_inputs=export_inputs if media else None,
        two_pass=quality["two_pass"],
        base_steps=quality["base_steps"],
        refine_steps=quality["refine_steps"],
        step_cache=quality.get("step_cache"),
        upscale_factor=quality.get("upscale", 2),
        continue_from=previous,
        frames=part["frames"],
        first_frame_reference=bool(previous) and job.get("first_frame_reference", True),
        hide_tail=bool(job.get("hide_tail")),
        # Черновик — только первый проход в половинном размере; доводка
        # (``refine_from``) продолжает именно его: тот же промт, сид и
        # входы, только апскейл латентов и шаги уточнения.
        preview=bool(job.get("preview")),
        refine_from=job.get("refine_from"),
        # Финальный апскейл латентов — наша правка движка (freevideo_patches).
        **({"final_upscale": tuple(job["final_upscale"])} if job.get("final_upscale") else {}),
    )


@contextmanager
def budget(vram_gib=None, ram_gib=None):
    """Номинальный бюджет памяти задания: ``--vram-gib``/``--ram-gib`` движка.

    Оркестратор передаёт CLI только резерв видеопамяти из
    ``resource_settings.read`` — каждое поле словаря становится ключом
    ``--поле``. Бюджет добавляется в этот же словарь на время задания: так
    движок планирует размещение, как на карте с ``vram_gib`` и машине с
    ``ram_gib`` (ниже реальных — не выше: движок сам зажимает их найденным).
    Нужен для честной проверки заявленных 8 ГБ / 16 ГБ на большой машине и
    для того, кто хочет оставить память другим программам.
    """
    if vram_gib is None and ram_gib is None:
        yield
        return
    from freevideo_engine import resource_settings

    original = resource_settings.read

    def read(root, *, backend="cuda"):
        values = dict(original(root, backend=backend))
        if vram_gib is not None:
            values["vram_gib"] = float(vram_gib)
        if ram_gib is not None:
            values["ram_gib"] = float(ram_gib)
        return values

    resource_settings.read = read
    try:
        yield
    finally:
        resource_settings.read = original


def run_job(job: dict) -> None:
    from freevideo_engine import chain_plan, comfy_bridge

    seconds = float(job["seconds"])
    continue_from = job.get("continue_from")
    plan = chain_plan.plan(seconds, continued=bool(continue_from))
    parts = plan["blocks"]
    if job.get("preview") or job.get("refine_from"):
        # Черновик и его доводка — одна генерация: продолжение их не берёт.
        parts = parts[:1]
        if len(plan["blocks"]) > 1:
            raise ValueError("A draft is limited to one part (15 s); shorten the video")
    emit("plan", id=job["id"], plan=_plain(plan))
    outputs: list[str] = []
    previous = continue_from
    with budget(job.get("vram_gib"), job.get("ram_gib")):
        for index, part in enumerate(parts):
            output = _generate_part(comfy_bridge, job, part, index, len(parts), previous)
            outputs.append(str(output))
            previous = str(output)
            emit("part", id=job["id"], part=index, parts=len(parts), output=str(output))

    # Продление отдаёт ролик целиком: исходник и новые части склеиваются
    # так же, как части длинного ролика между собой, — продолжение для
    # движка и есть следующая часть той же цепочки. У продлённого или
    # длинного ролика движок продолжает последнюю часть, а склеивается
    # ролик целиком (``continue_whole``).
    whole = job.get("continue_whole") or continue_from
    sources = ([whole] if whole else []) + outputs
    frames = None
    if len(sources) == 1:
        final = outputs[0]
    else:
        from freevideo_engine.chain_join import join_files

        final = str(Path(outputs[-1]).with_name("joined.mp4"))
        emit("progress", id=job["id"], part=len(parts), parts=len(parts),
             value={"label": "Joining parts", "phase": "join"})
        # Обрезка до плана — только для своих частей: длина исходника в план
        # продолжения не входит.
        joined = join_files(sources, final, cancelled=_cancel.is_set,
                            frames=None if continue_from or plan.get("single") else plan["frames"],
                            leads=_seam_leads(whole, continue_from))
        if isinstance(joined, dict):
            frames = joined.get("frames")
            # stderr драйвера приложение пишет в журнал; stdout — только события.
            print(f"Joined {len(sources)} videos: {frames} frames, "
                  f"seam sounds repaired {joined.get('seams_repaired')} of {len(sources) - 1}", file=sys.stderr, flush=True)
    # ``last`` — последняя сгенерированная часть: запись латентов для
    # следующего продления лежит у неё, а не у склейки.
    emit("done", id=job["id"], output=final, parts=outputs, last=outputs[-1], frames=frames)


def _seam_leads(whole, last):
    """Звук стыка с целым роликом — по записи его последней части.

    Движок чинит звук на стыке по записи новой части, сверяя, что она
    продолжает именно предыдущий файл. Предыдущий здесь — склейка, а запись
    ссылается на её последнюю часть: без подмены сверка не сошлась бы и стык
    остался бы без починки. Для остальных стыков — None: движок читает свою
    запись сам.
    """
    if not whole or not last or whole == last:
        return None
    from freevideo_engine.comfy_continuation import audio_lead_in

    whole_path = Path(whole).resolve()

    def leads(following, previous):
        if previous is not None and Path(previous).resolve() == whole_path:
            return audio_lead_in(following, previous=last)
        return None

    return leads


def release(reason: str = "requested") -> None:
    """Отпускает процесс модели, держащий видеопамять между заданиями."""
    try:
        from freevideo_engine.resident_process import OWNER
    except ImportError:
        return
    try:
        OWNER.release(reason)
    except Exception:  # noqa: BLE001 — отпускание не должно ронять драйвер
        traceback.print_exc()


def _pipe_lines(stream, poll: float = 0.05):
    """Строки канала Windows без висящего чтения: опрос ``PeekNamedPipe``.

    Синхронный ``ReadFile`` по каналу stdin, ждущий в соседнем потоке,
    блокирует на Windows загрузку любой DLL со своей CRT: при старте она
    опрашивает стандартные дескрипторы и ждёт, пока чтение не завершится.
    Так зависало продление ролика — главный поток вставал на ``import numpy``
    в ``continuation.prepare`` и ждал следующей команды, которой не было.
    Здесь чтение начинается, только когда байты уже лежат в канале, и
    возвращается сразу. Конец канала (приложение ушло) — сбой ``PeekNamedPipe``.
    """
    import ctypes
    import msvcrt
    import os
    import time
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    peek = kernel32.PeekNamedPipe
    peek.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p,
                     ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    peek.restype = wintypes.BOOL
    descriptor = stream.fileno()
    handle = msvcrt.get_osfhandle(descriptor)
    pending = b""
    while True:
        available = wintypes.DWORD()
        if not peek(handle, None, 0, None, ctypes.byref(available), None):
            break
        if not available.value:
            time.sleep(poll)
            continue
        pending += os.read(descriptor, available.value)
        *lines, pending = pending.split(b"\n")
        for line in lines:
            yield line.decode("utf-8", errors="replace")
    if pending:
        yield pending.decode("utf-8", errors="replace")


def command_lines(stream=None):
    """Строки команд из stdin; на Windows — без блокировки загрузки DLL."""
    stream = sys.stdin if stream is None else stream
    if sys.platform == "win32":
        try:
            import msvcrt

            # Канал ли это: консоль и файл висящего чтения не держат.
            if _is_pipe(msvcrt.get_osfhandle(stream.fileno())):
                yield from _pipe_lines(stream)
                return
        except (AttributeError, OSError, ValueError):
            pass
    yield from stream


def _is_pipe(handle) -> bool:
    import ctypes

    file_type_pipe = 3
    return ctypes.WinDLL("kernel32").GetFileType(handle) == file_type_pipe


def main() -> int:
    jobs: list[dict] = []
    ready = threading.Event()
    wake = threading.Condition()

    def read_commands() -> None:
        for line in command_lines():
            line = line.strip()
            if not line:
                continue
            try:
                command = json.loads(line)
            except ValueError:
                emit("error", id=None, message="Malformed command line")
                continue
            kind = command.get("cmd")
            if kind == "cancel":
                _cancel.set()
            else:
                # Флаг сбрасывается здесь, при постановке задания, а не при его
                # старте: «Стоп», пришедший между ними, иначе потерялся бы.
                if kind == "generate":
                    _cancel.clear()
                with wake:
                    jobs.append(command)
                    wake.notify()
        # stdin закрыт — приложение ушло: прерываем работу и выходим.
        _cancel.set()
        with wake:
            jobs.append({"cmd": "quit"})
            wake.notify()

    threading.Thread(target=read_commands, name="commands", daemon=True).start()
    ready.set()
    emit("ready", python=sys.version.split()[0])
    while True:
        with wake:
            while not jobs:
                wake.wait()
            command = jobs.pop(0)
        kind = command.get("cmd")
        if kind == "quit":
            release("application closed")
            return 0
        if kind == "release":
            release(command.get("reason", "requested"))
            emit("released", id=command.get("id"))
            continue
        if kind != "generate":
            emit("error", id=command.get("id"), message=f"Unknown command: {kind}")
            continue
        try:
            run_job(command)
        except KeyboardInterrupt:
            emit("cancelled", id=command.get("id"))
        except BaseException as error:  # noqa: BLE001 — любой сбой задания — событие, драйвер живёт
            traceback.print_exc()
            emit("error", id=command.get("id"), message=str(error) or type(error).__name__,
                 kind=type(error).__name__)


if __name__ == "__main__":
    sys.exit(main())

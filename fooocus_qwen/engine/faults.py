"""Сбой CUDA, после которого видеокарта в этом процессе больше не работает.

Найдено на 3090 под Windows (2026-09-30 и 2026-10-20): при переключении на
Turbo закрепление памяти падало с ``cudaErrorAlreadyMapped``, а первое же ядро
после этого — с ``cudaErrorSharedObjectInitFailed``. Тридцать шесть минут
спустя та же генерация падала тем же образом: контекст CUDA процесса не
восстанавливается, его не переинициализировать без перезапуска. Повторить
сбой нарочно не удалось — ни нехваткой RAM, ни занятой чужими программами
видеопамятью, ни переносом загрузки в другой поток; системный журнал
сброса драйвера (TDR) не показал. Поэтому здесь не лечение причины, а честное
поведение: такой сбой узнаётся и называется, а не маскируется.

Модуль не импортирует ``torch``: его читает и ``ui/``, которому torch при
сборке интерфейса тянуть запрещено (``test_studio_lazy.py``).
"""

from __future__ import annotations

# ``cudaErrorMemoryAllocation``: нехватка памяти — обычный, исправимый отказ.
CUDA_OUT_OF_MEMORY = 2


def is_gpu_fault(error: BaseException) -> bool:
    """Ошибка драйвера CUDA, кроме нехватки памяти.

    ``torch.AcceleratorError`` несёт код ошибки CUDA в ``error_code``; у
    остальных исключений его нет. Нехватка видеопамяти приходит отдельным
    ``torch.OutOfMemoryError`` без кода, нехватка закреплённой памяти — кодом
    ``CUDA_OUT_OF_MEMORY``; и то и другое контекст не ломает.
    """
    code = getattr(error, "error_code", None)
    return isinstance(code, int) and code != CUDA_OUT_OF_MEMORY


def snapshot() -> str:
    """Состояние машины в момент сбоя — для журнала, построчно.

    Двенадцать прогонов сценария сбоя (2026-10-20) прошли, а у человека он
    повторялся: условие живёт в его сеансе, и без снимка с момента сбоя его
    не найти. Каждый пункт собирается отдельно и сам по себе не валит снимок:
    на сломанном контексте падает что угодно.
    """
    lines: list[str] = []

    def add(title: str, collect) -> None:
        try:
            lines.append(f"{title}: {collect()}")
        except Exception as error:  # noqa: BLE001 — снимок не должен падать
            lines.append(f"{title}: unavailable ({type(error).__name__}: {error})")

    def host_memory() -> str:
        import torch

        stats = torch.cuda.host_memory_stats()
        return (f"{stats.get('allocated_bytes.current', 0) / 2**30:.1f} GiB pinned in "
                f"{stats.get('allocations.current', 0)} allocations")

    def ram() -> str:
        import psutil

        memory = psutil.virtual_memory()
        rss = psutil.Process().memory_info().rss
        return (f"{memory.available / 2**30:.1f} of {memory.total / 2**30:.1f} GiB available, "
                f"this process {rss / 2**30:.1f} GiB")

    def gpu_counters() -> str:
        # Закреплённая память на Windows — это «общая память GPU» (Shared
        # Usage); счётчики показывают её и видеопамять по процессам.
        import subprocess
        import sys

        if sys.platform != "win32":
            return "not on Windows"
        script = (
            "(Get-Counter '\\GPU Process Memory(*)\\Shared Usage','\\GPU Process Memory(*)\\Dedicated Usage')"
            ".CounterSamples | Where-Object { $_.CookedValue -gt 100MB } | ForEach-Object {"
            " $id = [int]($_.InstanceName -replace '^pid_(\\d+)_.*','$1');"
            " $name = (Get-Process -Id $id -ErrorAction SilentlyContinue).Name;"
            " '{0} {1} {2} {3:N2} GB' -f $name, $id, ($_.Path -replace '.*\\\\',''), ($_.CookedValue / 1GB) }"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=20,
        )
        return "; ".join(line.strip() for line in result.stdout.splitlines() if line.strip()) or "empty"

    def nvidia_smi() -> str:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        return result.stdout.strip() or result.stderr.strip()

    add("pinned host memory", host_memory)
    add("RAM", ram)
    add("GPU memory by process", gpu_counters)
    add("nvidia-smi", nvidia_smi)
    return "\n".join(lines)

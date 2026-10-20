"""Сбой CUDA, ломающий контекст процесса: распознавание и снимок для журнала."""

import subprocess

from fooocus_qwen.engine import faults


class _CudaError(RuntimeError):
    def __init__(self, code: int) -> None:
        super().__init__(f"CUDA error {code}")
        self.error_code = code


def test_only_cuda_errors_other_than_out_of_memory_are_faults():
    assert faults.is_gpu_fault(_CudaError(208))  # cudaErrorAlreadyMapped
    assert faults.is_gpu_fault(_CudaError(302))  # cudaErrorSharedObjectInitFailed
    assert not faults.is_gpu_fault(_CudaError(faults.CUDA_OUT_OF_MEMORY))
    assert not faults.is_gpu_fault(RuntimeError("shape mismatch"))
    assert not faults.is_gpu_fault(ValueError("bad value"))


def test_snapshot_survives_every_probe_failing(monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("no such program")

    monkeypatch.setattr(subprocess, "run", broken)
    text = faults.snapshot()
    assert "GPU memory by process: " in text and "nvidia-smi: unavailable (OSError" in text
    assert text.count("\n") == 3  # все четыре пункта на месте

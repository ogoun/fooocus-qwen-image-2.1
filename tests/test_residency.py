"""Перемещение весов между хостом и видеопамятью.

Тесты идут на процессоре, если CUDA недоступна: проверяется логика владения
весами, а она от устройства не зависит.
"""

import pytest
import torch

from fooocus_qwen.engine.residency import StagedModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN = torch.cuda.is_available()


def tiny_module():
    module = torch.nn.Linear(8, 4)
    module.register_buffer("scale", torch.ones(4))
    return module


def test_module_starts_on_the_host():
    staged = StagedModule(tiny_module(), DEVICE, pin_memory=PIN)
    assert staged.resident is False
    assert staged.module.weight.device.type == "cpu"


def test_to_device_moves_parameters_and_buffers():
    staged = StagedModule(tiny_module(), DEVICE, pin_memory=PIN)
    staged.to_device()

    assert staged.resident is True
    assert staged.module.weight.device.type == DEVICE
    assert staged.module.scale.device.type == DEVICE


def test_values_survive_a_round_trip():
    module = tiny_module()
    expected = module.weight.detach().clone()

    staged = StagedModule(module, DEVICE, pin_memory=PIN)
    staged.to_device()
    staged.to_host()

    assert torch.equal(staged.module.weight.detach(), expected)
    assert staged.module.weight.device.type == "cpu"


def test_host_copy_is_canonical_and_reused():
    # Возврат на хост не копирует веса обратно, а возвращает ссылку на исходную
    # копию: иначе закрепление памяти терялось бы после первой же перестановки.
    # Сравниваем через data_ptr(), а не через `is`: аксессор `.data` в PyTorch
    # заворачивает тензор в новый Python-объект при каждом обращении, даже без
    # единой операции между двумя чтениями (проверено: `t.data is t.data`
    # даёт False на торче 2.11.0+cu128) — идентичность обёртки не говорит о
    # том, используется ли то же самое хранилище, а data_ptr() говорит именно
    # это и потому ближе к сути проверяемой гарантии.
    staged = StagedModule(tiny_module(), DEVICE, pin_memory=PIN)
    host_ptr = staged.module.weight.data.data_ptr()
    staged.to_device()
    staged.to_host()
    assert staged.module.weight.data.data_ptr() == host_ptr


def test_repeated_calls_are_idempotent():
    staged = StagedModule(tiny_module(), DEVICE, pin_memory=PIN)
    staged.to_device()
    staged.to_device()
    staged.to_host()
    staged.to_host()
    assert staged.resident is False


def test_nbytes_counts_parameters_and_buffers():
    staged = StagedModule(tiny_module(), DEVICE, pin_memory=PIN)
    # 8*4 весов + 4 смещения + 4 буфера = 40 значений по 4 байта.
    assert staged.nbytes == 40 * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна CUDA")
def test_host_copy_is_pinned_when_asked():
    staged = StagedModule(tiny_module(), "cuda", pin_memory=True)
    assert staged.module.weight.data.is_pinned()

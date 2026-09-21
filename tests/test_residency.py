"""Перемещение весов между хостом и видеопамятью.

Тесты идут на процессоре, если CUDA недоступна: проверяется логика владения
весами, а она от устройства не зависит.
"""

import types

import pytest
import torch

from fooocus_qwen.engine.residency import ResidencyManager, StagedModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN = torch.cuda.is_available()


def tiny_module():
    module = torch.nn.Linear(8, 4)
    module.register_buffer("scale", torch.ones(4))
    return module


def tiny_pipe():
    """Заглушка пайплайна: три крошечных модуля вместо настоящих весов.

    ResidencyManager обращается только к атрибутам transformer/text_encoder/vae,
    поэтому SimpleNamespace достаточно — настоящий пайплайн появится в задаче 9.
    """
    return types.SimpleNamespace(
        transformer=tiny_module(),
        text_encoder=tiny_module(),
        vae=tiny_module(),
    )


def _device_of(module: torch.nn.Module) -> str:
    return next(module.parameters()).device.type


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


def test_to_host_never_copies_the_canonical_tensor():
    # data_ptr() совпадает и в случае, если to_host() скопировал бы данные
    # обратно в то же самое место — совпадение адреса само по себе не
    # доказывает отсутствие копирования. Доказывает только видимость внешней
    # мутации: если модуль хранит ссылку на тот самый хост-тензор, а не
    # независимую копию, изменение, сделанное снаружи, будет видно через модуль.
    staged = StagedModule(tiny_module(), DEVICE, pin_memory=PIN)
    host_tensor = staged.module.weight.data

    staged.to_device()
    staged.to_host()

    host_tensor[0, 0] = 12345.0
    assert staged.module.weight.data[0, 0].item() == 12345.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="нужна CUDA")
def test_pinning_survives_several_cycles():
    # Закрепление делается один раз в __init__; несколько перестановок не
    # должны ни терять его, ни пытаться закрепить память повторно.
    staged = StagedModule(tiny_module(), "cuda", pin_memory=True)
    for _ in range(3):
        staged.to_device()
        staged.to_host()
    assert staged.module.weight.data.is_pinned()


def test_module_without_buffers_round_trips():
    # register_buffer нигде не вызывался: _named_tensors должен спокойно
    # отработать на пустом named_buffers(), не породив ни ошибок, ни пропусков.
    staged = StagedModule(torch.nn.Linear(8, 4), DEVICE, pin_memory=PIN)
    staged.to_device()
    assert staged.module.weight.device.type == DEVICE
    staged.to_host()
    assert staged.resident is False
    assert staged.module.weight.device.type == "cpu"


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


def test_manager_start_places_transformer_and_vae_on_device_and_encoder_on_host():
    pipe = tiny_pipe()
    manager = ResidencyManager(pipe, device=DEVICE, pin_memory=PIN)
    manager.start()

    assert _device_of(pipe.transformer) == DEVICE
    assert _device_of(pipe.vae) == DEVICE
    assert _device_of(pipe.text_encoder) == "cpu"


def test_manager_never_stages_the_vae():
    # У ResidencyManager нет публичного признака "VAE не переставляется" —
    # наблюдаем это косвенно: ни один из атрибутов менеджера не является
    # StagedModule, оборачивающим именно VAE-модуль. StagedModule — это
    # единственный механизм в модуле, который порождает хост-копию весов;
    # если такого экземпляра для vae нет, хост-копия для него не создавалась.
    pipe = tiny_pipe()
    manager = ResidencyManager(pipe, device=DEVICE, pin_memory=PIN)
    manager.start()

    staged_instances = [value for value in vars(manager).values() if isinstance(value, StagedModule)]
    assert staged_instances, "ожидались StagedModule-обёртки для трансформера и энкодера"
    assert all(staged.module is not pipe.vae for staged in staged_instances)


def test_text_encoder_resident_swaps_and_restores():
    pipe = tiny_pipe()
    manager = ResidencyManager(pipe, device=DEVICE, pin_memory=PIN)
    manager.start()

    assert _device_of(pipe.transformer) == DEVICE
    assert _device_of(pipe.text_encoder) == "cpu"

    with manager.text_encoder_resident():
        assert _device_of(pipe.text_encoder) == DEVICE
        assert _device_of(pipe.transformer) == "cpu"

    assert _device_of(pipe.transformer) == DEVICE
    assert _device_of(pipe.text_encoder) == "cpu"


def test_swap_counter_increments_once_per_context_entry():
    pipe = tiny_pipe()
    manager = ResidencyManager(pipe, device=DEVICE, pin_memory=PIN)
    manager.start()

    assert manager.stats()["swaps"] == 0.0
    with manager.text_encoder_resident():
        pass
    assert manager.stats()["swaps"] == 1.0
    with manager.text_encoder_resident():
        pass
    assert manager.stats()["swaps"] == 2.0


def test_stats_is_callable_before_start():
    manager = ResidencyManager(tiny_pipe(), device=DEVICE, pin_memory=PIN)
    stats = manager.stats()
    assert stats["swaps"] == 0.0


def test_text_encoder_resident_restores_placement_even_if_body_raises():
    # Утечка размещения при исключении означала бы, что трансформер остаётся
    # снятым с устройства до конца сессии — генерация после сбоя просто не
    # заработает. finally в контекстном менеджере обязан вернуть всё на место.
    pipe = tiny_pipe()
    manager = ResidencyManager(pipe, device=DEVICE, pin_memory=PIN)
    manager.start()

    class DeliberateFailure(Exception):
        pass

    with pytest.raises(DeliberateFailure):
        with manager.text_encoder_resident():
            raise DeliberateFailure()

    assert _device_of(pipe.transformer) == DEVICE
    assert _device_of(pipe.text_encoder) == "cpu"


def test_text_encoder_resident_before_start_raises_clearly():
    manager = ResidencyManager(tiny_pipe(), device=DEVICE, pin_memory=PIN)
    with pytest.raises(RuntimeError):
        with manager.text_encoder_resident():
            pass

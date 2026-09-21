"""Размещение весов между хостом и видеопамятью.

Задача: 33 ГБ весов против 24 ГБ видеопамяти. Штатный
``enable_model_cpu_offload`` гоняет по шине всё и на каждую генерацию; при
быстром пресете это треть времени.

Принятая политика: трансформер и VAE резидентны, текстовый энкодер живёт на
хосте и поднимается только при промахе кэша эмбеддингов. Тогда перебор сида,
шагов и разрешения не создаёт трафика по шине вовсе.

Канонической копией весов считается копия на хосте, а не в модуле. Благодаря
этому закрепление памяти делается один раз: возврат «на хост» — это возврат
ссылки на уже закреплённый тензор, а не новое копирование.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import torch

LOGGER = logging.getLogger(__name__)


def _named_tensors(module: torch.nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """Параметры и буферы одним потоком.

    Буферы нельзя пропускать: у трансформера в них лежат таблицы поворотных
    вложений, и модуль без них на видеокарте не считается.
    """
    for name, parameter in module.named_parameters(recurse=True):
        yield f"p:{name}", parameter
    for name, buffer in module.named_buffers(recurse=True):
        yield f"b:{name}", buffer


class StagedModule:
    """Модуль, чьи веса хранятся на хосте и по требованию поднимаются на устройство."""

    def __init__(self, module: torch.nn.Module, device: str | torch.device, pin_memory: bool = True) -> None:
        self.module = module
        self._device = torch.device(device)
        self._resident = False
        self._host: dict[str, torch.Tensor] = {}
        self._nbytes = 0

        pin_failed = False
        for name, tensor in _named_tensors(module):
            host = tensor.detach().to("cpu")
            if pin_memory and not pin_failed:
                try:
                    host = host.pin_memory()
                except RuntimeError as error:
                    # Закрепить десятки гигабайт удаётся не всегда; работать без
                    # закрепления медленнее, но полностью корректно.
                    LOGGER.warning("Не удалось закрепить память, продолжаю без неё: %s", error)
                    pin_failed = True
            self._host[name] = host
            self._nbytes += host.numel() * host.element_size()
            tensor.data = host

    @property
    def resident(self) -> bool:
        return self._resident

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def to_device(self) -> None:
        if self._resident:
            return
        for name, tensor in _named_tensors(self.module):
            tensor.data = self._host[name].to(self._device, non_blocking=True)
        if self._device.type == "cuda":
            # Копирование из закреплённой памяти асинхронное: без синхронизации
            # первый же вызов модуля прочитал бы наполовину заполненные веса.
            torch.cuda.synchronize(self._device)
        self._resident = True

    def to_host(self) -> None:
        if not self._resident:
            return
        for name, tensor in _named_tensors(self.module):
            tensor.data = self._host[name]
        self._resident = False
        if self._device.type == "cuda":
            torch.cuda.empty_cache()


class ResidencyManager:
    """Владеет размещением трёх моделей пайплайна."""

    def __init__(self, pipe, device: str | torch.device = "cuda", pin_memory: bool = True) -> None:
        self._pipe = pipe
        self._device = torch.device(device)
        self._pin_memory = pin_memory
        self._transformer: StagedModule | None = None
        self._text_encoder: StagedModule | None = None
        self._swaps = 0

    def start(self) -> None:
        """Раскладывает модели по местам. Вызывается один раз после загрузки."""
        LOGGER.info("Готовлю копии весов на хосте (закрепление: %s)", "да" if self._pin_memory else "нет")
        self._transformer = StagedModule(self._pipe.transformer, self._device, self._pin_memory)
        self._text_encoder = StagedModule(self._pipe.text_encoder, self._device, self._pin_memory)

        # VAE не переставляется никогда, поэтому копия на хосте ему не нужна:
        # это сэкономленные 1.35 ГБ закреплённой памяти.
        self._pipe.vae.to(self._device)
        self._transformer.to_device()

        LOGGER.info(
            "Резидентно: трансформер %.1f ГБ, VAE на устройстве; на хосте: энкодер %.1f ГБ",
            self._transformer.nbytes / 2**30,
            self._text_encoder.nbytes / 2**30,
        )

    @contextmanager
    def text_encoder_resident(self) -> Iterator[None]:
        """Поднимает энкодер, вытеснив трансформер, и возвращает всё обратно.

        Вместе они не помещаются: 17.5 плюс 14.2 гигабайта против 24 доступных.
        """
        if self._text_encoder is None or self._transformer is None:
            raise RuntimeError("ResidencyManager.start() не вызывался")

        self._swaps += 1
        self._transformer.to_host()
        self._text_encoder.to_device()
        try:
            yield
        finally:
            self._text_encoder.to_host()
            self._transformer.to_device()

    def stats(self) -> dict[str, float]:
        allocated = torch.cuda.memory_allocated(self._device) / 2**30 if self._device.type == "cuda" else 0.0
        reserved = torch.cuda.memory_reserved(self._device) / 2**30 if self._device.type == "cuda" else 0.0
        return {"allocated_gib": allocated, "reserved_gib": reserved, "swaps": float(self._swaps)}

"""Подкласс официального пайплайна с кэшем эмбеддингов.

Переопределён ровно один защищённый метод. Переписывать ``__call__`` было бы
заманчиво — там пришлось бы прокинуть ``image_pad_mask``, — но это означало бы
скопировать к себе двести строк цикла денойзинга и получить расхождение с
апстримом при первом же его обновлении.

``_get_qwen_prompt_embeds`` возвращает результат кодирования до размножения на
число изображений и до обнуления маски, то есть ровно то, что стоит кэшировать.
Перестановка моделей происходит вокруг настоящего вызова и при попадании в кэш
не выполняется вовсе.
"""

from __future__ import annotations

import inspect
import logging

import torch
from diffusers import QwenImage21Pipeline

from .embeds_cache import EmbedsCache
from .residency import ResidencyManager

LOGGER = logging.getLogger(__name__)

_EXPECTED_PARAMETERS = ["self", "prompt", "image", "device"]


def assert_contract() -> None:
    """Проверяет, что апстрим не изменил точку встраивания."""
    actual = list(inspect.signature(QwenImage21Pipeline._get_qwen_prompt_embeds).parameters)
    if actual != _EXPECTED_PARAMETERS:
        raise RuntimeError(
            "Изменилась сигнатура QwenImage21Pipeline._get_qwen_prompt_embeds: "
            f"ожидалось {_EXPECTED_PARAMETERS}, получено {actual}. "
            "Кэш эмбеддингов встроен в этот метод — обновите fooocus_qwen/engine/pipeline.py "
            "или зафиксируйте прежний коммит diffusers в requirements.txt."
        )


class QwenImage21StudioPipeline(QwenImage21Pipeline):
    """Пайплайн Qwen-Image-2.1 с кэшем эмбеддингов и ручным размещением весов."""

    def attach(self, residency: ResidencyManager, cache: EmbedsCache, device: torch.device) -> None:
        self._studio_residency = residency
        self._studio_cache = cache
        self._studio_device = device

    @property
    def _execution_device(self) -> torch.device:
        """Устройство вычислений задаётся явно.

        Штатная реализация выводит его из размещения модулей, а у нас модули
        сознательно живут на разных устройствах — она вернула бы процессор.
        """
        override = getattr(self, "_studio_device", None)
        if override is not None:
            return override
        return QwenImage21Pipeline._execution_device.fget(self)

    def _get_qwen_prompt_embeds(self, prompt=None, image=None, device=None):
        cache: EmbedsCache | None = getattr(self, "_studio_cache", None)
        residency: ResidencyManager | None = getattr(self, "_studio_residency", None)
        target = device or self._execution_device

        if cache is None or residency is None:
            return super()._get_qwen_prompt_embeds(prompt, image, target)

        key = cache.key(prompt, image)
        cached = cache.get(key)
        if cached is not None:
            LOGGER.debug("Эмбеддинги промта взяты из кэша, энкодер не поднимался")
            return tuple(tensor.to(target) for tensor in cached)

        with residency.text_encoder_resident():
            embeds = super()._get_qwen_prompt_embeds(prompt, image, target)

        cache.put(key, embeds)
        return embeds

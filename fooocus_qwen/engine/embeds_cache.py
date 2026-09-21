"""Кэш эмбеддингов промта.

Кодирование промта — единственная операция, ради которой приходится вытеснять
трансформер из видеопамяти. Пока промт и набор условных изображений не менялись,
результат кодирования не меняется тоже, и перестановку можно не делать.

Что именно становится бесплатным, зависит от того, есть ли условные
изображения:

- **Чистый текст в изображение.** Бесплатен перебор сида, шагов, разрешения и
  числа изображений: ни один из этих параметров в ключ не входит.
- **Любой запрос с условными изображениями** (референсы, источник, маска).
  Бесплатен перебор сида, шагов и числа изображений; смена пресета качества —
  нет, потому что вместе с числом шагов пресет меняет и ``output_resolution``.
  Пайплайн приводит каждое условное изображение к размеру, выведенному из
  ``output_resolution``, ещё ДО кодирования (``calculate_dimensions`` в
  ``__call__``), и в ``_get_qwen_prompt_embeds`` приходят уже масштабированные
  картинки. Отпечаток пикселей у них другой — значит, другой ключ, промах и
  полная перестановка. Это не дефект, а необходимость: эмбеддинги, снятые с
  изображения в 1024 px, для 1536 px просто неверны, и отдавать их было бы
  хуже, чем пересчитать. Но и обещать, что смена пресета ничего не стоит, в
  режиме редактирования нельзя — там условное изображение есть всегда.

Записи хранятся на хосте: с десятью референсами последовательность разрастается,
и держать несколько таких записей в видеопамяти значит отнимать её у генерации.
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)

Embeds = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def fingerprint(image: Image.Image) -> str:
    """Отпечаток изображения по его пикселям, размеру и режиму.

    Сравнение по пикселям, а не по идентичности объекта: пользователь может
    подать то же изображение повторно другим объектом, и пересчитывать ради
    этого семнадцать гигабайт весов незачем.
    """
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.size}".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheKey:
    """Ключ: тексты промтов и отпечатки условных изображений."""

    prompts: tuple[str, ...]
    images: tuple[str, ...]


class EmbedsCache:
    """Кэш с вытеснением давно не использованных записей."""

    def __init__(self, capacity: int = 4) -> None:
        self._capacity = max(1, capacity)
        self._entries: OrderedDict[CacheKey, Embeds] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def size(self) -> int:
        return len(self._entries)

    def key(self, prompt: str | Sequence[str] | None, image: Sequence[Image.Image] | None) -> CacheKey:
        prompts = (prompt,) if isinstance(prompt, str) else tuple(prompt or ())
        images = tuple(fingerprint(item) for item in (image or []))
        return CacheKey(prompts=tuple(prompts), images=images)

    def get(self, key: CacheKey) -> Embeds | None:
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return entry

    def put(self, key: CacheKey, value: Embeds) -> None:
        self._entries[key] = tuple(tensor.detach().to("cpu") for tensor in value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            evicted, _ = self._entries.popitem(last=False)
            LOGGER.debug("Из кэша эмбеддингов вытеснена запись %s", evicted.prompts[:1])

    def clear(self) -> None:
        self._entries.clear()
        self._hits = 0
        self._misses = 0

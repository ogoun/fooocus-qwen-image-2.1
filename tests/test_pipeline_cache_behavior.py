"""Проверка того, ради чего вообще существует кэш эмбеддингов.

test_embeds_cache.py проверяет кэш в изоляции, test_pipeline_contract.py —
только соответствие сигнатуре апстрима. Ни один из них не доказывает, что
переопределённый ``_get_qwen_prompt_embeds`` при попадании в кэш действительно
не вызывает родителя и не поднимает энкодер. Регрессия здесь не выглядела бы
ошибкой: генерация просто стала бы медленнее, а такое никто не замечает
месяцами. Модель в этом тесте не загружается — проверяется один метод на
неинициализированном экземпляре.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from diffusers import QwenImage21Pipeline

from fooocus_qwen.engine.embeds_cache import EmbedsCache
from fooocus_qwen.engine.pipeline import QwenImage21StudioPipeline


class _CountingResidency:
    """Заглушка ResidencyManager: считает входы в контекст, весов не касается."""

    def __init__(self) -> None:
        self.enters = 0

    @contextmanager
    def text_encoder_resident(self):
        self.enters += 1
        yield


def _sample_embeds():
    return (
        torch.zeros(1, 4, 8),
        torch.ones(1, 4, dtype=torch.long),
        torch.zeros(1, 4, dtype=torch.bool),
    )


def test_cache_hit_skips_parent_call_and_model_swap(monkeypatch):
    calls = 0

    def fake_parent(self, prompt=None, image=None, device=None):
        nonlocal calls
        calls += 1
        return _sample_embeds()

    monkeypatch.setattr(QwenImage21Pipeline, "_get_qwen_prompt_embeds", fake_parent)

    # object.__new__: проверяется один метод, а не вся инициализация пайплайна;
    # настоящая модель здесь не нужна и не загружается.
    pipe = object.__new__(QwenImage21StudioPipeline)
    cache = EmbedsCache()
    residency = _CountingResidency()
    pipe.attach(residency, cache, torch.device("cpu"))

    pipe._get_qwen_prompt_embeds("кот", None, torch.device("cpu"))
    pipe._get_qwen_prompt_embeds("кот", None, torch.device("cpu"))

    assert calls == 1
    assert residency.enters == 1
    assert cache.misses == 1
    assert cache.hits == 1

    # Другой промт обязан промахнуться заново — тест, проверяющий только
    # "второй вызов дешёвый", прошёл бы и для реализации, которая отдаёт
    # устаревшие эмбеддинги для любого промта, вообще ничего не кодируя.
    pipe._get_qwen_prompt_embeds("пёс", None, torch.device("cpu"))

    assert calls == 2
    assert residency.enters == 2

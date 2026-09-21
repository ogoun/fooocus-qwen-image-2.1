"""Страж совместимости с diffusers.

Кэш эмбеддингов встроен переопределением защищённого метода. Если апстрим
изменит его сигнатуру, оболочка должна сказать об этом внятно при запуске, а не
падать посреди генерации с непонятной ошибкой.
"""

import inspect

import pytest

torch = pytest.importorskip("torch")


def test_parent_method_signature_is_what_we_expect():
    from diffusers import QwenImage21Pipeline

    parameters = list(inspect.signature(QwenImage21Pipeline._get_qwen_prompt_embeds).parameters)
    assert parameters == ["self", "prompt", "image", "device"]


def test_parent_returns_three_values_per_its_own_annotation():
    from diffusers import QwenImage21Pipeline

    source = inspect.getsource(QwenImage21Pipeline._get_qwen_prompt_embeds)
    assert "return prompt_embeds, encoder_attention_mask, image_pad_mask" in source


def test_call_does_not_forward_image_pad_mask():
    # Ровно это ограничение и вынуждает кэшировать на уровне
    # _get_qwen_prompt_embeds, а не передавать готовые prompt_embeds в __call__.
    from diffusers import QwenImage21Pipeline

    source = inspect.getsource(QwenImage21Pipeline.__call__)
    assert "self.encode_prompt(" in source
    assert "image_pad_mask=image_pad_mask" not in source


def test_contract_guard_passes_on_the_installed_version():
    from fooocus_qwen.engine.pipeline import assert_contract

    assert_contract()

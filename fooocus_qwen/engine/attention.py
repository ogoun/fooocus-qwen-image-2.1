"""Механизм внимания трансформера: штатный или SageAttention.

SageAttention считает произведение запросов на ключи в INT8 и на RTX 3090 в
2.3 раза быстрее штатного ядра (косинус с эталоном 0.99993); на полной
генерации это −15…25 % времени без изменения видеопамяти
(``docs/research/2026-09-24-uskorenie-turbo-sage-int8.md``). Пакет
необязательный: без него всё работает как раньше.

Одна тонкость. SageAttention не принимает ``attn_mask``, а трансформер
Qwen-Image-2.1 передаёт её на каждом шаге — маску «настоящих» ключей против
дополнения промта справа. При одном промте дополнения нет, маска целиком из
True и ничего не отсекает: такая маска опускается. Маска, которая что-то
отсекает (причинная маска первого прохода по промту), всегда уходит штатному
ядру — точность там важнее, а проход один на генерацию.
"""

from __future__ import annotations

import functools
import logging

LOGGER = logging.getLogger(__name__)

SAGE = "sage"
NATIVE = "native"


@functools.lru_cache(maxsize=1)
def sage_available() -> bool:
    """SageAttention установлен и diffusers готов им пользоваться."""
    try:
        from diffusers.models import attention_dispatch
    except ImportError:  # pragma: no cover — без diffusers не работает ничего
        return False
    return bool(getattr(attention_dispatch, "_CAN_USE_SAGE_ATTN", False))


def _backend_name(backend) -> str | None:
    return None if backend is None else str(getattr(backend, "value", backend))


def _install_mask_guard() -> None:
    """Оборачивает диспетчер внимания трансформера один раз за процесс."""
    from diffusers.models.transformers import transformer_qwenimage21 as module

    if getattr(module.dispatch_attention_fn, "_qs_mask_guard", False):
        return
    original = module.dispatch_attention_fn

    def dispatch(query, key, value, attn_mask=None, backend=None, **kwargs):
        if attn_mask is not None and _backend_name(backend) in (SAGE, None):
            # backend=None — «выбранный глобально», то есть тоже sage, если он включён.
            if bool(attn_mask.all()):
                attn_mask = None
            else:
                backend = NATIVE
        return original(query, key, value, attn_mask=attn_mask, backend=backend, **kwargs)

    dispatch._qs_mask_guard = True
    module.dispatch_attention_fn = dispatch


def apply(transformer, use_sage: bool) -> bool:
    """Выбирает механизм внимания. Возвращает, включён ли SageAttention.

    Просьба о SageAttention без установленного пакета не ошибка: работаем
    штатным ядром и говорим об этом в журнале.
    """
    enabled = use_sage and sage_available()
    if use_sage and not enabled:
        LOGGER.warning("SageAttention выбран, но не установлен — внимание остаётся штатным")
    if enabled:
        _install_mask_guard()
    transformer.set_attention_backend(SAGE if enabled else NATIVE)
    LOGGER.info("Внимание: %s", "SageAttention" if enabled else "штатное")
    return enabled

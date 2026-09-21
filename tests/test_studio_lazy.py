"""Главное обещание задачи 12: интерфейс собирается, не трогая модель.

Три предыдущих задачи в этом проекте сдавали корректный код, чьё главное
обещание не проверял ни один тест, и каждый раз это ловил только поздний
ревью. Здесь обещание — «сборка интерфейса не грузит веса» — проверяется
явно: если ленивую загрузку когда-нибудь «упростят» до жадной, этот тест
упадёт первым, а не после жалобы пользователя на минуту старта и занятые
14 ГиБ видеопамяти.
"""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.ui import app
from fooocus_qwen.ui.state import Studio


def _fail_if_called(*_args, **_kwargs):
    raise AssertionError("engine.loader.load не должен вызываться при сборке интерфейса")


def test_studio_construction_does_not_touch_the_model():
    cfg = config.AppConfig()
    studio = Studio(cfg)
    assert studio.model_loaded is False
    assert studio._generator is None


def test_building_the_blocks_app_does_not_load_the_model(monkeypatch):
    # Подменяем loader.load так, чтобы любой случайный вызов уронил тест, а не
    # молча утащил веса на видеокарту, пока никто не смотрит.
    from fooocus_qwen.engine import loader

    monkeypatch.setattr(loader, "load", _fail_if_called)

    cfg = config.AppConfig()
    demo = app.build(cfg)

    assert isinstance(demo, gr.Blocks)


def test_generator_property_is_absent_until_first_access():
    # Само наличие атрибута-заглушки для генератора не означает загрузки —
    # проверяем именно то, что видно тестам: генератор ещё не создан.
    cfg = config.AppConfig()
    studio = Studio(cfg)
    assert studio._generator is None
    assert studio._residency is None
    assert studio._cache is None

"""Двуязычные подписи."""

import pytest

gr = pytest.importorskip("gradio")

from fooocus_qwen.ui.i18n import Localizer, pick


def test_pick_returns_the_requested_language():
    assert pick("generate", "ru") != pick("generate", "en")


def test_unknown_key_returns_itself():
    # Пропущенная подпись не должна ронять интерфейс.
    assert pick("нет такого ключа", "ru") == "нет такого ключа"


def test_bind_registers_the_component():
    with gr.Blocks():
        localizer = Localizer()
        box = localizer.bind(gr.Textbox(label="Промт"), label=("Промт", "Prompt"))

    assert box in localizer.components
    assert len(localizer.components) == 1


def test_updates_match_the_registration_order():
    with gr.Blocks():
        localizer = Localizer()
        localizer.bind(gr.Textbox(), label=("Первый", "First"))
        localizer.bind(gr.Button(), value=("Второй", "Second"))

    updates = localizer.updates("en")
    assert len(updates) == 2
    assert updates[0]["label"] == "First"
    assert updates[1]["value"] == "Second"


def test_several_fields_on_one_component():
    with gr.Blocks():
        localizer = Localizer()
        localizer.bind(gr.Slider(), label=("Шаги", "Steps"), info=("Сколько шагов", "How many steps"))

    update = localizer.updates("ru")[0]
    assert update["label"] == "Шаги"
    assert update["info"] == "Сколько шагов"

"""Переключатель языка: одна кнопка в углу вместо строки с выпадающим списком.

Заголовок страницы и строка «RU / EN» с подписью отнимали у холста больше
сотни пикселей высоты на каждой вкладке ради того, что пользователь читает
один раз. Заголовка теперь нет вовсе — название приложения стоит в заголовке
окна браузера, — а язык переключается кнопкой, которая висит в правом
верхнем углу поверх полосы вкладок и собственной высоты не занимает.

Само значение языка при этом живёт в ``gr.State``: оно ходит последним
входом в каждый обработчик, который что-то сообщает пользователю, и
заменить его на кнопку было нельзя — кнопка значения не хранит.
"""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.ui import app, layout  # noqa: E402
from fooocus_qwen.ui.i18n import LANGUAGES  # noqa: E402


def _built(lang="ru"):
    return app.build(config.AppConfig(lang=lang))


def _switcher(demo):
    for block_fn in demo.fns.values():
        if block_fn.fn.__name__ == "switch_language":
            return block_fn.fn
    raise AssertionError("обработчик переключения языка не найден")


def test_the_page_has_no_heading_of_its_own():
    """Название приложения — в заголовке окна, а не строкой на странице."""
    demo = _built()
    headings = [
        c["props"].get("value", "")
        for c in demo.get_config_file()["components"]
        if c.get("type") == "markdown"
    ]
    assert not any("Qwen-Image-2.1" in str(v) for v in headings), (
        f"заголовок вернулся на страницу: {headings}"
    )
    assert "Qwen-Image-2.1" in demo.title, "название пропало и из заголовка окна"


def test_the_switch_is_a_button_showing_the_current_language():
    demo = _built("ru")
    buttons = [
        c["props"] for c in demo.get_config_file()["components"]
        if c.get("type") == "button" and layout.LANG in (c["props"].get("elem_classes") or [])
    ]
    assert len(buttons) == 1, f"кнопок языка не одна, а {len(buttons)}"
    assert buttons[0]["value"] == "RU", buttons[0]["value"]


def test_there_is_no_dropdown_for_the_language_any_more():
    demo = _built()
    labels = [
        c["props"].get("label")
        for c in demo.get_config_file()["components"]
        if c.get("type") == "dropdown"
    ]
    assert "RU / EN" not in labels, "выпадающий список языка остался и занимает строку"


def test_one_click_switches_the_language_and_the_caption():
    switch = _switcher(_built("ru"))
    following, update = switch("ru")
    assert following == "en"
    assert update["value"] == "EN"


def test_clicking_again_comes_back():
    # Языка два, и кнопка обязана возвращать к исходному, а не застревать.
    switch = _switcher(_built("ru"))
    assert switch(switch("ru")[0])[0] == "ru"


def test_the_cycle_covers_every_language():
    """Обход по кругу, а не жёсткая пара: языков может стать больше."""
    switch = _switcher(_built("ru"))
    seen, current = [], LANGUAGES[0]
    for _ in range(len(LANGUAGES)):
        seen.append(current)
        current = switch(current)[0]
    assert sorted(seen) == sorted(LANGUAGES), seen
    assert current == LANGUAGES[0], "круг не замкнулся"


def test_an_unknown_language_falls_back_instead_of_raising():
    # Значение могло прийти из испорченной конфигурации; падать обработчику
    # переключения языка не за что.
    switch = _switcher(_built())
    following, update = switch("шведский")
    assert following in LANGUAGES
    assert update["value"] == following.upper()

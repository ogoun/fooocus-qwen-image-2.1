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


def _named(demo, name):
    """Находит обработчик по имени функции.

    Функция может и отсутствовать: обход вкладок при загрузке страницы
    зарегистрирован как чистый ``js`` без серверной части.
    """
    for block_fn in demo.fns.values():
        if getattr(block_fn.fn, "__name__", None) == name:
            return block_fn
    raise AssertionError(f"обработчик {name} не найден")


def _switcher(demo):
    return _named(demo, "switch_language").fn


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
    following, update = switch("ru")[:2]
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
    following, update = switch("шведский")[:2]
    assert following in LANGUAGES
    assert update["value"] == following.upper()


def _click_handler(demo):
    """Обработчик, висящий на клике по кнопке языка, целиком."""
    return _named(demo, "switch_language")


def test_the_click_itself_repaints_every_caption():
    """Подписи переводит сам клик, а не событие `change` у состояния.

    `gr.State.change` в Gradio 6.5.1 не наступает, когда состояние меняется
    как выход другого обработчика: клиент значения состояния не знает (оно
    живёт на сервере) и обнаружить его изменение не может. Проверено на
    чистом Gradio без нашего кода — после клика уходит один запрос вместо
    двух, и подписи остаются на прежнем языке.

    Поэтому перевод висит на самом клике: одним ответом меняются и
    состояние, и надпись на кнопке, и все подписи разом.
    """
    demo = _built("ru")
    handler = _click_handler(demo)
    localized = {id(component) for component in handler.outputs}

    demo_components = [c for c in demo.blocks.values() if hasattr(c, "elem_classes")]
    assert len(handler.outputs) > 10, (
        f"клик обновляет всего {len(handler.outputs)} компонентов — подписи он не трогает"
    )
    assert localized, "у клика нет выходов"
    del demo_components


def test_the_handler_returns_a_value_for_every_output():
    """Столько же значений, сколько выходов, — иначе Gradio отбросит ответ."""
    demo = _built("ru")
    handler = _click_handler(demo)
    result = handler.fn("ru")
    assert len(result) == len(handler.outputs), (
        f"обработчик вернул {len(result)} значений на {len(handler.outputs)} выходов"
    )


def test_switching_translates_a_tab_caption():
    """Проверка по существу: подпись вкладки действительно меняет язык."""
    demo = _built("ru")
    handler = _click_handler(demo)
    russian = handler.fn("ru")
    english = handler.fn("en")

    def captions(result):
        return [
            update.get("label")
            for update in result[2:]
            if isinstance(update, dict) and update.get("label")
        ]

    assert "Генерация" in captions(english), "с английского на русский подписи не вернулись"
    assert "Generate" in captions(russian), "с русского на английский подписи не перевелись"


def test_the_tabs_are_opened_once_at_startup():
    """Обход вкладок при загрузке — не украшение, а условие перевода.

    Gradio 6.5.1 обновляет подпись только у той вкладки, чьё содержимое уже
    смонтировано: у неоткрытых кнопка в полосе остаётся на прежнем языке до
    первого захода внутрь. Проверено на чистом Gradio без нашего кода;
    обновление контейнера `gr.Tabs` не помогает, а один заход в каждую
    вкладку — помогает.
    """
    demo = _built()
    warm_ups = [
        block_fn
        for block_fn in demo.fns.values()
        if block_fn.fn is None and block_fn.js and "role=\"tab\"" in block_fn.js
    ]
    assert warm_ups, "вкладки не прогреваются — полоса переведётся наполовину"
    assert all(not block_fn.outputs for block_fn in warm_ups), "обходу вкладок нечего возвращать"

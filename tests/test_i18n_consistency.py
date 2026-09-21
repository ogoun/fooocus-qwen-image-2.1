"""Согласованность локализатора: стартовый текст обязан совпадать с переводом.

Дважды в этом проекте расхождение между начальным текстом компонента и его
переводимым кортежем оставалось незамеченным юнит-тестами и всплывало только
при живом переключении языка в браузере: сначала — choices радиокнопки
режима области во вкладке редактирования, затем — кнопки «Сохранить» во
вкладке настроек. Оба раза `Localizer.updates()` формально работал верно
(подставлял правильный текст ПРИ ПЕРЕКЛЮЧЕНИИ), но стартовое значение самого
компонента было прописано отдельно и разъезжалось с русской половиной того же
кортежа.

Этот тест строит всё приложение целиком и для каждого зарегистрированного
поля каждого компонента сверяет его ТЕКУЩЕЕ значение с той половиной перевода,
что соответствует языку запуска — заранее, без браузера и без переключения.
"""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.ui import app
from fooocus_qwen.ui.i18n import Localizer


def _build_and_capture_localizer(monkeypatch, lang: str) -> Localizer:
    """Строит приложение и перехватывает единственный использованный Localizer.

    ``app.build()`` создаёт локализатор внутри себя и не отдаёт его наружу —
    его тип не расширяет и без того устоявшийся контракт ``build(cfg) ->
    gr.Blocks`` (его проверяет test_studio_lazy.py), поэтому здесь он просто
    перехватывается через подмену ``Localizer.bind`` тем же приёмом, что и в
    test_tab_generate.py/test_tab_edit.py для обработчиков событий.
    """
    captured: dict[str, Localizer] = {}
    original_bind = Localizer.bind

    def spy_bind(self, component, **fields):
        captured["localizer"] = self
        return original_bind(self, component, **fields)

    monkeypatch.setattr(Localizer, "bind", spy_bind)

    cfg = config.AppConfig(lang=lang)
    app.build(cfg)
    return captured["localizer"]


def _mismatches(localizer: Localizer) -> list[str]:
    index = 0 if localizer.default == "ru" else 1
    problems = []
    for component, fields in localizer.entries:
        for name, translations in fields.items():
            expected = translations[index]
            actual = getattr(component, name, None)
            if actual != expected:
                problems.append(
                    f"{type(component).__name__}.{name}: на старте {actual!r}, "
                    f"а перевод для {localizer.default!r} обещает {expected!r}"
                )
    return problems


def test_startup_values_match_the_russian_half_of_every_translation(monkeypatch):
    localizer = _build_and_capture_localizer(monkeypatch, "ru")
    problems = _mismatches(localizer)
    assert not problems, "\n".join(problems)


def test_startup_values_match_the_english_half_of_every_translation(monkeypatch):
    # Тот же граф, собранный сразу на английском: ловит перекос в обратную
    # сторону, если он вдруг окажется зашит только для русского старта.
    localizer = _build_and_capture_localizer(monkeypatch, "en")
    problems = _mismatches(localizer)
    assert not problems, "\n".join(problems)


def test_at_least_the_known_regression_points_are_actually_checked(monkeypatch):
    """Страховка от тавтологичной проверки: убеждаемся, что тест видит именно

    те компоненты, на которых уже дважды случалась рассинхронизация — вкладки,
    кнопки «Сохранить» на вкладке настроек и радиокнопку режима области на
    вкладке редактирования, — а не пустой список из-за ошибки в перехвате.
    """
    localizer = _build_and_capture_localizer(monkeypatch, "ru")
    labels = [
        fields["label"][0]
        for _component, fields in localizer.entries
        if "label" in fields
    ]
    assert "Режим области" in labels
    values = [
        fields["value"][0]
        for _component, fields in localizer.entries
        if "value" in fields
    ]
    assert values.count("Сохранить") == 1
    # "Сохранить промт" законно встречается дважды — сохранение именованного
    # пресета на вкладке генерации и сохранение системного промта на вкладке
    # настроек; именно различие между ЭТИМИ label и текстом было найдено
    # вручную при проверке в браузере.
    assert values.count("Сохранить промт") == 2

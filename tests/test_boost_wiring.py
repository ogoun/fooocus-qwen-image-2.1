"""Что видно в поле «Переписанный промт», то и уходит в модель.

Правило простое, но до этого раунда оно держалось на внимательности
пользователя и нарушалось двумя способами, оба молча.

**Кнопка мимо галочки.** «Переписать сейчас» работала независимо от «AI
буст», а переписанный текст уходит в модель только при включённой галочке.
Нажать кнопку, увидеть текст в поле и получить картинку по старому промту
было проще простого, и ничто об этом не сообщало.

**Устаревшая переписка.** Переписанный текст составлен по конкретному
исходному промту. Поправив исходный, пользователь ожидает, что правка
учтётся, — а в модель по-прежнему уходит то, что лежит в поле.

Теперь кнопка включает галочку сама, а оба расхождения между показанным и
отправленным попадают в строку состояния.
"""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import presets  # noqa: E402
from fooocus_qwen.engine.generator import GeneratedImage  # noqa: E402
from fooocus_qwen.ui import tab_generate  # noqa: E402
from fooocus_qwen.ui.i18n import Localizer  # noqa: E402
from fooocus_qwen.ui.state import Studio  # noqa: E402


class _Generator:
    """Запоминает промт, который до него дошёл."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def generate(self, request, progress=None):
        from PIL import Image

        self.seen.append(request.prompt)
        return [GeneratedImage(image=Image.new("RGB", (8, 8)), seed=1, parameters={})]


def _handlers(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path)
    cfg = config.AppConfig()
    studio = Studio(cfg)
    engine = _Generator()
    studio._generator = engine

    with gr.Blocks() as demo:
        tab_generate.build(studio, Localizer(cfg.lang))
    found = {}
    for block_fn in demo.fns.values():
        found.setdefault(block_fn.fn.__name__, block_fn.fn)
    return found, engine, studio


def _run(handlers, prompt, boosted, source, use_boost):
    return handlers["run"](
        prompt, boosted, source, use_boost, [], presets.DEFAULT, "1:1", 1, [], "",
        1.0, -1, True, 0, "ru", progress=lambda *a, **k: None,
    )


# --- что уходит в модель ---


def test_the_rewritten_prompt_is_sent_without_copying_it_by_hand(monkeypatch, tmp_path):
    handlers, engine, _studio = _handlers(monkeypatch, tmp_path)
    _run(handlers, "рыжий кот", "a ginger cat on a windowsill", "рыжий кот", use_boost=True)
    assert engine.seen == ["a ginger cat on a windowsill"]


def test_with_the_box_unticked_the_original_prompt_is_sent(monkeypatch, tmp_path):
    handlers, engine, _studio = _handlers(monkeypatch, tmp_path)
    _run(handlers, "рыжий кот", "a ginger cat", "рыжий кот", use_boost=False)
    assert engine.seen == ["рыжий кот"]


# --- и почему об этом теперь говорят ---


def test_an_ignored_rewrite_is_reported(monkeypatch, tmp_path):
    handlers, _engine, _studio = _handlers(monkeypatch, tmp_path)
    _images, status = _run(handlers, "рыжий кот", "a ginger cat", "рыжий кот", use_boost=False)
    assert "выключен" in status, status


def test_an_empty_field_with_the_box_unticked_says_nothing_extra(monkeypatch, tmp_path):
    # Предупреждение на каждой обычной генерации быстро перестают читать.
    handlers, _engine, _studio = _handlers(monkeypatch, tmp_path)
    _images, status = _run(handlers, "рыжий кот", "", "", use_boost=False)
    assert "выключен" not in status, status


def test_a_stale_rewrite_is_reported(monkeypatch, tmp_path):
    handlers, engine, _studio = _handlers(monkeypatch, tmp_path)
    _images, status = _run(
        handlers, "рыжий кот на подоконнике", "a ginger cat", "рыжий кот", use_boost=True
    )
    assert "другому тексту" in status, status
    # Уходит всё равно то, что видно в поле: одно правило вместо двух.
    assert engine.seen == ["a ginger cat"]


def test_a_matching_rewrite_says_nothing_extra(monkeypatch, tmp_path):
    handlers, _engine, _studio = _handlers(monkeypatch, tmp_path)
    _images, status = _run(handlers, "рыжий кот", "a ginger cat", "рыжий кот", use_boost=True)
    assert "другому тексту" not in status, status


# --- кнопка «Переписать сейчас» ---


def test_the_rewrite_button_ticks_the_box_itself(monkeypatch, tmp_path):
    handlers, _engine, studio = _handlers(monkeypatch, tmp_path)
    monkeypatch.setattr(
        studio, "boost_prompt", lambda *a, **k: ("a ginger cat", "1:1", "готово")
    )

    text, _ratio, box, source, _message = handlers["rewrite"]("рыжий кот", [], "1:1", "ru")

    assert text == "a ginger cat"
    assert box["value"] is True, "галочка осталась выключенной — ловушка вернулась"
    assert source == "рыжий кот", "источник не запомнен — устаревание не отследить"


def test_a_failed_rewrite_does_not_tick_the_box_or_move_the_source(monkeypatch, tmp_path):
    """Отказ сервера не должен помечать прежний текст как свежий.

    Иначе после неудачной переписки галочка встала бы, поле осталось бы
    пустым или старым, а источник — новым: ровно та рассогласованность,
    ради устранения которой всё и делается.
    """
    handlers, _engine, studio = _handlers(monkeypatch, tmp_path)
    monkeypatch.setattr(studio, "boost_prompt", lambda *a, **k: ("", "", "сервер недоступен"))

    text, _ratio, box, source, message = handlers["rewrite"]("рыжий кот", [], "1:1", "ru")

    assert text == ""
    assert box["value"] is False
    assert isinstance(source, dict), "источник обязан остаться прежним, а не смениться"
    assert "недоступен" in message


# --- как результат доходит до экрана ---


def test_the_result_opens_large_not_as_a_grid(monkeypatch, tmp_path):
    """``preview=True`` действует только при первой загрузке: новое значение
    возвращало галерею к сетке, и единственная картинка становилась
    квадратной миниатюрой выше окна. Результат обязан открываться крупно."""
    handlers, _engine, _studio = _handlers(monkeypatch, tmp_path)
    shown, _status = _run(handlers, "рыжий кот", "", "", use_boost=False)
    assert isinstance(shown, gr.Gallery)
    assert shown.selected_index == 0
    assert len(shown.value) == 1

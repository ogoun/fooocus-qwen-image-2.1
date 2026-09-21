"""Сообщения, а не только подписи: строка состояния тоже говорит на языке интерфейса.

Требование 11 спецификации — двуязычный интерфейс. До этого раунда правок
двуязычными были только подписи компонентов (``T`` + ``Localizer``), а всё,
что пользователь читает по ходу работы, было русским литералом в коде: около
сорока строк по четырём вкладкам, ``ui/state.py`` и подписям миниатюр. При
переключении на английский вокруг английских подписей появлялось
«Референсов: 3 из 10» и «модель ещё не загружена» — вид не «язык не
переведён», а «что-то сломалось».

Здесь два уровня проверки. Первый — согласованность самой таблицы
``MESSAGES``: обе половины обязаны иметь один и тот же набор именованных
подстановок, иначе перевод упадёт в ``format()`` в момент показа, а не при
сборке. Второй — поведенческий: собранные обработчики, вызванные с ``"en"``,
не должны вернуть ни одной кириллической буквы в том тексте, который
интерфейс пишет сам. Процитированный текст исключения — исключение из
правила, и оно проведено осознанно: см. ``_assert_english_frame``.
"""

from __future__ import annotations

import re
import string

import pytest

from fooocus_qwen.ui.i18n import LANGUAGES, MESSAGES, say

CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def _placeholders(template: str) -> set[str]:
    """Имена подстановок шаблона, без спецификаций формата.

    ``{allocated:.1f}`` и ``{allocated}`` — одна и та же подстановка; сравнивать
    надо имена, иначе тест ругался бы на законную разницу в форматировании.
    """
    return {name for _text, name, _spec, _conv in string.Formatter().parse(template) if name}


@pytest.mark.parametrize("key", sorted(MESSAGES))
def test_both_halves_of_every_message_take_the_same_placeholders(key):
    russian, english = MESSAGES[key]
    assert _placeholders(russian) == _placeholders(english), (
        f"перевод «{key}» подставляет другой набор значений, чем оригинал: "
        f"{_placeholders(russian)} против {_placeholders(english)}"
    )


@pytest.mark.parametrize("key", sorted(MESSAGES))
def test_no_message_uses_positional_placeholders(key):
    # Позиционные {} привязывают перевод к порядку слов оригинала, а порядок
    # слов — первое, что меняется при переводе.
    for half in MESSAGES[key]:
        assert "{}" not in half, f"«{key}»: позиционная подстановка вместо именованной"


@pytest.mark.parametrize("key", sorted(MESSAGES))
def test_the_english_half_is_actually_english(key):
    assert not CYRILLIC.search(MESSAGES[key][1]), f"«{key}»: в английской половине кириллица"


@pytest.mark.parametrize("key", sorted(MESSAGES))
def test_the_two_halves_differ(key):
    # Страховка от копипасты: одинаковые половины означали бы, что перевод
    # забыли, а тест выше этого не заметит (кириллицы-то нет).
    russian, english = MESSAGES[key]
    assert russian != english, f"«{key}»: обе половины одинаковы — перевод не сделан"


def test_an_unknown_key_returns_itself_instead_of_raising():
    # То же правило, что и у pick(): пропущенное сообщение не роняет обработчик.
    assert say("нет такого сообщения", "ru") == "нет такого сообщения"


def test_say_picks_the_language_and_substitutes():
    assert say("references_counted", "ru", count=3, total=10) == "Референсов: 3 из 10"
    assert say("references_counted", "en", count=3, total=10) == "References: 3 of 10"


def test_every_message_formats_with_its_own_placeholders():
    # Опечатка в имени подстановки внутри шаблона проявилась бы KeyError в
    # момент показа сообщения — то есть ровно тогда, когда пользователю и так
    # плохо (половина этих сообщений про ошибки).
    for key, (russian, _english) in MESSAGES.items():
        values = {name: 1 for name in _placeholders(russian)}
        for lang in LANGUAGES:
            say(key, lang, **values)


# --- поведенческая проверка на собранных вкладках ---

gr = pytest.importorskip("gradio")

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import presets  # noqa: E402
from fooocus_qwen.ui import tab_edit, tab_gallery, tab_generate, tab_settings  # noqa: E402
from fooocus_qwen.ui.i18n import Localizer  # noqa: E402
from fooocus_qwen.ui.state import Studio  # noqa: E402


def _english_handlers(monkeypatch, tmp_path):
    """Собирает все четыре вкладки на английском и отдаёт их обработчики."""
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "ENDPOINT_FILE", tmp_path / "llm_endpoint.txt")
    monkeypatch.setattr(config, "SYSTEM_PROMPT_DIR", tmp_path)
    # Обработчик «открыть папку» вызывает файловый менеджер системы. Без подмены
    # каждый прогон набора распахивает пользователю окно проводника на временном
    # каталоге pytest — тест обязан проверять текст сообщения, а не дёргать
    # рабочий стол.
    monkeypatch.setattr(tab_gallery, "_open_folder", lambda path: None)
    cfg = config.AppConfig(lang="en")
    studio = Studio(cfg)

    with gr.Blocks() as demo:
        localizer = Localizer(cfg.lang)
        components = tab_generate.build(studio, localizer)
        tab_edit.build(studio, localizer)
        tab_gallery.build(studio, localizer, components)
        tab_settings.build(studio, localizer)

    handlers = {}
    for block_fn in demo.fns.values():
        handlers.setdefault(block_fn.fn.__name__, block_fn.fn)
    return handlers, studio


def _assert_english(value, what: str) -> None:
    """Сообщение целиком английское.

    Применимо к сообщениям, которые интерфейс пишет сам, от первого до
    последнего слова.
    """
    text = value if isinstance(value, str) else repr(value)
    assert not CYRILLIC.search(text), f"{what} остался русским при английском интерфейсе: {text!r}"


def _assert_english_frame(value: str, frame: str, what: str) -> None:
    """Английская рамка вокруг процитированного текста исключения.

    Граница проведена намеренно: интерфейс переводит то, что пишет сам, а
    техническую причину цитирует как есть — на языке того слоя, который её
    породил. Причины приходят из torch, diffusers, urllib и операционной
    системы и всегда английские; наши собственные исключения (``llm/``,
    ``prompting/``) написаны по-русски, и переводить их значило бы затащить
    таблицу переводов интерфейса в слои, которые о нём ничего не знают —
    ровно ту границу, ради которой они и выделены. См. докстринг
    ``i18n`` и отчёт по этому раунду.
    """
    assert value.startswith(frame), f"{what}: ожидалась рамка {frame!r}, получено {value!r}"


def test_every_status_line_a_user_can_reach_without_the_model_is_english(monkeypatch, tmp_path):
    handlers, _studio = _english_handlers(monkeypatch, tmp_path)

    # вкладка «Генерация»
    _assert_english(handlers["clear_references"]("en")[-1], "очистка референсов")
    _assert_english(handlers["stop"]("en"), "кнопка остановки")
    _assert_english(
        handlers["save"]("   ", "p", "", [], "LowQuality", "1:1", -1, 1.0, "en")[1],
        "сохранение без имени",
    )
    _assert_english(
        handlers["save"]("preset", "p", "", [], "LowQuality", "1:1", -1, 1.0, "en")[1],
        "сохранение пресета",
    )
    _assert_english(handlers["load"]("", "en")[-1], "загрузка без выбора")
    _assert_english(handlers["load"]("preset", "en")[-1], "загрузка пресета")
    _assert_english(handlers["delete"]("нет такого", "en")[1], "удаление несуществующего")

    # вкладка «Редактирование»
    _assert_english(handlers["expand_canvas"](None, ["right"], 0.5, "en")[2], "расширение без картинки")
    _assert_english(handlers["take_back"](None, "en")[1], "пустая отправка в редактор")

    # вкладка «Настройки»
    _assert_english(handlers["store_endpoint"]("x\n", "en"), "сохранение адреса")
    # Здесь цитируется текст исключения — проверяем рамку, см. _assert_english_frame.
    _assert_english_frame(handlers["check_connection"]("en"), "No connection:", "проверка связи")
    _assert_english(handlers["store_prompt_file"]("system_prompt_t2i.txt", "text", "en"), "сохранение промта")
    _assert_english(handlers["memory_report"]("en"), "отчёт о памяти")

    # вкладка «Галерея»: у gr.JSON переводимы и ключи словаря
    _assert_english(handlers["restore"](None, None, "en")[-1], "восстановление без источника")
    _assert_english(handlers["open_outputs"]("en"), "открытие каталога")


def test_reference_captions_follow_the_language(monkeypatch, tmp_path):
    from PIL import Image

    single = [Image.new("RGB", (8, 8))]
    _assert_english(tab_generate._captions(single, "en")[0], "подпись единственного референса")
    assert CYRILLIC.search(tab_generate._captions(single, "ru")[0])
    # Теги при двух и более изображениях языком не управляются: это литералы
    # протокола модели, а не текст для человека.
    pair = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    assert tab_generate._captions(pair, "en") == tab_generate._captions(pair, "ru")


def test_a_model_failure_is_reported_in_english_too(monkeypatch, tmp_path):
    handlers, studio = _english_handlers(monkeypatch, tmp_path)

    class _Boom:
        def generate(self, _request, progress=None):
            raise RuntimeError("out of memory")

    studio._generator = _Boom()
    _images, status = handlers["run"](
        "cat", "", False, [], presets.DEFAULT, "1:1", 1, [], "", 1.0, -1, True, "en",
        progress=lambda *args, **kwargs: None,
    )
    _assert_english(status, "сообщение о сбое генерации")

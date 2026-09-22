"""Журналирование не должно падать на кириллице независимо от консоли.

Весь проект логирует по-русски, а на Windows консоль по умолчанию открыта в
однобайтовой кодовой странице (cp1252, cp866...), которая кириллицу не
представляет. Без принудительной перенастройки потока первое же сообщение
уровня INFO валит логирование с UnicodeEncodeError — оболочка теряет основной
канал наблюдения именно тогда, когда он нужнее всего (при чтении лога вживую
на задачах 10-15).
"""

from __future__ import annotations

import ast
import io
import logging

import pytest

from fooocus_qwen import config, logging_setup
from fooocus_qwen.logging_setup import _configure_console_stream


@pytest.fixture
def isolated_root_logger():
    """Возвращает состояние root-логгера после теста.

    setup_logging() чистит и переставляет обработчики глобально; без отката
    последующие тесты в этом же процессе получили бы файловый и консольный
    обработчики, оставленные предыдущим тестом.
    """
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    yield root
    root.handlers.clear()
    root.handlers.extend(handlers)
    root.setLevel(level)


def test_reconfigurable_stream_is_switched_to_utf8_with_replace():
    # Симулируем реальную консоль Windows: TextIOWrapper поверх cp1252 со
    # строгой обработкой ошибок — ровно то, на чём кириллица падала раньше.
    console_like = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")

    _configure_console_stream(console_like)

    assert console_like.encoding.lower() == "utf-8"
    assert console_like.errors == "replace"


def test_unconfigured_cp1252_stream_actually_raises_on_cyrillic():
    # Доказывает, что дефект воспроизводим, а не выдуман: без настройки
    # запись кириллицы в такой поток действительно падает.
    console_like = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    with pytest.raises(UnicodeEncodeError):
        console_like.write("Загружаю модель")
        console_like.flush()


def test_configured_stream_accepts_cyrillic_without_raising():
    buffer = io.BytesIO()
    console_like = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")
    _configure_console_stream(console_like)

    console_like.write("Загружаю модель")
    console_like.flush()

    assert buffer.getvalue().decode("utf-8") == "Загружаю модель"


def test_stream_without_reconfigure_is_left_alone():
    class NoReconfigure:
        def write(self, text: str) -> None:
            pass

    stream = NoReconfigure()
    # Не должно падать даже там, где перенастроить поток нечем (например,
    # под pytest в некоторых режимах захвата вывода).
    assert _configure_console_stream(stream) is stream


def test_stream_whose_reconfigure_fails_is_tolerated():
    class HostileStream:
        def reconfigure(self, **kwargs) -> None:
            raise ValueError("этот поток настраиваться отказывается")

    stream = HostileStream()
    assert _configure_console_stream(stream) is stream


def test_setup_logging_does_not_raise_on_cyrillic(tmp_path, monkeypatch, isolated_root_logger):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    logging_setup.setup_logging(verbose=True)

    logger = logging.getLogger("test.logging_setup.cyrillic")
    logger.info("Загружаю модель из %s", "Qwen-Image-2.1")


def test_console_handler_stream_is_configured_for_utf8(tmp_path, monkeypatch, isolated_root_logger):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    logging_setup.setup_logging(verbose=False)

    console_handlers = [
        handler
        for handler in isolated_root_logger.handlers
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
    ]
    assert len(console_handlers) == 1

    stream = console_handlers[0].stream
    # Проверяется настройка потока, а не факт отсутствия исключения на
    # потоке, который и так уже был в UTF-8 (типичная ситуация не под
    # Windows) — иначе тест прошёл бы и без исправления.
    if hasattr(stream, "reconfigure"):
        assert stream.encoding.lower() == "utf-8"
        assert stream.errors == "replace"


# --- публичный вход для точек входа без логгера --------------------------


def test_use_utf8_console_configures_both_standard_streams(monkeypatch):
    """Оба потока, а не только stdout.

    Инструменты в ``tools/`` печатают по-русски раньше, чем существует
    логгер: argparse выводит справку ``--help`` до первой строки кода
    ``main()``. Поэтому вход публичный и зовётся первым делом. Стандартная
    ошибка настраивается вместе со стандартным выводом: от читаемой справки
    мало толку, если сообщение о сбое всё равно нечитаемо.
    """
    out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    err = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr("sys.stdout", out)
    monkeypatch.setattr("sys.stderr", err)

    logging_setup.use_utf8_console()

    for stream, name in ((out, "stdout"), (err, "stderr")):
        assert stream.encoding.lower() == "utf-8", f"{name} остался в {stream.encoding}"
        stream.write("Генерация завершена\n")  # не должно бросить


def test_use_utf8_console_survives_streams_without_reconfigure(monkeypatch):
    # Под некоторыми перехватчиками вывода у потока нет reconfigure(); это не
    # повод падать до вывода справки — именно там вызов и стоит.
    class _Plain:
        encoding = "cp1252"

        def write(self, _text: str) -> int:
            return 0

    monkeypatch.setattr("sys.stdout", _Plain())
    monkeypatch.setattr("sys.stderr", _Plain())

    logging_setup.use_utf8_console()  # отсутствие исключения и есть проверка


def test_every_tool_prepares_the_console_before_parsing_arguments():
    """Правило распространяется на все точки входа, а не на ту, где заметили.

    Сломалось это одинаково во всех инструментах сразу, и починка в одном
    ничего не говорит про остальные. Проверяются **все** скрипты в
    ``tools/`` и его подкаталогах, а не только те, что разбирают аргументы:
    ``fetch_styles.py`` и ``fetch_system_prompts.py`` argparse не
    используют, печатают по-русски и ровно поэтому выпали из первой
    редакции этой проверки, оставшись сломанными. Там, где argparse есть,
    вызов обязан стоять ещё и выше него: справка ``--help`` печатается до
    первой строки ``main()``.
    """
    # rglob, а не glob: опытные скрипты лежат в подкаталоге ``experiments``,
    # и при переезде туда гарантия для них молча исчезла бы.
    tools = config.PROJECT_ROOT / "tools"
    scripts = sorted(tools.rglob("*.py"))
    assert scripts, "инструменты не найдены — проверка выродилась бы в пустую"

    for script in scripts:
        source = script.read_text(encoding="utf-8")
        tree = ast.parse(source)
        parses = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "ArgumentParser"
        ]
        prepares = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "use_utf8_console"
        ]
        # Разбор дерева, а не поиск подстроки: закомментированный вызов текст
        # содержит, а выполнять его никто не будет. Первая редакция этого
        # теста искала подстроку и спокойно проходила с отключённой починкой —
        # проверено диверсией.
        assert prepares, f"{script.name}: консоль не подготовлена"
        if parses:
            assert min(prepares) < min(parses), (
                f"{script.name}: подготовка консоли (строка {min(prepares)}) стоит ниже "
                f"разбора аргументов (строка {min(parses)})"
            )

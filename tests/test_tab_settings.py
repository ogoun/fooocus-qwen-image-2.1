"""Вкладка «Настройки»: адрес языковой модели и системные промты.

Отсутствие настроенной языковой модели — обычное состояние свежей установки,
а не ошибка (см. решения задачи), поэтому первый и главный тест здесь —
что проверка связи не роняет вкладку, если файла адреса нет вовсе.
"""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.ui import tab_settings
from fooocus_qwen.ui.i18n import Localizer
from fooocus_qwen.ui.state import Studio


def _build_handlers(monkeypatch, tmp_path, studio=None):
    monkeypatch.setattr(config, "ENDPOINT_FILE", tmp_path / "llm_endpoint.txt")
    monkeypatch.setattr(config, "SYSTEM_PROMPT_DIR", tmp_path)
    if studio is None:
        studio = Studio(config.AppConfig())

    with gr.Blocks() as demo:
        localizer = Localizer(studio.config.lang)
        tab_settings.build(studio, localizer)

    handlers = {}
    for block_fn in demo.fns.values():
        handlers.setdefault(block_fn.fn.__name__, block_fn.fn)
    return handlers, studio


# --- главное обещание: проверка связи не бросает исключение ---


def test_check_connection_reports_readable_failure_when_no_endpoint_is_configured(monkeypatch, tmp_path):
    # Файла llm_endpoint.txt нет вовсе — обычное состояние до первой настройки.
    handlers, _studio = _build_handlers(monkeypatch, tmp_path)

    message = handlers["check_connection"]("ru")
    assert isinstance(message, str)
    assert message  # непустое сообщение, а не молчаливый провал
    assert "связ" in message.lower() or "модел" in message.lower()


def test_check_connection_reports_failure_when_the_server_is_unreachable(monkeypatch, tmp_path):
    (tmp_path / "llm_endpoint.txt").write_text("127.0.0.1:1\n", encoding="utf-8")
    handlers, _studio = _build_handlers(monkeypatch, tmp_path)

    message = handlers["check_connection"]("ru")
    assert isinstance(message, str) and message


# --- адрес языковой модели ---


def test_store_endpoint_writes_the_file(monkeypatch, tmp_path):
    handlers, _studio = _build_handlers(monkeypatch, tmp_path)

    handlers["store_endpoint"]("llama.cpp\n192.168.0.1:8000\n", "ru")
    assert (tmp_path / "llm_endpoint.txt").read_text(encoding="utf-8") == "llama.cpp\n192.168.0.1:8000\n"


def test_read_endpoint_missing_file_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENDPOINT_FILE", tmp_path / "нет_такого_файла.txt")
    text = tab_settings._read_endpoint("ru")
    assert isinstance(text, str) and text


# --- системные промты ---


def test_load_prompt_file_reads_from_the_system_prompt_dir(monkeypatch, tmp_path):
    (tmp_path / "system_prompt_t2i.txt").write_text("исходный текст", encoding="utf-8")
    handlers, _studio = _build_handlers(monkeypatch, tmp_path)

    assert handlers["load_prompt_file"]("system_prompt_t2i.txt", "ru") == "исходный текст"


def test_store_prompt_file_writes_the_edited_text_to_disk(monkeypatch, tmp_path):
    (tmp_path / "system_prompt_edit.txt").write_text("старый текст", encoding="utf-8")
    handlers, _studio = _build_handlers(monkeypatch, tmp_path)

    message = handlers["store_prompt_file"]("system_prompt_edit.txt", "новый текст", "ru")
    assert (tmp_path / "system_prompt_edit.txt").read_text(encoding="utf-8") == "новый текст"
    assert "system_prompt_edit.txt" in message


def test_read_prompt_missing_file_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SYSTEM_PROMPT_DIR", tmp_path)
    text = tab_settings._read_prompt("system_prompt_describe.txt", "ru")
    assert isinstance(text, str) and text

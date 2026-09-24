"""Вкладка «Настройки»: адрес языковой модели и системные промты.

Отсутствие настроенной языковой модели — обычное состояние свежей установки,
а не ошибка (см. решения задачи), поэтому первый и главный тест здесь —
что проверка связи не роняет вкладку, если файла адреса нет вовсе.
"""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.llm import load_endpoint
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


SECRET = "s3cret-token-value"


def _configure(tmp_path, text):
    (tmp_path / "llm_endpoint.txt").write_text(text, encoding="utf-8")


def test_the_token_never_reaches_the_browser(monkeypatch, tmp_path):
    """Интерфейс слушает 0.0.0.0: всё, что в нём показано, видит вся сеть.

    Раньше вкладка показывала файл адреса целиком, с токеном открытым
    текстом. Теперь токена нет ни в одном значении, которое уходит в
    браузер, — проверяется по конфигурации всего интерфейса, а не по
    одному полю.
    """
    _configure(tmp_path, "llama.cpp\n192.0.2.10:8000\ntoken=" + SECRET + "\n")
    monkeypatch.setattr(config, "ENDPOINT_FILE", tmp_path / "llm_endpoint.txt")
    monkeypatch.setattr(config, "SYSTEM_PROMPT_DIR", tmp_path)
    with gr.Blocks() as demo:
        tab_settings.build(Studio(config.AppConfig()), Localizer("ru"))
    assert SECRET not in str(demo.get_config_file())


def test_the_address_is_shown_and_the_token_only_as_present(monkeypatch, tmp_path):
    _configure(tmp_path, "192.0.2.10:8000\ntoken=" + SECRET + "\n")
    monkeypatch.setattr(config, "ENDPOINT_FILE", tmp_path / "llm_endpoint.txt")
    text = tab_settings.describe_endpoint("ru")
    assert "192.0.2.10:8000" in text and "токен задан" in text and SECRET not in text


def test_saving_with_an_empty_token_keeps_the_current_one(monkeypatch, tmp_path):
    """Сервер токен не показывает — значит, пустое поле не может его стереть."""
    _configure(tmp_path, "llama.cpp\n192.0.2.10:8000\ntoken=" + SECRET + "\n")
    handlers, _studio = _build_handlers(monkeypatch, tmp_path)

    cleared, message = handlers["store_endpoint"]("192.0.2.20:9000", "", "ru")

    saved = load_endpoint(tmp_path / "llm_endpoint.txt")
    assert saved.base_url == "http://192.0.2.20:9000"
    assert saved.token == SECRET
    assert saved.backend == "llama.cpp", "имя бэкенда не должно теряться при правке адреса"
    assert cleared == "" and SECRET not in message


def test_a_new_token_replaces_the_old_one(monkeypatch, tmp_path):
    _configure(tmp_path, "192.0.2.10:8000\ntoken=old\n")
    handlers, _studio = _build_handlers(monkeypatch, tmp_path)
    handlers["store_endpoint"]("192.0.2.10:8000", "new", "ru")
    assert load_endpoint(tmp_path / "llm_endpoint.txt").token == "new"


def test_the_token_can_be_removed(monkeypatch, tmp_path):
    _configure(tmp_path, "192.0.2.10:8000\ntoken=" + SECRET + "\n")
    handlers, _studio = _build_handlers(monkeypatch, tmp_path)
    message = handlers["forget_token"]("ru")
    assert load_endpoint(tmp_path / "llm_endpoint.txt").token is None
    assert "без токена" in message


def test_saving_without_an_address_changes_nothing(monkeypatch, tmp_path):
    _configure(tmp_path, "192.0.2.10:8000\n")
    handlers, _studio = _build_handlers(monkeypatch, tmp_path)
    _cleared, message = handlers["store_endpoint"]("   ", "x", "ru")
    assert "адрес" in message.lower()
    assert load_endpoint(tmp_path / "llm_endpoint.txt").base_url == "http://192.0.2.10:8000"


def test_a_fresh_install_says_what_works_without_a_language_model(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENDPOINT_FILE", tmp_path / "нет_такого_файла.txt")
    text = tab_settings.describe_endpoint("ru")
    assert "не настроена" in text


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


def test_a_prompt_file_outside_the_list_is_never_written(monkeypatch, tmp_path):
    """Имя приходит из браузера и ложится в путь записи."""
    handlers, _studio = _build_handlers(monkeypatch, tmp_path)
    message = handlers["store_prompt_file"]("../escape.txt", "злой текст", "ru")
    assert not (tmp_path.parent / "escape.txt").exists()
    assert "escape" in message


def test_opening_the_tab_rereads_the_address(monkeypatch, tmp_path):
    """Поля собираются при запуске; открытие вкладки обязано их обновить.

    Иначе вкладка показывала бы адрес на момент старта процесса и после
    правки файла руками или из другого окна — устаревшее состояние.
    """
    monkeypatch.setattr(config, "ENDPOINT_FILE", tmp_path / "llm_endpoint.txt")
    monkeypatch.setattr(config, "SYSTEM_PROMPT_DIR", tmp_path)
    with gr.Blocks():
        components = tab_settings.build(Studio(config.AppConfig()), Localizer("ru"))
    assert components["address"].value == ""

    _configure(tmp_path, "192.0.2.10:8000\ntoken=" + SECRET + "\n")
    address, status, _memory, _precision, _sage, _performance = components["refresh"]("ru")
    assert "192.0.2.10:8000" in address
    assert "192.0.2.10:8000" in status and SECRET not in status and SECRET not in address


# --- производительность: точность и SageAttention ---


def _performance(monkeypatch, tmp_path):
    handlers, studio = _build_handlers(monkeypatch, tmp_path)
    studio.config = config.AppConfig(preload=False)
    return handlers, studio


def _quiet(*_args, **_kwargs):
    pass


def test_the_same_precision_is_not_reapplied(monkeypatch, tmp_path):
    handlers, studio = _performance(monkeypatch, tmp_path)
    monkeypatch.setattr(studio, "ensure_precision_weights", lambda _p: pytest.fail("качать нечего"))
    message = handlers["apply_precision"]("bf16", "ru", progress=_quiet)
    assert "уже выбрана" in message


def test_switching_precision_fetches_weights_then_unloads(monkeypatch, tmp_path):
    from fooocus_qwen import settings

    handlers, studio = _performance(monkeypatch, tmp_path)
    steps = []
    monkeypatch.setattr(studio, "ensure_precision_weights", lambda p: steps.append(("fetch", p)))
    monkeypatch.setattr(studio, "unload", lambda: steps.append("unload"))
    message = handlers["apply_precision"]("int8", "ru", progress=_quiet)
    assert steps == [("fetch", "int8"), "unload"], "сначала веса, потом выгрузка — иначе загрузиться было бы не из чего"
    assert settings.load().precision == "int8"
    assert "INT8" in message


def test_a_failed_download_leaves_the_precision_alone(monkeypatch, tmp_path):
    from fooocus_qwen import settings

    handlers, studio = _performance(monkeypatch, tmp_path)

    def fail(_precision):
        raise OSError("нет сети")

    monkeypatch.setattr(studio, "ensure_precision_weights", fail)
    monkeypatch.setattr(studio, "unload", lambda: pytest.fail("выгружать модель без весов нельзя"))
    message = handlers["apply_precision"]("int8", "ru", progress=_quiet)
    assert "нет сети" in message and "не изменена" in message
    assert settings.load().precision == "bf16"


def test_the_sage_switch_is_remembered(monkeypatch, tmp_path):
    from fooocus_qwen import settings

    handlers, _studio = _performance(monkeypatch, tmp_path)
    status = handlers["toggle_sage"](True, "en")
    assert settings.load().sage_attention is True
    assert "Precision: BF16" in status

"""Запуск через run.ps1 / run.sh открывает страницу в браузере.

Открывает её приложение, а не скрипт оболочки, и по одной причине: скрипт не
знает, когда сервер поднялся. Python с torch и diffusers стартует секунды,
и браузер, открытый сразу после запуска процесса, упёрся бы в «не удаётся
подключиться». Приложение же знает точный момент: ``demo.launch`` возвращает
управление после того, как Gradio сам достучался до своего сервера.

Адрес — ``127.0.0.1``, а не ``localhost``: на машине заказчика имя localhost
разрешается сначала в ``::1``, где никто не слушает, и каждое обращение
стоит лишних 4.4 секунды (см.
`docs/research/2026-09-22-zapusk-padal-na-proverke-localhost.md`).
"""

from __future__ import annotations

import pytest

from fooocus_qwen import config


def test_the_flag_is_off_unless_asked():
    """Прямой запуск модуля браузер не открывает: так удобнее отлаживать."""
    assert config.AppConfig().open_browser is False
    assert config.build_parser().parse_args([]).open_browser is False
    assert config.build_parser().parse_args(["--open-browser"]).open_browser is True
    assert config.parse_args(["--open-browser"]).open_browser is True


def test_the_address_is_the_loopback_not_the_listening_host():
    """``0.0.0.0`` — это «слушать везде», открывать по нему нельзя."""
    gr = pytest.importorskip("gradio")  # noqa: F841 — app тянет gradio
    from fooocus_qwen.ui import app

    assert app.browser_url(config.AppConfig(host="0.0.0.0", port=7865)) == "http://127.0.0.1:7865"
    assert app.browser_url(config.AppConfig(host="::", port=7865)) == "http://127.0.0.1:7865"


def test_an_explicit_host_is_respected():
    """Если адрес задан руками, открывать надо его: 127.0.0.1 там не слушают."""
    gr = pytest.importorskip("gradio")  # noqa: F841
    from fooocus_qwen.ui import app

    assert app.browser_url(config.AppConfig(host="192.0.2.10", port=7870)) == "http://192.0.2.10:7870"


def test_localhost_is_never_used():
    """Имя вместо адреса стоит здесь 4.4 секунды — и на старте они заметны."""
    gr = pytest.importorskip("gradio")  # noqa: F841
    from fooocus_qwen.ui import app

    assert "localhost" not in app.browser_url(config.AppConfig())


def test_the_browser_opens_after_the_server_is_up(monkeypatch):
    """Порядок важнее всего: страница открывается, когда её уже отдают."""
    gr = pytest.importorskip("gradio")  # noqa: F841
    from fooocus_qwen.ui import app

    events: list[str] = []

    class _Demo:
        def queue(self, **_kwargs):
            return self

        def launch(self, **_kwargs):
            events.append("launch")

        def block_thread(self):
            events.append("block")

    class _Studio:
        def preload_in_background(self):
            events.append("preload")

    monkeypatch.setattr(app, "build", lambda cfg, return_studio=False: (_Demo(), _Studio()))
    monkeypatch.setattr(app, "open_in_browser", lambda url: events.append("browser") or True)

    app.launch(config.AppConfig(open_browser=True))

    assert events.index("launch") < events.index("browser"), "браузер открылся до готовности сервера"
    assert events.index("browser") < events.index("block")


def test_nothing_opens_without_the_flag(monkeypatch):
    gr = pytest.importorskip("gradio")  # noqa: F841
    from fooocus_qwen.ui import app

    opened: list[str] = []

    class _Demo:
        def queue(self, **_kwargs):
            return self

        def launch(self, **_kwargs):
            pass

        def block_thread(self):
            pass

    monkeypatch.setattr(app, "build", lambda cfg, return_studio=False: (_Demo(), type("S", (), {"preload_in_background": lambda self: None})()))
    monkeypatch.setattr(app, "open_in_browser", lambda url: opened.append(url) or True)

    app.launch(config.AppConfig(open_browser=False, preload=False))
    assert not opened


def test_a_machine_without_a_browser_still_starts(monkeypatch):
    """На сервере без графики открывать нечем — это не повод падать.

    Приложение запускают и по SSH, и из планировщика; уронить рабочий сервер
    из-за ненайденного браузера значило бы поменять местами главное и
    второстепенное.
    """
    gr = pytest.importorskip("gradio")  # noqa: F841
    from fooocus_qwen.ui import app

    def explode(*_args, **_kwargs):
        raise RuntimeError("браузера нет")

    monkeypatch.setattr(app.webbrowser, "open", explode)
    assert app.open_in_browser("http://127.0.0.1:7865") is False


def test_the_user_can_turn_it_off_even_though_the_script_turns_it_on():
    """`run.ps1` ставит --open-browser первым; ключ пользователя должен победить.

    Иначе обещание в справке скрипта оказалось бы враньём: отключить
    открытие браузера было бы нечем, кроме правки самого скрипта.
    """
    assert config.parse_args(["--open-browser", "--no-open-browser"]).open_browser is False
    assert config.parse_args(["--open-browser"]).open_browser is True
    assert config.parse_args(["--no-open-browser"]).open_browser is False

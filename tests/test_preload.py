"""Модель грузится в фоне, пока пользователь набирает промт.

Загрузка занимает от двадцати до пятидесяти секунд — чтение весов с диска
и подготовка закреплённых копий на хосте. До этой правки их платил первый
же запрос, причём молча: прогресс приходит из обратного вызова пайплайна,
то есть уже после первого шага денойзинга, и отличить работу от зависания
было нельзя (см. `docs/research/2026-09-22-pravka-ne-pomeshalas.md`).

Теперь запуск греет модель сам. Ленивость при этом никуда не делась: сборка
интерфейса модель по-прежнему не трогает, и с `--no-preload` поведение в
точности прежнее.
"""

from __future__ import annotations

import threading
import time

import pytest

from fooocus_qwen import config
from fooocus_qwen.ui.state import Studio


class _SlowLoad:
    """Считает обращения к генератору и изображает долгую загрузку."""

    def __init__(self, delay: float = 0.05) -> None:
        self.calls = 0
        self._delay = delay
        self._lock = threading.Lock()

    def __call__(self, *_args, **_kwargs):
        with self._lock:
            self.calls += 1
        time.sleep(self._delay)
        # Тройка, которую отдаёт loader.load: пайплайн, размещение, кэш.
        return object(), object(), object()


def _studio(monkeypatch, loader_stub) -> Studio:
    """Подменяется загрузчик, а не само свойство ``generator``.

    Первая редакция подменяла свойство своей копией — и проверяла тем самым
    собственную заглушку, а не код. Диверсия это показала: снятие двойной
    проверки внутри замка тесты проходили насквозь. Теперь настоящий
    ``Studio.generator`` со своим замком и есть предмет проверки.
    """
    monkeypatch.setattr("fooocus_qwen.engine.loader.load", loader_stub)
    monkeypatch.setattr(
        "fooocus_qwen.engine.generator.Generator",
        lambda pipe, residency, cache, catalogue: pipe,
    )
    return Studio(config.AppConfig())


def test_preload_loads_the_model_without_a_request(monkeypatch):
    stub = _SlowLoad()
    studio = _studio(monkeypatch, stub)

    assert not studio.model_loaded
    studio.preload_in_background()

    deadline = time.perf_counter() + 5
    while not studio.model_loaded and time.perf_counter() < deadline:
        time.sleep(0.01)

    assert studio.model_loaded, "фоновая загрузка не довела модель до готовности"
    assert stub.calls == 1


def test_a_request_during_preload_does_not_load_twice(monkeypatch):
    """Замок и двойная проверка обязаны выдержать гонку.

    Без них запрос, пришедший посреди фоновой загрузки, начал бы вторую:
    тридцать три гигабайта весов читались бы дважды одновременно, и карта
    этого не переживёт.
    """
    stub = _SlowLoad(delay=0.2)
    studio = _studio(monkeypatch, stub)

    studio.preload_in_background()
    time.sleep(0.02)  # дать потоку войти в загрузку
    studio.generator  # noqa: B018 — обращение и есть запрос

    assert stub.calls == 1, f"модель загружалась {stub.calls} раза"


def test_preload_on_an_already_loaded_model_does_nothing(monkeypatch):
    stub = _SlowLoad()
    studio = _studio(monkeypatch, stub)
    studio.generator  # noqa: B018
    studio.preload_in_background()
    time.sleep(0.05)
    assert stub.calls == 1


def test_a_failing_preload_does_not_raise(monkeypatch):
    """Показать ошибку некому: интерфейс ещё никто не открыл.

    Настоящий запрос упрётся в ту же ошибку и покажет её строкой состояния,
    как и раньше. Уронить поток — значит уронить приложение на старте.
    """
    def explode(*_args, **_kwargs):
        raise FileNotFoundError("весов нет")

    studio = _studio(monkeypatch, explode)
    studio.preload_in_background()
    time.sleep(0.1)
    assert not studio.model_loaded


def test_building_the_ui_still_does_not_touch_the_model(monkeypatch):
    """Ленивость сохранена: греет запуск, а не сборка.

    Сборку интерфейса делают и тесты, и режим ``--selftest``; тащить за ней
    тридцать три гигабайта весов было бы не ускорением, а подарком наоборот.
    """
    gr = pytest.importorskip("gradio")
    from fooocus_qwen.ui import app

    stub = _SlowLoad()
    monkeypatch.setattr("fooocus_qwen.engine.loader.load", stub)

    with gr.Blocks():
        demo = app.build(config.AppConfig())

    assert demo is not None
    assert stub.calls == 0


def test_preload_can_be_switched_off():
    assert config.AppConfig().preload is True
    assert config.AppConfig(preload=False).preload is False


def test_preload_starts_only_after_the_server_passed_its_checks(monkeypatch):
    """Веса читаются после того, как Gradio проверил доступность localhost.

    Gradio заканчивает запуск контрольным ``HEAD`` на собственный адрес и с
    таймаутом в три секунды (`gradio/networking.py`, ``url_ok``); не дождался —
    решает, что localhost недоступен, и валит запуск требованием ``share=True``.
    Греющий поток в этот момент читает тридцать три гигабайта весов и отнимает
    у сервера и диск, и GIL: на прогретом файловом кэше проверка успевает, на
    холодном — нет, и запуск падает через раз (см.
    `docs/research/2026-09-22-zapusk-padal-na-proverke-localhost.md`).

    Лечится не таймаутом, а порядком: ``prevent_thread_lock=True`` возвращает
    управление сразу после всех проверок, греем мы уже потом, а поток держим
    сами тем же ``block_thread``, что позвал бы и сам Gradio.
    """
    events: list[tuple] = []

    class _Demo:
        def queue(self, **_kwargs):
            return self

        def launch(self, **kwargs):
            events.append(("launch", kwargs.get("prevent_thread_lock")))

        def block_thread(self):
            events.append(("block", None))

    class _Studio:
        def preload_in_background(self):
            events.append(("preload", None))

    from fooocus_qwen.ui import app

    monkeypatch.setattr(app, "build", lambda cfg, return_studio=False: (_Demo(), _Studio()))
    app.launch(config.AppConfig())

    assert [name for name, _ in events] == ["launch", "preload", "block"]
    assert events[0][1] is True, "Gradio должен вернуть управление до прогрева"

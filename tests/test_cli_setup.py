"""Установка умеет две вещи помимо зависимостей: достать веса и спросить про LLM.

Обе делаются одной и той же точкой входа для обеих операционных систем:
`install.ps1` и `install.sh` зовут `python -m fooocus_qwen` с флагом, а не
повторяют логику дважды на двух языках оболочки — повторённая логика
разъезжается, и первым это замечает пользователь той системы, которой у
разработчика под рукой нет.
"""

from __future__ import annotations

import sys

import pytest

from fooocus_qwen import __main__ as entry
from fooocus_qwen import config


def test_the_parser_knows_both_flags():
    args = config.build_parser().parse_args(["--fetch-model"])
    assert args.fetch_model is True
    assert config.build_parser().parse_args(["--setup-llm"]).setup_llm is True
    assert config.build_parser().parse_args([]).fetch_model is False


def test_fetch_model_reports_that_weights_were_already_there(monkeypatch, capsys):
    monkeypatch.setattr("fooocus_qwen.engine.fetch.ensure_model", lambda *_a, **_k: False)
    assert entry.main(["--fetch-model"]) == 0
    assert "на месте" in capsys.readouterr().out.lower()


def test_fetch_model_reports_a_download(monkeypatch, capsys):
    monkeypatch.setattr("fooocus_qwen.engine.fetch.ensure_model", lambda *_a, **_k: True)
    assert entry.main(["--fetch-model"]) == 0
    assert "скачан" in capsys.readouterr().out.lower()


def test_a_failed_download_is_a_non_zero_exit(monkeypatch, capsys):
    """Установка обязана остановиться: без весов дальше идти некуда."""
    from fooocus_qwen.engine import fetch

    def explode(*_a, **_k):
        raise fetch.ModelDownloadError("сеть пропала")

    monkeypatch.setattr("fooocus_qwen.engine.fetch.ensure_model", explode)
    assert entry.main(["--fetch-model"]) == 1
    assert "сеть пропала" in capsys.readouterr().out


def test_setup_llm_asks_and_returns_zero(monkeypatch):
    asked = []
    monkeypatch.setattr("fooocus_qwen.llm.setup.configure", lambda *a, **k: asked.append(a) or True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    assert entry.main(["--setup-llm"]) == 0
    assert asked, "диалог не состоялся"


def test_setup_llm_is_skipped_without_a_console(monkeypatch, capsys):
    """Установку запускают и из сценариев: спрашивать там некого.

    Без этой проверки `input()` получил бы конец файла и уронил установку на
    ровном месте — уже после того, как зависимости поставлены.
    """
    asked = []
    monkeypatch.setattr("fooocus_qwen.llm.setup.configure", lambda *a, **k: asked.append(a) or True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

    assert entry.main(["--setup-llm"]) == 0
    assert not asked, "без консоли спрашивать нельзя"
    assert "пропус" in capsys.readouterr().out.lower()


def test_a_refused_llm_setup_is_still_a_success(monkeypatch):
    """Отказ настраивать LLM — не ошибка установки: буст необязателен."""
    monkeypatch.setattr("fooocus_qwen.llm.setup.configure", lambda *a, **k: False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    assert entry.main(["--setup-llm"]) == 0


@pytest.mark.parametrize("flag", ["--fetch-model", "--setup-llm"])
def test_neither_flag_touches_the_model(monkeypatch, flag):
    """Ни один из режимов установки не грузит тридцать три гигабайта весов."""
    monkeypatch.setattr("fooocus_qwen.engine.fetch.ensure_model", lambda *_a, **_k: False)
    monkeypatch.setattr("fooocus_qwen.llm.setup.configure", lambda *a, **k: False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)

    def forbidden(*_a, **_k):
        raise AssertionError("установка не должна грузить модель")

    monkeypatch.setattr("fooocus_qwen.engine.loader.load", forbidden)
    assert entry.main([flag]) == 0

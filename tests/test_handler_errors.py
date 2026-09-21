"""Сбой на пути к модели — это строка состояния, а не трейсбек в тосте.

До этого раунда правок ни в ``ui/tab_generate.py``, ни в ``ui/tab_edit.py`` не
было ни одного ``try``. Отсутствующие веса, испорченный ``model_index.json`` и
нехватка видеопамяти в цикле денойзинга выходили пользователю сырым стеком
вызовов, то есть выглядели как падение приложения, а не как «поставьте пресет
пониже». Отдельно от них стояла библиотека промтов: ``library.load_prompt``
бросает ``FileNotFoundError`` на удалённый мимо приложения пресет и
``ValueError`` на испорченный, и выбор строки в выпадающем списке превращался
в тост с ошибкой — при том, что ``tab_gallery.restore_fields`` тремя файлами
рядом был намеренно сделан снисходительным ровно к такому классу входных
данных.
"""

from __future__ import annotations

import json
import types

import pytest

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.engine import presets
from fooocus_qwen.prompting import library
from fooocus_qwen.ui import tab_edit, tab_generate
from fooocus_qwen.ui.i18n import Localizer
from fooocus_qwen.ui.state import Studio, describe_failure


def _handlers(monkeypatch, tmp_path, module):
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    cfg = config.AppConfig()
    studio = Studio(cfg)

    with gr.Blocks() as demo:
        localizer = Localizer(cfg.lang)
        module.build(studio, localizer)

    found = {}
    for block_fn in demo.fns.values():
        found.setdefault(block_fn.fn.__name__, block_fn.fn)
    return studio, found


# --- библиотека промтов ---


def test_loading_a_preset_deleted_outside_the_application_reports_instead_of_raising(
    monkeypatch, tmp_path
):
    _studio, handlers = _handlers(monkeypatch, tmp_path, tab_generate)

    handlers["save"]("исчезающий", "промт", "", [], "LowQuality", "1:1", -1, 1.0, "ru")
    for path in tmp_path.glob("*.json"):
        path.unlink()

    result = handlers["load"]("исчезающий", "ru")
    assert "не загружен" in result[-1]


def test_loading_a_corrupt_preset_reports_instead_of_raising(monkeypatch, tmp_path):
    _studio, handlers = _handlers(monkeypatch, tmp_path, tab_generate)

    handlers["save"]("битый", "промт", "", [], "LowQuality", "1:1", -1, 1.0, "ru")
    saved = next(iter(tmp_path.glob("*.json")))
    saved.write_text("{ это не JSON", encoding="utf-8")

    result = handlers["load"]("битый", "ru")
    assert "не загружен" in result[-1]


def test_a_preset_whose_json_is_not_a_dictionary_is_reported_too(monkeypatch, tmp_path):
    # `_load_payload` считает повреждённым и валидный JSON, который не словарь.
    _studio, handlers = _handlers(monkeypatch, tmp_path, tab_generate)

    handlers["save"]("список", "промт", "", [], "LowQuality", "1:1", -1, 1.0, "ru")
    saved = next(iter(tmp_path.glob("*.json")))
    saved.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    result = handlers["load"]("список", "ru")
    assert "не загружен" in result[-1]


def test_delete_survives_a_filesystem_refusal(monkeypatch, tmp_path):
    _studio, handlers = _handlers(monkeypatch, tmp_path, tab_generate)
    handlers["save"]("занятый", "промт", "", [], "LowQuality", "1:1", -1, 1.0, "ru")

    def refuse(*_args, **_kwargs):
        raise PermissionError("файл занят другой программой")

    monkeypatch.setattr(library, "delete_prompt", refuse)

    _update, message = handlers["delete"]("занятый", "ru")
    assert "не удалён" in message


# --- классификация сбоя ---


class _FakeOutOfMemory(RuntimeError):
    pass


def test_out_of_memory_is_named_as_such(monkeypatch):
    import torch

    error = torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    text = describe_failure(error, "ru")
    assert "видеопамят" in text.lower()
    assert "пресет" in text.lower()


def test_a_missing_file_is_named_as_such():
    text = describe_failure(FileNotFoundError("model_index.json"), "ru")
    assert "не найден" in text.lower()


def test_an_unexpected_error_still_names_its_type():
    text = describe_failure(KeyError("transformer"), "ru")
    assert "KeyError" in text


def test_a_multiline_message_is_squeezed_into_one_line():
    # Сообщение torch о нехватке памяти — несколько строк со сводкой
    # аллокатора; поле состояния однострочное, полный текст идёт в журнал.
    text = describe_failure(RuntimeError("первая строка\nвторая строка\n" + "x" * 500), "ru")
    assert "\n" not in text
    assert len(text) < 400


# --- Studio.run_generation ---


class _ExplodingGenerator:
    """Двойник генератора: сбой в ``generate()`` и вызов ``recover()`` после него.

    ``recover()`` списан с настоящего ``Generator.recover()`` в миниатюре —
    после FIX 2 ``Studio.recover_residency()`` маршрутизирует вызов через
    генератор, а не обращается к ``ResidencyManager`` напрямую, и любой его
    двойник обязан отвечать на ``recover()``. При отсутствии резидентности
    (как в тестах ниже, где она не нужна) ничего не делает — тот же контракт,
    что и у боевого метода.
    """

    def __init__(self, error: BaseException, residency=None) -> None:
        self._error = error
        self._residency = residency
        self.calls = 0

    def generate(self, _request, progress=None):
        self.calls += 1
        raise self._error

    def recover(self) -> None:
        if self._residency is not None:
            self._residency.restore()


class _RecordingResidency:
    def __init__(self) -> None:
        self.restored = 0

    def restore(self) -> None:
        self.restored += 1


def _request():
    from fooocus_qwen.engine.generator import GenerationRequest

    return GenerationRequest(prompt="кот", preset=presets.get("LowQuality"))


def test_run_generation_turns_a_failure_into_a_message_and_restores_residency():
    studio = Studio(config.AppConfig())
    residency = _RecordingResidency()
    studio._generator = _ExplodingGenerator(RuntimeError("модель не загрузилась"), residency)

    produced, failure = studio.run_generation(_request(), "ru")

    assert produced == []
    assert failure is not None and "модель не загрузилась" in failure
    # Сбой мог оборвать перестановку моделей где угодно; без восстановления
    # трансформер остался бы на хосте, а следующий запрос попал бы в кэш
    # эмбеддингов и не зашёл бы в text_encoder_resident() чинить это.
    assert residency.restored == 1


def test_run_generation_reports_success_without_touching_residency():
    studio = Studio(config.AppConfig())
    studio._generator = types.SimpleNamespace(generate=lambda request, progress=None: ["картинка"])
    residency = _RecordingResidency()
    studio._residency = residency

    produced, failure = studio.run_generation(_request(), "ru")

    assert produced == ["картинка"]
    assert failure is None
    assert residency.restored == 0


def test_recover_residency_is_safe_when_restoring_itself_fails():
    # Перехват исключения переехал внутрь Generator.recover() вместе с замком
    # (FIX 2): Studio лишь маршрутизирует вызов. Поэтому здесь нужен настоящий
    # Generator, а не двойник, — иначе фиктивный try/except в тесте мог бы
    # разойтись с боевым и замаскировать регрессию.
    from fooocus_qwen.engine.generator import Generator

    class _BrokenResidency:
        def restore(self):
            raise RuntimeError("и восстановление не удалось")

    studio = Studio(config.AppConfig())
    studio._generator = Generator(pipe=None, residency=_BrokenResidency(), cache=None, catalogue={})
    studio.recover_residency()  # не должно выбросить наружу


# --- обработчики генерации и правки ---


def test_generate_handler_reports_a_model_failure_in_the_status_line(monkeypatch, tmp_path):
    studio, handlers = _handlers(monkeypatch, tmp_path, tab_generate)
    studio._generator = _ExplodingGenerator(FileNotFoundError("model_index.json"))

    images, status = handlers["run"](
        "кот", "", False, [], presets.DEFAULT, "1:1", 1, [], "", 1.0, -1, True, "ru",
        progress=lambda *args, **kwargs: None,
    )

    assert images == []
    assert "model_index.json" in status


def test_edit_handler_reports_a_model_failure_in_the_status_line(monkeypatch, tmp_path):
    from PIL import Image

    studio, handlers = _handlers(monkeypatch, tmp_path, tab_edit)
    studio._generator = _ExplodingGenerator(RuntimeError("денойзинг не удался"))

    value = {"background": Image.new("RGBA", (64, 64), "red"), "layers": [], "composite": None}
    images, status = handlers["run"](
        value, "убрать фон", False, "none", presets.DEFAULT, 8, 12, True, -1, "ru",
        progress=lambda *args, **kwargs: None,
    )

    assert images == []
    assert "денойзинг не удался" in status


def test_generate_handler_survives_a_broken_argument(monkeypatch, tmp_path):
    # Не только модель: любой сбой внутри обработчика (здесь — нечисловой сид,
    # который роняет int()) обязан стать строкой состояния.
    _studio, handlers = _handlers(monkeypatch, tmp_path, tab_generate)

    images, status = handlers["run"](
        "кот", "", False, [], presets.DEFAULT, "1:1", 1, [], "", 1.0, "не число", True, "ru",
        progress=lambda *args, **kwargs: None,
    )

    assert images == []
    assert status

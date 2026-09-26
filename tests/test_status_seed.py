"""Сид результата — в строке состояния и генерации, и правки.

Сид — то, чем удачный результат повторяют. Генерация его показывала, правка
молчала: повторить удачную правку было нечем, кроме как читать метаданные
файла. Обработчики вызываются настоящие, генератор подставной — он отдаёт
картинки с заранее известными сидами.
"""

from __future__ import annotations

import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.engine import presets
from fooocus_qwen.engine.generator import GeneratedImage
from fooocus_qwen.ui import tab_edit, tab_generate
from fooocus_qwen.ui.i18n import Localizer
from fooocus_qwen.ui.state import Studio, seeds_phrase


class _SeededGenerator:
    """Отдаёт по картинке на каждый сид из списка."""

    def __init__(self, seeds: list[int]) -> None:
        self.seeds = seeds

    def generate(self, request, progress=None):
        return [GeneratedImage(Image.new("RGB", (32, 32), "gray"), seed, {"seed": seed}) for seed in self.seeds]


def _handlers(monkeypatch, tmp_path, module, seeds):
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    studio = Studio(config.AppConfig())
    studio._generator = _SeededGenerator(seeds)
    with gr.Blocks() as demo:
        module.build(studio, Localizer("ru"))
    found = {}
    for block_fn in demo.fns.values():
        found.setdefault(block_fn.fn.__name__, block_fn.fn)
    return found


def _quiet(*args, **kwargs):
    return None


def test_one_seed_is_singular_and_several_are_plural():
    assert seeds_phrase([5], "ru") == "Сид: 5"
    assert seeds_phrase([5, 6], "ru") == "Сиды: 5, 6"
    assert seeds_phrase([5], "en") == "Seed: 5"
    assert seeds_phrase([5, 6], "en") == "Seeds: 5, 6"


def test_generation_status_names_the_seeds(monkeypatch, tmp_path):
    handlers = _handlers(monkeypatch, tmp_path, tab_generate, [11, 12])
    _images, status = handlers["run"](
        "кот", "", "", False, [], presets.DEFAULT, "1:1", 2, [], "", 1.0, -1, True, 0, "ru",
        progress=_quiet,
    )
    assert "Сиды: 11, 12" in status


def test_edit_status_names_the_seed(monkeypatch, tmp_path):
    from fooocus_qwen.ui.painter import payload

    handlers = _handlers(monkeypatch, tmp_path, tab_edit, [4242])
    monkeypatch.setattr(payload, "upload_root", lambda: tmp_path)
    value = payload.encode(Image.new("RGBA", (64, 64), "red"), None)
    _images, status = handlers["run"](
        value, "сделай синим", False, "none", presets.DEFAULT, 8, 12, True, -1, [], "ru",
        progress=_quiet,
    )
    assert "Сид: 4242" in status

"""Вкладка «Редактирование»: обработчик расширения холста.

Три режима области проверяются отдельно, в ``test_edit_collect.py`` — там же,
где живёт ``collect``. Здесь — то, что нельзя проверить без сборки вкладки:
обработчик кнопки «Расширить холст» достаёт значение редактора, строит план
расширения и должен вернуть редактору новый холст с помеченной новой площадью,
переключив режим области на «маска».
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.engine import generator as gen
from fooocus_qwen.ui import tab_edit
from fooocus_qwen.ui.i18n import Localizer
from fooocus_qwen.ui.state import Studio


def _build_handlers():
    """Собирает вкладку и достаёт обработчики по имени функции.

    Тот же приём, что и в ``test_tab_generate._build_handlers``: обработчики —
    замыкания внутри ``build`` и не экспортируются, но Gradio хранит исходную
    функцию каждого события в ``Blocks.fns``.
    """
    cfg = config.AppConfig()
    studio = Studio(cfg)

    with gr.Blocks() as demo:
        localizer = Localizer(cfg.lang)
        tab_edit.build(studio, localizer)

    handlers = {}
    for block_fn in demo.fns.values():
        handlers.setdefault(block_fn.fn.__name__, block_fn.fn)
    return handlers


def _editor_value(size=(64, 64)):
    background = Image.new("RGBA", size, (10, 20, 30, 255))
    return {"background": background, "layers": [], "composite": None}


def test_expand_canvas_enlarges_the_background_and_marks_only_the_new_area():
    handlers = _build_handlers()

    new_value, mode, message = handlers["expand_canvas"](_editor_value((64, 64)), ["right"], 0.5)

    background = new_value["background"]
    assert background.size[0] > 64
    assert background.size[1] == 64
    assert mode == gen.MASK_MASK
    assert "64" not in message or "×" in message  # сообщение содержит итоговый размер

    # Painted-слой обязан покрывать ровно новую площадь: старая область — там,
    # где был оригинал, — не должна оказаться помечена как правочная.
    layer = new_value["layers"][0]
    alpha = np.asarray(layer)[..., 3]
    assert alpha[32, 4] == 0  # исходная область (прижата влево) не помечена
    assert alpha[32, background.size[0] - 4] == 255  # новая площадь помечена


def test_expand_canvas_without_source_asks_to_upload_first():
    handlers = _build_handlers()

    update, mode, message = handlers["expand_canvas"](None, ["right"], 0.5)
    assert mode == gen.MASK_MASK
    assert message  # сообщение непустое — просит сначала загрузить изображение


def test_expand_canvas_without_sides_asks_to_choose_one():
    handlers = _build_handlers()

    update, mode, message = handlers["expand_canvas"](_editor_value(), [], 0.5)
    assert mode == gen.MASK_MASK
    assert message

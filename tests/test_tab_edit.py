"""Вкладка «Редактирование»: обработчики расширения холста и запуска правки.

Три режима области сами по себе проверяются отдельно, в
``test_edit_collect.py`` — там же, где живёт ``collect``. Здесь — то, что
нельзя проверить без сборки вкладки: обработчик кнопки «Расширить холст»
достаёт значение редактора, строит план расширения и должен вернуть редактору
новый холст с помеченной новой площадью, переключив режим области на «маска»;
и обработчик «Применить правку» — как результат ``collect()`` превращается в
поле ``mask_mode`` запроса, минуя обращение к модели.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.engine import generator as gen
from fooocus_qwen.imaging import aspect as aspect_module
from fooocus_qwen.ui import tab_edit
from fooocus_qwen.ui.i18n import Localizer
from fooocus_qwen.ui.state import Studio


def _build_handlers(studio=None):
    """Собирает вкладку и достаёт обработчики по имени функции.

    Тот же приём, что и в ``test_tab_generate._build_handlers``: обработчики —
    замыкания внутри ``build`` и не экспортируются, но Gradio хранит исходную
    функцию каждого события в ``Blocks.fns``. Возвращает и сам ``Studio`` —
    тестам, проверяющим ``run()``, нужно подменить его генератор до вызова.
    """
    if studio is None:
        studio = Studio(config.AppConfig())

    with gr.Blocks() as demo:
        localizer = Localizer(studio.config.lang)
        tab_edit.build(studio, localizer)

    handlers = {}
    for block_fn in demo.fns.values():
        handlers.setdefault(block_fn.fn.__name__, block_fn.fn)
    return handlers, studio


def _editor_value(size=(64, 64), painted=None):
    """Значение редактора; при ``painted`` в первом слое закрашен прямоугольник.

    Совпадает по форме со вспомогательной функцией ``editor()`` из
    ``test_edit_collect.py``, но не переиспользует её напрямую — так каждый
    тестовый модуль остаётся самодостаточным (собственный приём проекта: ту же
    независимость видно у ``test_tab_generate.py`` и ``test_outpaint.py``).
    """
    background = Image.new("RGBA", size, (10, 20, 30, 255))
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if painted:
        layer.paste((255, 0, 0, 255), painted)
    composite = background.copy()
    composite.alpha_composite(layer)
    return {"background": background, "layers": [layer], "composite": composite}


class _FakeGenerator:
    """Подменяет ``Generator.generate`` без единого обращения к GPU.

    Единственная задача — запомнить запрос, который построил ``run()``, и
    вернуть пустой список: обработчик тогда просто сообщит "Правка прервана" и
    не полезет ни в ``gallery.next_path``, ни в ``metadata.save_png`` с
    реальным изображением.
    """

    def __init__(self) -> None:
        self.captured: gen.GenerationRequest | None = None

    def generate(self, request, progress=None):
        self.captured = request
        return []


class _NoLoadStudio(Studio):
    """``Studio``, у которой свойство ``generator`` не грузит модель.

    Подменять ``studio._generator`` напрямую нельзя без потери сигнала: тогда
    ``model_loaded`` стало бы ``True`` и тест не смог бы отличить «модель и
    правда не грузилась» от «мы сами её подложили». Переопределение самого
    свойства оставляет ``_generator`` в ``None``, так что ``model_loaded``
    достоверно показывает, что ленивый путь через ``engine.loader.load`` ни
    разу не был пройден.
    """

    def __init__(self, cfg, fake: _FakeGenerator) -> None:
        super().__init__(cfg)
        self._fake = fake

    @property
    def generator(self):
        return self._fake


@pytest.mark.parametrize(
    ("mode_value", "painted", "expected_mode", "expect_mask"),
    [
        (gen.MASK_MASK, (8, 8, 24, 24), gen.MASK_MASK, True),
        (gen.MASK_MASK, None, gen.MASK_NONE, False),
        (gen.MASK_REGION, (8, 8, 24, 24), gen.MASK_REGION, True),
        (gen.MASK_REGION, None, gen.MASK_NONE, False),
        (gen.MASK_ANNOTATION, (8, 8, 24, 24), gen.MASK_ANNOTATION, False),
        # Аннотация без рисунка — тот самый случай, ради которого в выражении
        # `run()` стоит `or mode_value == MASK_ANNOTATION`: у аннотации нет
        # отдельной маски в принципе, `collect()` всегда вернёт mask=None, и
        # naивная проверка "mask is not None" одна, без `or`, откатила бы
        # режим на MASK_NONE даже когда пользователь честно нарисовал пометки.
        (gen.MASK_ANNOTATION, None, gen.MASK_ANNOTATION, False),
    ],
)
def test_run_reconstructs_mask_mode_from_what_collect_actually_returned(
    mode_value, painted, expected_mode, expect_mask
):
    fake = _FakeGenerator()
    studio = _NoLoadStudio(config.AppConfig(), fake)
    handlers, studio = _build_handlers(studio)

    assert studio.model_loaded is False  # модель ещё не грузилась до вызова run()

    paths, _message = handlers["run"](
        _editor_value(painted=painted), "prompt", False, mode_value,
        config.AppConfig().preset, 8, 12, True, -1,
    )

    assert paths == []  # фиктивный генератор всегда возвращает пустой список
    assert studio.model_loaded is False  # ...и не грузит модель, чтобы это узнать

    request = fake.captured
    assert request is not None, "run() обязан дойти до studio.generator.generate"
    assert request.mask_mode == expected_mode
    assert (request.mask is not None) is expect_mask
    # Правка обязана наследовать размеры исходника, а не квадрат по умолчанию —
    # раньше эту роль играло значение "1:1", неотличимое от отсутствия выбора.
    assert request.aspect == aspect_module.FOLLOW_REFERENCE


def test_expand_canvas_enlarges_the_background_and_marks_only_the_new_area():
    handlers, _studio = _build_handlers()

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
    handlers, _studio = _build_handlers()

    update, mode, message = handlers["expand_canvas"](None, ["right"], 0.5)
    assert mode == gen.MASK_MASK
    assert message  # сообщение непустое — просит сначала загрузить изображение


def test_expand_canvas_without_sides_asks_to_choose_one():
    handlers, _studio = _build_handlers()

    update, mode, message = handlers["expand_canvas"](_editor_value(), [], 0.5)
    assert mode == gen.MASK_MASK
    assert message

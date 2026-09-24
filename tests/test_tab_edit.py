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
from fooocus_qwen.ui import painter, tab_edit
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


def _editor_images(size=(64, 64), painted=None):
    """Исходник и слой пометок; при ``painted`` в слое закрашен прямоугольник."""
    background = Image.new("RGBA", size, (10, 20, 30, 255))
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    if painted:
        layer.paste((255, 0, 0, 255), painted)
    return background, layer


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
    mode_value, painted, expected_mode, expect_mask, painter_value
):
    fake = _FakeGenerator()
    studio = _NoLoadStudio(config.AppConfig(), fake)
    handlers, studio = _build_handlers(studio)

    assert studio.model_loaded is False  # модель ещё не грузилась до вызова run()

    paths, _message = handlers["run"](
        painter_value(*_editor_images(painted=painted)), "prompt", False, mode_value,
        config.AppConfig().preset, 8, 12, True, -1, "ru",
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


def test_expand_canvas_enlarges_the_background_and_marks_only_the_new_area(painter_value):
    handlers, _studio = _build_handlers()

    raw, mode, message = handlers["expand_canvas"](painter_value(*_editor_images((64, 64))), ["right"], 0.5, "ru")

    # Ответ — значение кисти, как его разберёт сервер при следующем нажатии:
    # проверяется весь круг «кодирование → разбор», а не промежуточный словарь.
    new_value = painter.decode(raw).as_editor_value()
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

    update, mode, message = handlers["expand_canvas"](None, ["right"], 0.5, "ru")
    assert mode == gen.MASK_MASK
    assert message  # сообщение непустое — просит сначала загрузить изображение


def test_expand_canvas_without_sides_asks_to_choose_one(painter_value):
    handlers, _studio = _build_handlers()

    update, mode, message = handlers["expand_canvas"](painter_value(*_editor_images()), [], 0.5, "ru")
    assert mode == gen.MASK_MASK
    assert message


@pytest.mark.parametrize("prompt", ["", "   "])
def test_an_edit_without_a_prompt_is_not_started(prompt, painter_value):
    """Без инструкции модели правки нечего делать, а после расширения холста
    пустой промт даёт прозрачную новую площадь — просим описать правку."""
    fake = _FakeGenerator()
    handlers, _studio = _build_handlers(_NoLoadStudio(config.AppConfig(), fake))

    paths, message = handlers["run"](
        painter_value(*_editor_images(painted=(8, 8, 24, 24))), prompt, False, gen.MASK_MASK,
        config.AppConfig().preset, 8, 12, True, -1, "ru",
    )

    assert paths == [] and fake.captured is None
    assert "Промт" in message and "прозрачной" in message


# --- картинку результата — в кисть -------------------------------------------


def _saved(tmp_path, name, colour):
    path = tmp_path / name
    Image.new("RGB", (16, 12), colour).save(path)
    return str(path)


def test_the_chosen_result_is_sent_not_always_the_first(tmp_path):
    from fooocus_qwen.ui import tab_edit

    first, second = _saved(tmp_path, "a.png", "red"), _saved(tmp_path, "b.png", "blue")
    produced = [(first, None), (second, None)]
    assert tab_edit.chosen_path(produced, second) == second
    assert tab_edit.chosen_path(produced, None) == first
    assert tab_edit.chosen_path(produced, str(tmp_path / "old.png")) == first, "выбор из прошлого результата устарел"
    assert tab_edit.chosen_path([], second) is None


def test_send_to_editor_fills_the_brush_and_opens_the_edit_tab(tmp_path, painter_value):
    from fooocus_qwen.ui import layout, tab_edit
    from fooocus_qwen.ui.painter import payload

    path = _saved(tmp_path, "r.png", "green")
    value, edit_status, generate_status, tabs = tab_edit.send_to_editor([(path, None)], None, "en")
    assert payload.decode(value).as_editor_value()["background"].size == (16, 12)
    assert "editor" in edit_status.lower()
    assert tabs.selected == layout.TAB_EDIT

    _value, _edit, generate_status, tabs = tab_edit.send_to_editor([], None, "en")
    assert generate_status and "Nothing" in generate_status, "пустой результат — сообщение там, где человек остался"

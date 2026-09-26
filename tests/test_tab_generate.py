"""Вкладка «Генерация»: подписи референсов и обработчики пресетов промтов.

Обработчики генерации и AI-буста здесь не проверяются — они трогают модель и
языковой сервер соответственно, а в этой задаче трогать модель запрещено.
"""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.ui import tab_generate
from fooocus_qwen.ui.i18n import Localizer
from fooocus_qwen.ui.state import Studio

# --- подписи миниатюр референсов ---


def _images(count: int) -> list:
    from PIL import Image

    return [Image.new("RGB", (8, 8)) for _ in range(count)]


def test_no_references_have_no_captions():
    assert tab_generate._captions(_images(0), "ru") == []


def test_single_reference_is_captioned_without_a_tag():
    # Спецификация Qwen запрещает теги при единственном изображении.
    captions = tab_generate._captions(_images(1), "ru")
    assert len(captions) == 1
    assert "<image1>" not in captions[0]


def test_two_references_get_image_tags():
    assert tab_generate._captions(_images(2), "ru") == ["<image1>", "<image2>"]


def test_ten_references_get_ten_distinct_tags():
    captions = tab_generate._captions(_images(10), "ru")
    assert captions == [f"<image{i}>" for i in range(1, 11)]
    assert len(set(captions)) == 10


def test_captions_come_from_the_same_slot_builder_as_the_model_call():
    """Подписи и порядок для модели обязаны считаться одной функцией.

    До этой правки документ архитектуры утверждал, что так и есть, а на деле
    ``_captions`` был вторым, независимым описанием правила тегов, и
    совпадали они только потому, что на вкладке генерации нет исходного
    изображения. Проверка строится на случае, где две формулы расходятся:
    когда в списке есть источник, ``<image1>`` принадлежит ему, а первому
    референсу достаётся ``<image2>``.
    """
    from PIL import Image

    from fooocus_qwen.engine.generator import MASK_NONE, condition_slots

    references = _images(2)
    slots = condition_slots(
        source=Image.new("RGB", (8, 8)), mask_mode=MASK_NONE, references=tuple(references)
    )
    tags_after_a_source = [slot.tag for slot in slots if slot.role == "reference"]
    assert tags_after_a_source == ["<image2>", "<image3>"]

    # А без источника — то, что пользователь видит на вкладке генерации.
    assert tab_generate._captions(references, "ru") == ["<image1>", "<image2>"]


# --- обработчики пресетов промтов вкладки ---


def _build_handlers(monkeypatch, tmp_path):
    """Собирает вкладку и достаёт обработчики save/load/delete по имени.

    Обработчики — замыкания внутри ``build`` и не экспортируются в возвращаемый
    словарь (в нём только компоненты, нужные другим вкладкам). Gradio хранит
    исходную функцию каждого события в ``Dependency.fn``/``Blocks.fns``, поэтому
    их можно найти и вызвать напрямую, не поднимая браузер и не эмулируя клики.
    """
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path)
    cfg = config.AppConfig()
    studio = Studio(cfg)

    with gr.Blocks() as demo:
        localizer = Localizer(cfg.lang)
        tab_generate.build(studio, localizer)

    handlers = {}
    for block_fn in demo.fns.values():
        handlers.setdefault(block_fn.fn.__name__, block_fn.fn)
    return handlers


def test_save_writes_a_preset_file_under_prompt_dir(monkeypatch, tmp_path):
    handlers = _build_handlers(monkeypatch, tmp_path)

    update, message = handlers["save"](
        "Мой пресет", "кот на подоконнике", "", [], config.AppConfig().preset, "1:1", -1, 1.0, "ru"
    )
    assert list(tmp_path.glob("*.json"))
    assert "Мой пресет" in message
    assert update["choices"] == ["Мой пресет"]


def test_load_restores_the_saved_fields(monkeypatch, tmp_path):
    handlers = _build_handlers(monkeypatch, tmp_path)

    handlers["save"]("пресет", "промт", "негатив", ["sai-anime"], "MaxQuality", "16:9", 7, 2.5, "ru")
    (
        prompt_text, negative_text, style_names, quality_name,
        ratio_value, seed_value, cfg_value, message,
    ) = handlers["load"]("пресет", "ru")

    assert prompt_text == "промт"
    assert negative_text == "негатив"
    assert style_names == ["sai-anime"]
    assert quality_name == "MaxQuality"
    assert ratio_value == "16:9"
    assert seed_value == 7
    assert cfg_value == 2.5
    assert "пресет" in message


def test_delete_removes_the_preset_and_reports_when_missing(monkeypatch, tmp_path):
    handlers = _build_handlers(monkeypatch, tmp_path)

    handlers["save"]("временный", "промт", "", [], "LowQuality", "1:1", -1, 1.0, "ru")
    update, message = handlers["delete"]("временный", "ru")
    assert not list(tmp_path.glob("*.json"))
    assert "удалён" in message

    _, missing_message = handlers["delete"]("временный", "ru")
    assert "не найден" in missing_message


def test_save_rejects_an_empty_name(monkeypatch, tmp_path):
    handlers = _build_handlers(monkeypatch, tmp_path)

    handlers["save"]("   ", "промт", "", [], "LowQuality", "1:1", -1, 1.0, "ru")
    assert not list(tmp_path.glob("*.json"))


# --- сетка референсов: десять слотов слева от результата ---------------------
#
# Референс кладут в слот кликом или перетаскиванием. Слоты позиционные и
# бывают с дырами (заполнены 1, 3 и 7), а модель получает сплошной список —
# поэтому подпись слота считается по порядку **заполненных**, той же
# функцией, что строит вход модели: иначе тег в подписи разошёлся бы с тем,
# на что он ссылается в промте.


def _img(colour="red", size=(8, 8)):
    from PIL import Image

    return Image.new("RGB", size, colour)


def _slots(filled: dict[int, object]) -> list:
    """Десять слотов: в указанных позициях картинки, в остальных пусто."""
    return [filled.get(index) for index in range(tab_generate.MAX_REFERENCES)]


def test_the_grid_is_two_rows_of_five():
    assert tab_generate.REFERENCE_ROWS == 2
    assert tab_generate.REFERENCE_COLUMNS == 5
    assert tab_generate.REFERENCE_ROWS * tab_generate.REFERENCE_COLUMNS == tab_generate.MAX_REFERENCES


def test_filled_drops_empty_slots_and_keeps_order():
    first, second, third = _img("red"), _img("green"), _img("blue")
    assert tab_generate.filled(_slots({0: first, 2: second, 6: third})) == [first, second, third]
    assert tab_generate.filled([]) == []
    assert tab_generate.filled(None) == []


def test_labels_follow_the_order_of_filled_slots_not_their_positions():
    """Слоты 1, 3 и 7 — это <image1>, <image2>, <image3>.

    Подпись по позиции («<image7>») ссылалась бы на изображение, которого у
    модели нет: пустые слоты в неё не уходят.
    """
    updates = tab_generate.slot_labels(_slots({0: _img(), 2: _img(), 6: _img()}), "ru")
    assert len(updates) == tab_generate.MAX_REFERENCES
    shown = {index: u["label"] for index, u in enumerate(updates) if u.get("show_label")}
    assert shown == {0: "<image1>", 2: "<image2>", 6: "<image3>"}


def test_empty_slots_carry_no_tag():
    updates = tab_generate.slot_labels(_slots({3: _img(), 4: _img()}), "ru")
    for index in (0, 1, 2, 5, 6, 7, 8, 9):
        assert updates[index]["show_label"] is False, f"пустой слот {index} подписан"


def test_a_single_reference_is_labelled_untagged():
    """При одном условном изображении теги запрещены спецификацией Qwen."""
    updates = tab_generate.slot_labels(_slots({4: _img()}), "en")
    assert updates[4]["show_label"] is True
    assert "<image" not in updates[4]["label"]


def test_labels_never_touch_the_slot_image():
    """Обновление подписи не должно перезаписывать картинку в слоте.

    Иначе каждая подпись заново отправляла бы изображение в браузер, а
    событие изменения слота срабатывало бы от собственного же ответа.
    """
    for update in tab_generate.slot_labels(_slots({0: _img(), 1: _img()}), "ru"):
        assert "value" not in update


def _result(tmp_path, colour="blue", size=(30, 12)):
    from PIL import Image

    path = tmp_path / f"{colour}.png"
    Image.new("RGB", size, colour).save(path)
    return [(str(path), None)]


def test_a_sent_result_goes_into_the_first_empty_slot(tmp_path):
    references = _slots({0: _img("red"), 2: _img("green")})
    state, *slots, status = tab_generate.send_to_references(_result(tmp_path), None, references, "en")

    assert state[1] is not None and state[1].size == (30, 12), "в первый свободный — второй слот"
    assert state[0] is references[0] and state[2] is references[2], "занятые слоты не тронуты"
    assert slots[1]["value"].size == (30, 12), "картинка показана в своём слоте"
    assert all("value" not in slot for index, slot in enumerate(slots) if index != 1)
    assert "3 of 10" in status


def test_sending_into_an_untouched_grid_starts_at_the_first_slot(tmp_path):
    state, *_slots_, _status = tab_generate.send_to_references(_result(tmp_path), None, [], "en")
    assert len(state) == tab_generate.MAX_REFERENCES
    assert state[0] is not None and all(item is None for item in state[1:])


def test_the_eleventh_reference_is_refused(tmp_path):
    references = [_img() for _ in range(tab_generate.MAX_REFERENCES)]
    outputs = tab_generate.send_to_references(_result(tmp_path), None, references, "en")
    assert "10" in outputs[-1] and "no more" in outputs[-1]
    assert all(value == {"__type__": "update"} for value in outputs[:-1]), "ничего не меняется"


def test_nothing_to_send_says_so():
    outputs = tab_generate.send_to_references([], None, [], "en")
    assert "Nothing" in outputs[-1]
    assert all(value == {"__type__": "update"} for value in outputs[:-1])


def test_sending_from_the_edit_tab_first_opens_the_generate_tab_then_sends():
    """Два шага в строгом порядке: сначала вкладка, потом картинка.

    Одним ответом нельзя: новое значение слота на скрытой вкладке Gradio
    6.5.1 не отрисовывает — подпись менялась, а картинки не было
    (см. ``open_generate_tab``). Проверяется сама цепочка событий кнопки
    правки: первым звеном переход, следующим — отправка в слоты.
    """
    from fooocus_qwen.ui import app, layout

    demo = app.build(config.AppConfig())
    opens = [fn for fn in demo.fns.values() if fn.fn is tab_generate.open_generate_tab]
    assert len(opens) == 1
    assert any(isinstance(block, gr.Tabs) for block in opens[0].outputs)
    assert tab_generate.open_generate_tab().selected == layout.TAB_GENERATE

    sends = [
        fn for fn in demo.fns.values()
        if fn.fn is tab_generate.send_to_references and fn.trigger_after == opens[0]._id
    ]
    assert len(sends) == 1, "отправка в слоты должна идти следом за переходом (.then)"
    assert not any(isinstance(block, gr.Tabs) for block in sends[0].outputs), (
        "переключение вкладки в одном ответе со значением слота и есть дефект"
    )


# --- обработчики вкладки, связанные с сеткой -----------------------------------


def _handlers_with_references():
    studio = Studio(config.AppConfig())
    with gr.Blocks() as demo:
        components = tab_generate.build(studio, Localizer("en"))
    return {fn.fn.__name__: fn.fn for fn in demo.fns.values()}, studio, components


def test_a_slot_change_rebuilds_the_list_from_all_slots():
    """Список собирается из всех слотов разом, а не дописывается.

    Слот меняют в любом порядке и чистят крестиком; источник истины — то,
    что сейчас стоит в слотах, а не история изменений.
    """
    handlers, _studio, _components = _handlers_with_references()
    values = _slots({1: _img("red"), 5: _img("blue")})
    state, *updates, status = handlers["slots_changed"](*values, "en")

    assert state == values
    assert len(updates) == tab_generate.MAX_REFERENCES
    assert updates[1]["label"] == "<image1>" and updates[5]["label"] == "<image2>"
    assert "2 of 10" in status


def test_clear_empties_every_slot():
    handlers, _studio, _components = _handlers_with_references()
    state, *updates, _status = handlers["clear_references"]("en")
    assert state == [None] * tab_generate.MAX_REFERENCES
    assert all(update["value"] is None for update in updates[: tab_generate.MAX_REFERENCES])


def test_the_grid_stands_left_of_the_result():
    """Десять слотов в колонке левее холста результата, два ряда по пять."""
    from fooocus_qwen.ui import layout

    studio = Studio(config.AppConfig())
    with gr.Blocks() as demo:
        components = tab_generate.build(studio, Localizer("ru"))

    slots = components["reference_slots"]
    assert len(slots) == tab_generate.MAX_REFERENCES
    assert all(isinstance(slot, gr.Image) for slot in slots)
    assert all(layout.REF_SLOT in (slot.elem_classes or []) for slot in slots)
    assert all(slot.sources == ["upload"] for slot in slots), "клик и перетаскивание, без веб-камеры"

    rows = {id(slot.parent) for slot in slots}
    assert len(rows) == tab_generate.REFERENCE_ROWS, "слоты стоят в двух рядах"
    for row_id in rows:
        assert sum(1 for slot in slots if id(slot.parent) == row_id) == tab_generate.REFERENCE_COLUMNS

    # Колонка сетки и колонка холста — соседи в рабочей строке, сетка первой.
    grid_column = slots[0].parent.parent
    canvas_column = components["result"].parent
    while layout.CANVAS_COL not in (canvas_column.elem_classes or []):
        canvas_column = canvas_column.parent
    work_row = grid_column.parent
    assert canvas_column.parent is work_row
    children = list(work_row.children)
    assert children.index(grid_column) < children.index(canvas_column)
    assert demo is not None


def test_run_gets_only_the_filled_slots():
    """Пустые слоты в модель не уходят, а порядок заполненных сохраняется."""
    from fooocus_qwen.engine import presets
    from fooocus_qwen.engine.generator import GeneratedImage

    seen = []

    class _Engine:
        def generate(self, request, progress=None):
            seen.append(request.references)
            return [GeneratedImage(image=_img(), seed=1, parameters={})]

    handlers, studio, _components = _handlers_with_references()
    studio._generator = _Engine()
    red, blue = _img("red"), _img("blue")
    handlers["run"](
        "cat", "", "", False, _slots({2: red, 7: blue}), presets.DEFAULT, "1:1", 1, [], "",
        1.0, -1, True, 0, "en", progress=lambda *a, **k: None,
    )
    assert seen and list(seen[0]) == [red, blue]

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


# --- сетка референсов: десять ячеек слева от результата ----------------------
#
# Референс кладут в ячейку кликом или перетаскиванием. Ячейки позиционные и
# бывают с дырами (заполнены 1, 3 и 7), а модель получает сплошной список —
# поэтому тег под ячейкой считается по порядку **заполненных**, той же
# функцией, что строит вход модели: иначе тег разошёлся бы с тем, на что он
# ссылается в промте.
#
# Выходы обработчиков сетки: состояние, десять ячеек, десять строк тегов,
# строка состояния.

NBSP = chr(0xA0)
N = 10


def _img(colour="red", size=(8, 8)):
    from PIL import Image

    return Image.new("RGB", size, colour)


def _slots(filled: dict[int, object]) -> list:
    """Десять ячеек: в указанных позициях картинки, в остальных пусто."""
    return [filled.get(index) for index in range(tab_generate.MAX_REFERENCES)]


def _split(outputs):
    """Разбирает выходы обработчика сетки на части."""
    state, rest = outputs[0], list(outputs[1:])
    return state, rest[:N], rest[N: 2 * N], rest[2 * N]


def test_the_grid_is_two_columns_of_five():
    assert tab_generate.REFERENCE_ROWS == 5
    assert tab_generate.REFERENCE_COLUMNS == 2
    assert tab_generate.REFERENCE_ROWS * tab_generate.REFERENCE_COLUMNS == tab_generate.MAX_REFERENCES == N


def test_filled_drops_empty_slots_and_keeps_order():
    first, second, third = _img("red"), _img("green"), _img("blue")
    assert tab_generate.filled(_slots({0: first, 2: second, 6: third})) == [first, second, third]
    assert tab_generate.filled([]) == []
    assert tab_generate.filled(None) == []


def test_tags_follow_the_order_of_filled_cells_not_their_positions():
    """Ячейки 1, 3 и 7 — это <image1>, <image2>, <image3>.

    Подпись по номеру («<image7>») ссылалась бы на изображение, которого у
    модели нет: пустые ячейки в неё не уходят.
    """
    tags = tab_generate.slot_tags(_slots({0: _img(), 2: _img(), 6: _img()}), "ru")
    assert len(tags) == N
    shown = {index: tag["value"] for index, tag in enumerate(tags) if tag["value"] != NBSP}
    assert shown == {0: "`<image1>`", 2: "`<image2>`", 6: "`<image3>`"}


def test_tags_are_wrapped_in_code_so_markdown_keeps_them():
    """Голый <image1> Markdown принял бы за тег разметки и не показал вовсе."""
    tags = tab_generate.slot_tags(_slots({0: _img(), 1: _img()}), "ru")
    assert tags[0]["value"].startswith("`") and tags[0]["value"].endswith("`")


def test_empty_cells_keep_a_blank_line_not_nothing():
    """Пустая строка схлопнулась бы, и ряды с подписью и без поехали бы."""
    tags = tab_generate.slot_tags(_slots({3: _img(), 4: _img()}), "ru")
    for index in (0, 1, 2, 5, 6, 7, 8, 9):
        assert tags[index]["value"] == NBSP, f"пустая ячейка {index}"


def test_a_single_reference_is_marked_untagged_and_short():
    """При одном условном изображении теги запрещены спецификацией Qwen.

    И подпись коротка: она одной строкой под ячейкой шириной в семьдесят
    пикселей.
    """
    tags = tab_generate.slot_tags(_slots({4: _img()}), "en")
    assert "<image" not in tags[4]["value"]
    # «Не нужен», а не «нет»: подпись «no tag» читалась как «ячейка не
    # подписана». Причина — во всплывающей подсказке.
    assert tags[4]["value"].endswith(">no tag needed</span>")
    assert 'title="Only one reference' in tags[4]["value"]


def test_the_status_line_explains_a_single_reference():
    """Строка состояния при единственном референсе говорит, как на него ссылаться."""
    from fooocus_qwen.ui import references

    one = references.with_single_hint("References: 1 of 10", _slots({4: _img()}), "en", None)
    assert "in words" in one
    two = references.with_single_hint("References: 2 of 10", _slots({1: _img(), 4: _img()}), "en", None)
    assert two == "References: 2 of 10"
    editing = references.with_single_hint("References: 1 of 10", _slots({4: _img()}), "en", "mask")
    assert editing == "References: 1 of 10", "на правке у единственного референса тег есть"


def _result(tmp_path, colour="blue", size=(30, 12)):
    from PIL import Image

    path = tmp_path / f"{colour}.png"
    Image.new("RGB", size, colour).save(path)
    return [(str(path), None)]


def test_a_sent_result_goes_into_the_first_empty_cell(tmp_path):
    references = _slots({0: _img("red"), 2: _img("green")})
    state, slots, tags, status = _split(
        tab_generate.send_to_references(_result(tmp_path), None, references, "en")
    )

    assert state[1] is not None and state[1].size == (30, 12), "в первую свободную — вторую"
    assert state[0] is references[0] and state[2] is references[2], "занятые ячейки не тронуты"
    assert slots[1]["value"].size == (30, 12), "картинка показана в своей ячейке"
    assert all("value" not in slot for index, slot in enumerate(slots) if index != 1), (
        "остальные ячейки не пересылаются заново"
    )
    assert [tags[i]["value"] for i in (0, 1, 2)] == ["`<image1>`", "`<image2>`", "`<image3>`"]
    assert "3 of 10" in status


def test_sending_into_an_untouched_grid_starts_at_the_first_cell(tmp_path):
    state, *_rest = tab_generate.send_to_references(_result(tmp_path), None, [], "en")
    assert len(state) == N
    assert state[0] is not None and all(item is None for item in state[1:])


def test_the_eleventh_reference_is_refused(tmp_path):
    references = [_img() for _ in range(N)]
    outputs = tab_generate.send_to_references(_result(tmp_path), None, references, "en")
    assert "10" in outputs[-1] and "no more" in outputs[-1]
    assert all(value == {"__type__": "update"} for value in outputs[:-1]), "ничего не меняется"


def test_nothing_to_send_says_so():
    outputs = tab_generate.send_to_references([], None, [], "en")
    assert "Nothing" in outputs[-1]
    assert all(value == {"__type__": "update"} for value in outputs[:-1])


def test_sending_from_the_edit_tab_first_opens_the_generate_tab_then_sends():
    """Два шага в строгом порядке: сначала вкладка, потом картинка.

    Одним ответом нельзя: новое значение ячейки на скрытой вкладке Gradio
    6.5.1 не отрисовывает — подпись менялась, а картинки не было
    (см. ``open_generate_tab``). Проверяется сама цепочка событий кнопки
    правки: первым звеном переход, следующим — отправка в ячейки.
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
    assert len(sends) == 1, "отправка в ячейки должна идти следом за переходом (.then)"
    assert not any(isinstance(block, gr.Tabs) for block in sends[0].outputs), (
        "переключение вкладки в одном ответе со значением ячейки и есть дефект"
    )


# --- обработчики вкладки, связанные с сеткой -----------------------------------


def _handlers_with_references():
    studio = Studio(config.AppConfig())
    with gr.Blocks() as demo:
        components = tab_generate.build(studio, Localizer("en"))
    return {fn.fn.__name__: fn.fn for fn in demo.fns.values()}, studio, components


def test_a_cell_change_rebuilds_the_list_from_all_cells():
    """Список собирается из всех ячеек разом, а не дописывается.

    Ячейку меняют в любом порядке и чистят крестиком; источник истины — то,
    что сейчас стоит в ячейках, а не история изменений.
    """
    handlers, _studio, _components = _handlers_with_references()
    values = _slots({1: _img("red"), 5: _img("blue")})
    state, slots, tags, status = _split(handlers["slots_changed"](*values, "en"))

    assert state == values
    assert all("value" not in slot for slot in slots), "картинки не пересылаются обратно"
    assert tags[1]["value"] == "`<image1>`" and tags[5]["value"] == "`<image2>`"
    assert "2 of 10" in status


def test_clear_empties_every_cell():
    handlers, _studio, _components = _handlers_with_references()
    state, slots, tags, _status = _split(handlers["clear_references"]("en"))
    assert state == [None] * N
    assert all(slot["value"] is None for slot in slots)
    assert all(tag["value"] == NBSP for tag in tags)


def test_the_grid_stands_left_of_the_result():
    """Десять ячеек в колонке левее поля результата: пять рядов по две ячейки,
    колонка сетки — в одной строке с полем, чтобы быть в его рост."""
    from fooocus_qwen.ui import layout

    studio = Studio(config.AppConfig())
    with gr.Blocks() as demo:
        components = tab_generate.build(studio, Localizer("ru"))

    slots, tags = components["reference_slots"], components["reference_tags"]
    assert len(slots) == N and len(tags) == N
    assert all(isinstance(slot, gr.Image) for slot in slots)
    assert all(layout.REF_SLOT in (slot.elem_classes or []) for slot in slots)
    assert all(slot.sources == ["upload"] for slot in slots), "клик и перетаскивание, без веб-камеры"
    assert all(slot.show_label is False for slot in slots), "тег — строкой под ячейкой, не внутри"

    # Тег — сосед картинки в её ячейке.
    for slot, tag in zip(slots, tags):
        assert tag.parent is slot.parent
        assert layout.REF_CELL in (slot.parent.elem_classes or [])

    rows = {id(slot.parent.parent) for slot in slots}
    assert len(rows) == tab_generate.REFERENCE_ROWS, "ячейки стоят в пяти рядах"
    assert all(layout.REF_ROW in (slot.parent.parent.elem_classes or []) for slot in slots)
    for row_id in rows:
        assert sum(1 for slot in slots if id(slot.parent.parent) == row_id) == tab_generate.REFERENCE_COLUMNS

    # Колонка сетки и поле результата — соседи в одной строке, сетка первой;
    # строка стоит в колонке холста. Равную высоту обеспечивает CSS — её
    # проверяет браузерная проверка (tools/ui_check.py).
    grid_column = slots[0].parent.parent.parent
    assert layout.REFS_COL in (grid_column.elem_classes or [])
    result_row = grid_column.parent
    assert layout.RESULT_ROW in (result_row.elem_classes or [])
    assert components["result"].parent is result_row
    children = list(result_row.children)
    assert children.index(grid_column) < children.index(components["result"])
    assert layout.CANVAS_COL in (result_row.parent.elem_classes or [])
    assert demo is not None


def test_run_gets_only_the_filled_cells():
    """Пустые ячейки в модель не уходят, а порядок заполненных сохраняется."""
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

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


# --- «Отправить в референсы» ---------------------------------------------------


class _Upload:
    """Файл так, как его отдаёт gr.File: объект с полем name."""

    def __init__(self, path):
        self.name = str(path)


def _png(tmp_path, name, colour, size=(20, 10)):
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", size, colour).save(path)
    return path


def test_a_sent_result_is_appended_after_the_uploaded_files(tmp_path):
    uploaded = _Upload(_png(tmp_path, "u.png", "red"))
    result = str(_png(tmp_path, "r.png", "blue", size=(30, 12)))
    sent, images, _gallery, box, _clear, status = tab_generate.send_to_references(
        [(result, None)], None, [uploaded], [], "en"
    )
    assert len(sent) == 1 and len(images) == 2
    assert images[0].size == (20, 10) and images[1].size == (30, 12), "загруженные — первыми, присланные — следом"
    assert box["open"] is True
    assert "2 of 10" in status


def test_uploading_more_files_keeps_what_was_sent(tmp_path):
    """Виджет загрузки отдаёт весь свой набор: присланное не должно из-за этого пропасть."""
    handlers, _engine = _handlers_with_references(tmp_path)
    from PIL import Image

    sent = [Image.new("RGB", (8, 8), "green")]
    images, *_rest = handlers["add_references"]([_Upload(_png(tmp_path, "u.png", "red"))], sent, "en")
    assert len(images) == 2 and images[1].size == (8, 8)


def test_the_eleventh_reference_is_refused(tmp_path):
    from PIL import Image

    result = str(_png(tmp_path, "r.png", "blue"))
    sent = [Image.new("RGB", (8, 8)) for _ in range(tab_generate.MAX_REFERENCES)]
    outputs = tab_generate.send_to_references([(result, None)], None, [], sent, "en")
    assert "10" in outputs[-1] and "no more" in outputs[-1]
    assert all(value == {"__type__": "update"} for value in outputs[:5]), "ничего не меняется"


def test_nothing_to_send_says_so(tmp_path):
    outputs = tab_generate.send_to_references([], None, [], [], "en")
    assert "Nothing" in outputs[-1]


def test_sending_from_the_edit_tab_opens_the_generate_tab(tmp_path):
    from fooocus_qwen.ui import layout

    result = str(_png(tmp_path, "r.png", "blue"))
    *_outputs, tabs = tab_generate.send_reference_from_edit([(result, None)], None, [], [], "en")
    assert tabs.selected == layout.TAB_GENERATE


def _handlers_with_references(tmp_path):
    studio = Studio(config.AppConfig())
    with gr.Blocks() as demo:
        tab_generate.build(studio, Localizer("en"))
    return {fn.fn.__name__: fn.fn for fn in demo.fns.values()}, studio

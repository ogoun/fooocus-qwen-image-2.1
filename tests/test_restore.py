"""Восстановление параметров генерации из метаданных PNG.

Главное обещание задачи: семь восстановленных значений совпадают позиционно
с семью полями вкладки генерации, в объявленном порядке. Ничего в сигнатурах
не бросит исключение, если это когда-нибудь разойдётся — сид молча окажется
в поле стилей, а guidance — в поле сида. Поэтому порядок проверяется отдельным
тестом, построенным на реально собранном интерфейсе, а не на переписанном
вручную списке ожиданий.
"""

from __future__ import annotations

import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.imaging import metadata
from fooocus_qwen.ui import tab_gallery, tab_generate
from fooocus_qwen.ui.i18n import Localizer
from fooocus_qwen.ui.state import Studio

PARAMS = {
    "prompt": "кот в шляпе",
    "prompt_boosted": "a cat wearing a hat",
    "negative_prompt": "blurry",
    "styles": ["sai-anime"],
    "preset": "MaxQuality",
    "seed": 4242,
    "true_cfg_scale": 2.5,
    "width": 2048,
    "height": 2048,
}


def test_fields_come_back_in_the_declared_order():
    prompt, boosted, negative, styles, preset, seed, cfg = tab_gallery.restore_fields(PARAMS)
    assert prompt == "кот в шляпе"
    assert boosted == "a cat wearing a hat"
    assert negative == "blurry"
    assert styles == ["sai-anime"]
    assert preset == "MaxQuality"
    assert seed == 4242
    assert cfg == 2.5


def test_missing_parameters_fall_back_to_defaults():
    prompt, boosted, negative, styles, preset, seed, cfg = tab_gallery.restore_fields({})
    assert prompt == "" and boosted == "" and negative == ""
    assert styles == []
    assert preset == "MiddleQuality"
    assert seed == -1
    assert cfg == 1.0


def test_none_yields_defaults_too():
    assert tab_gallery.restore_fields(None)[0] == ""


def test_round_trip_through_a_real_png(tmp_path):
    path = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "x.png", PARAMS)
    assert tab_gallery.restore_fields(metadata.read_png(path))[0] == "кот в шляпе"


def test_unknown_preset_falls_back():
    # Пресет мог называться иначе в старой сборке.
    assert tab_gallery.restore_fields({"preset": "Ultra"})[4] == "MiddleQuality"


def test_non_numeric_seed_and_cfg_fall_back_to_defaults_instead_of_raising():
    # PNG с нашим ключом чанка, но нечисловым значением — правленный руками
    # файл или чужой инструмент, переиспользовавший ключ, — не должен ронять
    # обработчик кнопки «Восстановить» внутри int()/float().
    prompt, boosted, negative, styles, preset, seed, cfg = tab_gallery.restore_fields(
        {"seed": "не число", "true_cfg_scale": "тоже не число"}
    )
    assert seed == -1
    assert cfg == 1.0


def test_foreign_png_yields_full_defaults_without_raising(tmp_path):
    """Чужой PNG (без нашего чанка) не должен ронять восстановление.

    ``metadata.read_png`` возвращает ``None`` для файла без наших метаданных
    (см. ``test_metadata.test_foreign_png_returns_none``), а
    ``restore_fields(None)`` обязан отдать ПОЛНЫЙ набор дефолтов, а не только
    пустой промт — иначе часть полей вкладки генерации осталась бы заполнена
    значениями от предыдущего восстановления.
    """
    path = tmp_path / "foreign.png"
    Image.new("RGB", (8, 8), "blue").save(path)

    parameters = metadata.read_png(path)
    assert parameters is None

    prompt, boosted, negative, styles, preset, seed, cfg = tab_gallery.restore_fields(parameters)
    assert (prompt, boosted, negative, styles, preset, seed, cfg) == ("", "", "", [], "MiddleQuality", -1, 1.0)


def test_restore_output_order_matches_generate_components(monkeypatch, tmp_path):
    """Контракт порядка: то, что возвращает ``restore_fields``, обязано лечь

    ровно в те компоненты вкладки генерации, которые обработчик кнопки
    «Восстановить» объявляет как выходы, и ровно в этом порядке. Проверка
    построена на реально собранном графе Gradio (``BlockFunction.outputs``),
    а не на переписанном вручную списке: если кто-то вставит поле в середину
    только одного из двух списков (порядок в ``restore_fields`` или порядок
    в ``outputs=[...]`` кнопки), тест обязан упасть.
    """
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    cfg = config.AppConfig()
    studio = Studio(cfg)

    with gr.Blocks() as demo:
        localizer = Localizer(cfg.lang)
        generate_components = tab_generate.build(studio, localizer)
        tab_gallery.build(studio, localizer, generate_components)

    restore_fn = None
    for block_fn in demo.fns.values():
        if block_fn.fn.__name__ == "restore":
            restore_fn = block_fn
            break
    assert restore_fn is not None, "обработчик restore() не найден среди событий вкладки галереи"

    # Из полного списка выходов restore() оставляем только те, что и правда
    # принадлежат вкладке генерации — остаток (например, поле статуса самой
    # галереи) к контракту порядка не относится.
    generate_component_ids = {id(component) for component in generate_components.values()}
    generation_outputs = [output for output in restore_fn.outputs if id(output) in generate_component_ids]

    expected_components = [
        generate_components["prompt"],
        generate_components["boosted"],
        generate_components["negative"],
        generate_components["styles"],
        generate_components["quality"],
        generate_components["seed"],
        generate_components["cfg"],
    ]
    assert generation_outputs == expected_components

    sample = tab_gallery.restore_fields(PARAMS)
    assert len(sample) == len(generation_outputs)

    def _expected_type(component):
        if isinstance(component, gr.Dropdown) and component.multiselect:
            return list
        if isinstance(component, (gr.Number, gr.Slider)):
            return (int, float)
        return str

    # Тип каждого значения обязан подходить принимающему компоненту: если сид
    # (int) окажется на месте стилей (list) или наоборот, тест упадёт здесь,
    # даже если сама Gradio не бросит ни одной ошибки.
    for value, component in zip(sample, generation_outputs):
        expected_type = _expected_type(component)
        assert isinstance(value, expected_type), (
            f"{component.label!r} ожидает {expected_type}, получено {type(value)} ({value!r})"
        )

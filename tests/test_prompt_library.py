"""Сохранённые промты у поля промта: промты по умолчанию и вкладка правки.

Хранилище одно на генерацию, правку и API (``user/prompts``). Правка знает
только текст и не должна стирать настройки, сохранённые под тем же именем на
генерации; промт по умолчанию (Upscale) ставится один раз и не воскресает
после удаления.
"""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from fooocus_qwen import config
from fooocus_qwen.prompting import library
from fooocus_qwen.ui import tab_edit, tab_generate
from fooocus_qwen.ui.i18n import Localizer
from fooocus_qwen.ui.state import Studio


def _handlers(monkeypatch, tmp_path, module):
    monkeypatch.setattr(config, "PROMPT_DIR", tmp_path)
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    with gr.Blocks() as demo:
        components = module.build(Studio(config.AppConfig()), Localizer("ru"))
    found = {}
    for block_fn in demo.fns.values():
        if block_fn.fn is not None:
            found.setdefault(block_fn.fn.__name__, block_fn.fn)
    return components, found


# --- промты по умолчанию ---


def test_shipped_upscale_prompt_is_installed_once(tmp_path):
    user = tmp_path / "prompts"
    assert library.install_defaults(config.DEFAULT_PROMPT_DIR, user) == ["Upscale"]
    text = library.load_prompt("Upscale", user)["prompt"]
    assert text.startswith("LOW-LEVEL PHOTOGRAPHIC RESTORATION ONLY.")
    assert text.endswith("always choose preservation.")
    assert text.count("\n\n") == 6, "семь абзацев, как прислано"
    assert library.list_prompts(user) == ["Upscale"], "метка установки — не пресет"

    assert library.install_defaults(config.DEFAULT_PROMPT_DIR, user) == []


def test_a_deleted_default_does_not_come_back(tmp_path):
    library.install_defaults(config.DEFAULT_PROMPT_DIR, tmp_path)
    assert library.delete_prompt("Upscale", tmp_path)
    assert library.install_defaults(config.DEFAULT_PROMPT_DIR, tmp_path) == []
    assert library.list_prompts(tmp_path) == []


def test_a_default_does_not_overwrite_the_users_own_prompt(tmp_path):
    library.save_prompt("Upscale", {"prompt": "мой вариант"}, tmp_path)
    assert library.install_defaults(config.DEFAULT_PROMPT_DIR, tmp_path) == []
    assert library.load_prompt("Upscale", tmp_path)["prompt"] == "мой вариант"


def test_text_update_keeps_the_rest_of_the_preset(tmp_path):
    library.save_prompt("кот", {"prompt": "старый", "seed": 7, "preset": "MaxQuality"}, tmp_path)
    library.update_prompt_text("кот", "новый", tmp_path)
    assert library.load_prompt("кот", tmp_path) == {"prompt": "новый", "seed": 7, "preset": "MaxQuality"}

    library.update_prompt_text("новое имя", "текст", tmp_path)
    assert library.load_prompt("новое имя", tmp_path) == {"prompt": "текст"}


# --- вкладка правки ---


def test_edit_tab_saves_loads_and_deletes_prompt_text(monkeypatch, tmp_path):
    components, handlers = _handlers(monkeypatch, tmp_path, tab_edit)
    assert "prompts" in components

    _choices, message = handlers["save_prompt_text"]("улучшить", "make it sharper", "ru")
    assert "сохранён" in message
    assert library.load_prompt("улучшить", tmp_path) == {"prompt": "make it sharper"}

    text, message = handlers["load_prompt_text"]("улучшить", "ru")
    assert text == "make it sharper"
    assert "загружен" in message

    *_, message = handlers["delete"]("улучшить", "ru")
    assert "удалён" in message
    assert library.list_prompts(tmp_path) == []


def test_edit_tab_does_not_wipe_a_generation_setup(monkeypatch, tmp_path):
    _components, handlers = _handlers(monkeypatch, tmp_path, tab_edit)
    library.save_prompt("общий", {"prompt": "a fox", "seed": 5, "styles": ["sai-anime"]}, tmp_path)

    handlers["save_prompt_text"]("общий", "a red fox", "ru")
    assert library.load_prompt("общий", tmp_path) == {"prompt": "a red fox", "seed": 5, "styles": ["sai-anime"]}


def test_edit_tab_reports_missing_and_unnamed(monkeypatch, tmp_path):
    _components, handlers = _handlers(monkeypatch, tmp_path, tab_edit)
    assert "Укажите имя" in handlers["save_prompt_text"]("  ", "текст", "ru")[-1]
    assert not list(tmp_path.glob("*.json"))
    assert "не загружен" in handlers["load_prompt_text"]("нет такого", "ru")[-1]
    assert "не выбран" in handlers["load_prompt_text"]("", "ru")[-1]


def test_both_tabs_put_the_library_under_the_prompt(monkeypatch, tmp_path):
    for module in (tab_generate, tab_edit):
        components, _handlers_ = _handlers(monkeypatch, tmp_path, module)
        prompts = components["prompts"]
        assert prompts.saved.label.startswith("Saved prompt —")
        assert prompts.saved.allow_custom_value, "в то же поле вписывают новое имя"  # язык запуска — английский

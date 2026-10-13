"""Сохранённые промты — строка под полем промта на вкладках генерации и правки.

Список, имя, «Сохранить», «Удалить». Хранилище одно на обе вкладки и на API:
``user/prompts/*.json`` (``prompting.library``). Что сохраняет и что
загружает вкладка — решает она сама: генерация — всю настройку (промт,
негатив, стили, качество, пропорции, сид, cfg, LoRA), правка — только текст,
не стирая остального под тем же именем (``library.update_prompt_text``).

Загрузка — по ``input``, а не ``change``: после «Сохранить» список получает
новое значение программно, и ``change`` тут же «загрузил» бы только что
сохранённое, заменив в строке состояния «сохранён» на «загружен».
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import gradio as gr

from .. import config
from ..prompting import library
from .i18n import Localizer, T, pick, say

LOGGER = logging.getLogger(__name__)


@dataclass
class PromptLibrary:
    """Компоненты строки сохранённых промтов."""

    saved: object
    name: object
    save: object
    delete: object


def names() -> list[str]:
    return library.list_prompts(config.PROMPT_DIR)


def build(localizer: Localizer, lang: str, elem_classes: list[str] | None = None) -> PromptLibrary:
    with gr.Row(equal_height=True, elem_classes=elem_classes or []):
        saved = localizer.bind(
            gr.Dropdown(choices=names(), value=None, label=pick("load_prompt", lang), scale=3),
            label=T["load_prompt"],
        )
        name = localizer.bind(
            gr.Textbox(label=pick("preset_name", lang), max_lines=1, scale=3),
            label=T["preset_name"],
        )
        save = localizer.bind(
            gr.Button(pick("save_prompt", lang), scale=1, min_width=120), value=T["save_prompt"],
        )
        delete = localizer.bind(
            gr.Button(pick("delete_prompt", lang), scale=1, min_width=120), value=T["delete_prompt"],
        )
    return PromptLibrary(saved=saved, name=name, save=save, delete=delete)


def refresh():
    """Свежий список: другая вкладка или API могли его изменить."""
    return gr.update(choices=names())


def after_save(name: str, lang: str):
    """Ответ «Сохранить»: список с новым именем выбранным и строка состояния."""
    return gr.update(choices=names(), value=name.strip()), say("preset_saved", lang, name=name.strip())


def load(name: str | None, lang: str) -> tuple[dict | None, str]:
    """Пресет по имени или ``None`` с сообщением, почему его нет.

    Файлы лежат в ``user/prompts/`` и правятся чем угодно: удалённый или
    испорченный мимо приложения пресет — строка состояния, а не тост с
    трейсбеком.
    """
    if not name:
        return None, say("preset_not_selected", lang)
    try:
        return library.load_prompt(name, config.PROMPT_DIR), say("preset_loaded", lang, name=name)
    except (FileNotFoundError, ValueError, OSError) as error:
        LOGGER.warning("Preset '%s' not loaded: %s", name, error)
        return None, say("preset_load_failed", lang, name=name, error=error)


def delete(name: str | None, lang: str):
    """Ответ «Удалить»: (список, имя, строка состояния)."""
    if not name:
        return gr.update(), gr.update(), say("preset_not_selected", lang)
    try:
        removed = library.delete_prompt(name, config.PROMPT_DIR)
    except OSError as error:
        # Файл может быть открыт другой программой или лежать на томе,
        # доступном только на чтение: сказать об этом строкой состояния.
        LOGGER.warning("Preset '%s' not deleted: %s", name, error)
        return gr.update(), gr.update(), say("preset_delete_failed", lang, name=name, error=error)
    message = say("preset_deleted", lang, name=name) if removed else say("preset_missing", lang)
    return gr.update(choices=names(), value=None), "", message

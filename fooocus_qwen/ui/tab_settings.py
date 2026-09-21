"""Вкладка настроек: адрес языковой модели, системные промты, память видеокарты.

Системные промты правятся прямо здесь и перечитываются заново при каждом
обращении к языковой модели (см. ``prompting.boost``), а адрес — при каждом
``Studio.llm_client()``: оболочку не нужно перезапускать, чтобы подхватить
правку.

Отсутствие настроенной языковой модели — обычное состояние свежей установки,
а не ошибка: ``check_connection`` обязана сообщить об этом читаемым текстом,
а не уронить вкладку исключением.
"""

from __future__ import annotations

import gradio as gr

from .. import config
from ..llm import LlmError
from .i18n import Localizer, pick

_PROMPT_FILES: tuple[str, ...] = (
    "system_prompt_t2i.txt",
    "system_prompt_edit.txt",
    "system_prompt_describe.txt",
)

# Дисплей-имена файлов промтов — переводимый текст, а не служебные
# идентификаторы, поэтому у выпадающего списка ниже, как и у mask_mode во
# вкладке редактирования, переводятся и label, и choices.
_PROMPT_KEYS: dict[str, str] = {
    "system_prompt_t2i.txt": "sysprompt_t2i",
    "system_prompt_edit.txt": "sysprompt_edit",
    "system_prompt_describe.txt": "sysprompt_describe",
}


def _read_endpoint() -> str:
    try:
        return config.ENDPOINT_FILE.read_text(encoding="utf-8")
    except OSError:
        return "# Файл не найден. Укажите бэкенд, адрес и token=…\n"


def _read_prompt(name: str) -> str:
    try:
        return (config.SYSTEM_PROMPT_DIR / name).read_text(encoding="utf-8")
    except OSError:
        return f"# Файл {name} не найден. Запустите tools/fetch_system_prompts.py\n"


def build(studio, localizer: Localizer) -> dict:
    lang = studio.config.lang

    with gr.Row():
        with gr.Column():
            endpoint_text = localizer.bind(
                gr.Textbox(
                    label=pick("llm_endpoint", lang),
                    placeholder=pick("llm_endpoint_placeholder", lang),
                    lines=5,
                    value=_read_endpoint(),
                ),
                label=("Адрес языковой модели", "Language model endpoint"),
                placeholder=(
                    "имя бэкенда, адрес хоста:порт, token=…",
                    "backend name, host:port, token=…",
                ),
            )
            with gr.Row():
                save_endpoint = localizer.bind(
                    gr.Button(pick("save", lang)), value=("Сохранить", "Save")
                )
                check = localizer.bind(
                    gr.Button(pick("llm_check", lang), variant="primary"),
                    value=("Проверить связь", "Check connection"),
                )
            endpoint_status = localizer.bind(
                gr.Textbox(label=pick("status", lang), interactive=False, lines=2),
                label=("Состояние", "Status"),
            )

        with gr.Column():
            ru_choices = [(pick(_PROMPT_KEYS[name], "ru"), name) for name in _PROMPT_FILES]
            en_choices = [(pick(_PROMPT_KEYS[name], "en"), name) for name in _PROMPT_FILES]
            chosen_file = localizer.bind(
                gr.Dropdown(
                    choices=ru_choices if lang == "ru" else en_choices,
                    value=_PROMPT_FILES[0],
                    label=pick("sysprompt_select", lang),
                ),
                label=("Системный промт", "System prompt"),
                choices=(ru_choices, en_choices),
            )
            prompt_text = gr.Textbox(lines=18, value=_read_prompt(_PROMPT_FILES[0]), show_label=False)
            save_prompt_button = localizer.bind(
                gr.Button(pick("save_prompt", lang)), value=("Сохранить промт", "Save prompt")
            )
            prompt_status = localizer.bind(
                gr.Textbox(label=pick("status", lang), interactive=False, lines=2),
                label=("Состояние", "Status"),
            )

    with gr.Row():
        memory = localizer.bind(
            gr.Textbox(
                label=pick("memory", lang), interactive=False, lines=2, value="модель ещё не загружена"
            ),
            label=("Память видеокарты", "GPU memory"),
        )
        memory_refresh = localizer.bind(gr.Button(pick("refresh", lang)), value=("Обновить", "Refresh"))

    def store_endpoint(text):
        config.ENDPOINT_FILE.write_text(text, encoding="utf-8")
        return "Адрес сохранён"

    def check_connection():
        # load_endpoint бросает FileNotFoundError (файла нет — обычное дело
        # для свежей установки) или ValueError (в файле нет ни одного хоста),
        # ping() — LlmError (сервер недоступен или ответил ошибкой). Все три
        # случая — читаемое сообщение, а не падение вкладки.
        try:
            client = studio.llm_client()
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            return f"Нет связи: {error}"
        return f"Связь есть. Выбранная модель: {client.model or 'сервер не назвал ни одной'}"

    def load_prompt_file(name):
        return _read_prompt(name)

    def store_prompt_file(name, text):
        (config.SYSTEM_PROMPT_DIR / name).write_text(text, encoding="utf-8")
        return f"Файл {name} сохранён"

    save_endpoint.click(store_endpoint, endpoint_text, endpoint_status)
    check.click(check_connection, None, endpoint_status)
    chosen_file.change(load_prompt_file, chosen_file, prompt_text)
    save_prompt_button.click(store_prompt_file, [chosen_file, prompt_text], prompt_status)
    memory_refresh.click(studio.memory_report, None, memory)

    return {"memory": memory, "endpoint_status": endpoint_status, "prompt_status": prompt_status}

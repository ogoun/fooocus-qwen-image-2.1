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
from . import layout
from .i18n import MESSAGES, Localizer, pick, say

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


def _read_endpoint(lang: str) -> str:
    try:
        return config.ENDPOINT_FILE.read_text(encoding="utf-8")
    except OSError:
        # Заглушка — не содержимое файла, а обращение к пользователю, поэтому
        # она переводится, хотя и оформлена комментарием внутри текстового поля.
        return say("endpoint_file_missing", lang)


def _read_prompt(name: str, lang: str) -> str:
    try:
        return (config.SYSTEM_PROMPT_DIR / name).read_text(encoding="utf-8")
    except OSError:
        return say("prompt_file_missing", lang, name=name)


def build(studio, localizer: Localizer, language=None) -> dict:
    """Собирает вкладку.

    ``language`` — компонент с текущим языком; см. докстринг
    ``tab_generate.build``.
    """
    lang = studio.config.lang
    if language is None:
        language = gr.State(lang)

    with gr.Row(elem_classes=[layout.WORK_ROW]):
        with gr.Column(min_width=layout.SIDE_MIN_WIDTH, elem_classes=[layout.FORM_COL]):
            endpoint_text = localizer.bind(
                gr.Textbox(
                    label=pick("llm_endpoint", lang),
                    placeholder=pick("llm_endpoint_placeholder", lang),
                    lines=5,
                    value=_read_endpoint(lang),
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

        # Системный промт — единственное место вкладки, где ширина идёт в
        # дело: это абзацы текста, который правят руками.
        with gr.Column(min_width=layout.SIDE_MIN_WIDTH, elem_classes=[layout.TEXT_COL]):
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
            prompt_text = gr.Textbox(
                lines=18,
                value=_read_prompt(_PROMPT_FILES[0], lang),
                show_label=False,
                elem_classes=[layout.LONG_TEXT],
            )
            save_prompt_button = localizer.bind(
                gr.Button(pick("save_prompt", lang)), value=("Сохранить промт", "Save prompt")
            )
            prompt_status = localizer.bind(
                gr.Textbox(label=pick("status", lang), interactive=False, lines=2),
                label=("Состояние", "Status"),
            )

        # Память стояла отдельной строкой под обеими колонками и занимала
        # её целиком ради двух строк текста. Третьей колонкой она и читается
        # лучше (рядом с тем, что тоже про состояние), и не отнимает высоту
        # у того, что под ней.
        with gr.Column(min_width=layout.SIDE_MIN_WIDTH, elem_classes=[layout.FORM_COL]):
            memory = localizer.bind(
                gr.Textbox(
                    label=pick("memory", lang),
                    interactive=False,
                    lines=2,
                    value=say("model_not_loaded", lang),
                ),
                label=("Память видеокарты", "GPU memory"),
                # Значение поля тоже переводимо: это заглушка «ещё не
                # загружена», а не результат измерения. Переключение языка
                # сбрасывает поле к ней — для снимка, который обновляют
                # кнопкой, это верное поведение, а вот русская строка вокруг
                # английских подписей — нет.
                value=MESSAGES["model_not_loaded"],
            )
            memory_refresh = localizer.bind(
                gr.Button(pick("refresh", lang)), value=("Обновить", "Refresh")
            )

    def store_endpoint(text, lang):
        config.ENDPOINT_FILE.write_text(text, encoding="utf-8")
        return say("endpoint_saved", lang)

    def check_connection(lang):
        # load_endpoint бросает FileNotFoundError (файла нет — обычное дело
        # для свежей установки) или ValueError (в файле нет ни одного хоста),
        # ping() — LlmError (сервер недоступен или ответил ошибкой). Все три
        # случая — читаемое сообщение, а не падение вкладки.
        try:
            client = studio.llm_client()
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            return say("llm_no_connection", lang, error=error)
        return say(
            "llm_connected", lang, model=client.model or say("llm_no_model_named", lang)
        )

    def load_prompt_file(name, lang):
        return _read_prompt(name, lang)

    def store_prompt_file(name, text, lang):
        (config.SYSTEM_PROMPT_DIR / name).write_text(text, encoding="utf-8")
        return say("prompt_file_saved", lang, name=name)

    def memory_report(lang):
        return studio.memory_report(lang)

    save_endpoint.click(store_endpoint, [endpoint_text, language], endpoint_status)
    check.click(check_connection, language, endpoint_status)
    chosen_file.change(load_prompt_file, [chosen_file, language], prompt_text)
    save_prompt_button.click(
        store_prompt_file, [chosen_file, prompt_text, language], prompt_status
    )
    memory_refresh.click(memory_report, language, memory)

    return {"memory": memory, "endpoint_status": endpoint_status, "prompt_status": prompt_status}

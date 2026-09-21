"""Сборка интерфейса и запуск сервера."""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from .. import config
from . import tab_generate
from .i18n import LANGUAGES, Localizer, pick
from .state import Studio

LOGGER = logging.getLogger(__name__)


def build(cfg: config.AppConfig) -> gr.Blocks:
    studio = Studio(cfg)
    localizer = Localizer(cfg.lang)

    with gr.Blocks(title=pick("app_title", cfg.lang), analytics_enabled=False) as demo:
        with gr.Row():
            localizer.bind(
                gr.Markdown(f"## {pick('app_title', cfg.lang)}"),
                value=("## Qwen-Image-2.1 — студия", "## Qwen-Image-2.1 Studio"),
            )
            language = gr.Dropdown(
                choices=list(LANGUAGES), value=cfg.lang, label="RU / EN", scale=0, min_width=120
            )

        with gr.Tabs():
            # Заголовок вкладки — такая же переводимая подпись, как и всё
            # остальное: без регистрации в localizer он застыл бы на языке
            # запуска и не откликался бы на переключатель RU/EN.
            generate_tab = localizer.bind(
                gr.Tab(pick("tab_generate", cfg.lang)),
                label=("Генерация", "Generate"),
            )
            with generate_tab:
                tab_generate.build(studio, localizer)

        # Вкладки редактирования, галереи и настроек добавляются в задачах 13 и 14.

        language.change(localizer.updates, language, localizer.components, queue=False)

    return demo


def launch(cfg: config.AppConfig) -> None:
    demo = build(cfg)
    demo.queue(default_concurrency_limit=1)
    # В Gradio 6 параметр css переехал из конструктора Blocks в launch() — передача
    # его в конструктор всё ещё работает, но с предупреждением об устаревании.
    css = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
    LOGGER.info("Интерфейс на http://%s:%s", cfg.host, cfg.port)
    # В Gradio 6.5.1 параметра show_api больше нет ни в Blocks, ни в launch() —
    # он был убран выше по течению. Ссылку «Use via API» и так прячет footer
    # в style.css, так что отдельно отключать нечего.
    demo.launch(server_name=cfg.host, server_port=cfg.port, inbrowser=False, css=css)

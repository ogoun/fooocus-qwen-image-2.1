"""Вкладка галереи: история генераций и возврат к их параметрам.

Параметры генерации живут внутри самих PNG (см. ``imaging.metadata``), поэтому
история переживает и перезапуск оболочки, и перенос файлов результатов на
другую машину — отдельная база для этого не нужна: любой наш PNG сам себе
запись в журнале.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import gradio as gr

from .. import config
from ..engine import presets
from ..imaging import metadata
from ..storage import gallery
from .i18n import Localizer, pick

LOGGER = logging.getLogger(__name__)

# Порядок обязан совпадать с порядком выходов кнопки «Восстановить» в build():
# восстановление читает по этому же порядку значения из словаря параметров, а
# build() раскладывает их по тем же семи полям вкладки генерации. Оба места
# проверяет test_restore.test_restore_output_order_matches_generate_components,
# построенный на реально собранном графе Gradio, а не на переписанном вручную
# списке ожиданий — так что рассинхронизация здесь не пройдёт тесты молча.


def restore_fields(parameters: dict | None) -> tuple:
    """Значения полей вкладки генерации в фиксированном порядке.

    Порядок: промт, переписанный промт, негатив, стили, пресет, сид, guidance.
    Отсутствующий или незнакомый пресет (например, из старой сборки) тихо
    заменяется дефолтным — таблица пресетов меняется быстрее, чем метаданные
    в уже сохранённых PNG.
    """
    data = parameters or {}
    preset = data.get("preset", presets.DEFAULT)
    if preset not in presets.PRESETS:
        preset = presets.DEFAULT

    return (
        data.get("prompt", ""),
        data.get("prompt_boosted", ""),
        data.get("negative_prompt", ""),
        list(data.get("styles", [])),
        preset,
        int(data.get("seed", -1)),
        float(data.get("true_cfg_scale", 1.0)),
    )


def _open_folder(path: Path) -> None:
    """Открывает каталог в файловом менеджере системы."""
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def build(studio, localizer: Localizer, generate_components: dict) -> dict:
    lang = studio.config.lang

    with gr.Row():
        with gr.Column(scale=3):
            history = localizer.bind(
                gr.Gallery(
                    label=pick("tab_gallery", lang),
                    columns=6,
                    height=560,
                    object_fit="contain",
                    value=[str(path) for path in gallery.recent(config.OUTPUT_DIR)],
                ),
                label=("Галерея", "Gallery"),
            )
        with gr.Column(scale=1):
            refresh = localizer.bind(gr.Button(pick("refresh", lang)), value=("Обновить", "Refresh"))
            open_button = localizer.bind(
                gr.Button(pick("open_folder", lang)), value=("Открыть папку", "Open folder")
            )
            dropped = localizer.bind(
                gr.File(label=pick("restore_params", lang), file_types=[".png"]),
                label=("Восстановить параметры из PNG", "Restore parameters from PNG"),
            )
            restore_button = localizer.bind(
                gr.Button(pick("restore_params", lang), variant="primary"),
                value=("Восстановить параметры из PNG", "Restore parameters from PNG"),
            )
            details = localizer.bind(
                gr.JSON(label=pick("status", lang)), label=("Состояние", "Status")
            )

    # Хранит путь последнего выбранного в галерее файла — кнопка «Восстановить»
    # должна знать источник, даже если пользователь ничего не перетаскивал.
    selected = gr.State(None)

    def refresh_history():
        return [str(path) for path in gallery.recent(config.OUTPUT_DIR)]

    def on_select(event: gr.SelectData):
        value = event.value
        path = Path(value["image"]["path"]) if isinstance(value, dict) else Path(value)
        parameters = metadata.read_png(path)
        return str(path), (parameters or {"сообщение": "в этом PNG нет наших параметров"})

    def open_outputs():
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _open_folder(config.OUTPUT_DIR)
        return {"открыт каталог": str(config.OUTPUT_DIR)}

    def restore(path, uploaded):
        # Перетащенный файл важнее выбора в галерее: пользователь явно принёс
        # новый PNG, значит, речь уже не о том, что было выбрано раньше.
        source = Path(uploaded) if uploaded else (Path(path) if path else None)
        if source is None:
            return (gr.update(),) * 7 + ({"сообщение": "выберите изображение в галерее или перетащите PNG"},)

        parameters = metadata.read_png(source)
        return restore_fields(parameters) + (parameters or {"сообщение": "параметры не найдены"},)

    refresh.click(refresh_history, None, history)
    open_button.click(open_outputs, None, details)
    history.select(on_select, None, [selected, details])
    restore_button.click(
        restore,
        [selected, dropped],
        [
            generate_components["prompt"],
            generate_components["boosted"],
            generate_components["negative"],
            generate_components["styles"],
            generate_components["quality"],
            generate_components["seed"],
            generate_components["cfg"],
            details,
        ],
    )

    return {"history": history, "details": details, "selected": selected}

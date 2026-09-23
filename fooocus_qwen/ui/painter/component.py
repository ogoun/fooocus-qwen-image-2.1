"""Компонент Gradio вокруг кисти маски.

Наследник ``gr.HTML``: Gradio 6 позволяет задать компоненту шаблон, стили и
скрипт, и этого хватает на полноценный редактор без сборки фронтенда. Все
механизмы, на которые здесь опирается конструкция, проверены прототипом на
Gradio 6.5.1 до того, как на них что-то построено:

* разметка в теневом корне переживает перерисовку шаблона — Gradio
  переписывает светлый DOM компонента на каждое изменение свойств;
* асинхронный ``js`` у события подменяет значение компонента перед отправкой
  (так кнопки забирают свежую маску, не дожидаясь отложенной синхронизации);
* значение и собственные свойства (язык, режим области), выставленные из
  Python, доходят до скрипта — при условии, что конструктор принимает их
  явными параметрами: Gradio при обновлении пересоздаёт компонент, и
  свойство, пришедшее через ``**kwargs`` вторым путём, роняет его с
  «got multiple values for keyword argument».
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr

HERE = Path(__file__).parent

# Канал «сервер → кисть»: скрипт следит за этим узлом и берёт свежие значения
# из ``props``. Сам узел не виден — светлый DOM закрыт теневым корнем.
TEMPLATE = (
    '<script type="application/json" data-qs-painter '
    'data-lang="${lang}" data-region="${region}">${value}</script>'
)

# Скрипт кнопки, которой нужна маска: дожидается загрузки исходника и
# свежего слоя и подставляет значение кисти первым аргументом события.
# Первым — договорённость: у всех событий вкладки правки кисть стоит в
# списке входов первой, это проверяется тестом.
FLUSH_JS = """
async (...args) => {
    const painter = (window.__qsPainters || {})[%s];
    if (painter) args[0] = await painter.flush();
    return args;
}
"""


@lru_cache(maxsize=1)
def client_script() -> str:
    """Скрипт кисти со встроенными стилями — один раз на процесс."""
    css = (HERE / "painter.css").read_text(encoding="utf-8")
    script = (HERE / "painter.js").read_text(encoding="utf-8")
    return f"const QP_CSS = {json.dumps(css)};\n{script}"


class MaskPainter(gr.HTML):
    """Холст с кистью маски и пометок.

    Значение — строка JSON (см. ``payload``); сервер разбирает её через
    ``payload.decode`` и получает словарь той же формы, что отдавал
    ``gr.ImageEditor``.
    """

    def __init__(
        self,
        value: str | None = None,
        *,
        lang: str = "ru",
        region: str = "mask",
        labels: dict[str, list[str]] | None = None,
        palette: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # Gradio пересоздаёт компонент при обновлении свойств и передаёт
        # шаблон и скрипт обратно; свои значения всё равно главнее.
        for key in ("html_template", "js_on_load", "css_template", "apply_default_css"):
            kwargs.pop(key, None)
        super().__init__(
            value=value or "",
            html_template=TEMPLATE,
            css_template="height: 100%;",
            js_on_load=client_script(),
            apply_default_css=False,
            lang=lang,
            region=region,
            labels=labels or {},
            palette=list(palette or ["#ff0000"]),
            **kwargs,
        )

    def api_info(self) -> dict[str, Any]:
        return {"type": "string", "description": "JSON-значение кисти маски"}


def flush_js(elem_id: str) -> str:
    """Скрипт события, которому нужна свежая маска кисти ``elem_id``."""
    return FLUSH_JS % json.dumps(elem_id)

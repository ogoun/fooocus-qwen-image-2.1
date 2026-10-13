"""Картинки любого формата в полях интерфейса: перевод в PNG до Gradio.

Перетащенный HEIC не доходил до ячейки референса молча: Chrome под Windows
не знает MIME-типа ``.heic`` (``file.type`` пуст), а поле изображения Gradio
пропускает при перетаскивании только ``image/*``. Дойди файл — браузер бы его
не показал, а Pillow не открыл. То же с RAW, TIFF, PSD, JPEG XL, EXR.

Поэтому перевод — на входе, одной точкой для всех полей:

* скрипт страницы (``image_import.js``) перехватывает в фазе захвата
  ``drop`` и ``change`` полей выбора файла, отправляет такие файлы сюда и
  повторяет событие уже с PNG — поле Gradio, кисть правки и окно поз видят
  обычную картинку, ничего о форматах не зная; SVG растеризует сам браузер;
* маршрут ``CONVERT_PATH`` переводит файл (``imaging.formats``) и отдаёт PNG.

Маршрут живёт в приложении Gradio и закрыт тем же входом, что и остальное
(``--username``/``--password``): проверка повторяет ``get_current_user``
Gradio — кука сессии, выданная его же страницей входа.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import unquote

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.routing import Route

from ..imaging import formats
from .i18n import T

LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).parent

CONVERT_PATH = "/fooocus/convert-image"
# Потолок файла: RAW среднего формата — до 200 МБ, TIFF со слоями — бывает
# больше, но такой картинке в референсе делать нечего.
MAX_BYTES = 300 * 1024 * 1024
TEXT_KEYS = ("import_converting", "import_failed")


def _signed_in(app, request: Request) -> bool:
    """Та же проверка, что ``login_check`` Gradio: вход не нужен или кука сессии верна."""
    if getattr(app, "auth", None) is None and getattr(app, "auth_dependency", None) is None:
        return True
    if getattr(app, "auth_dependency", None) is not None:
        return app.auth_dependency(request) is not None
    cookie_id = getattr(app, "cookie_id", "")
    token = request.cookies.get(f"access-token-{cookie_id}") or request.cookies.get(
        f"access-token-unsecure-{cookie_id}"
    )
    return getattr(app, "tokens", {}).get(token) is not None


def install(app) -> None:
    """Добавляет маршрут перевода в приложение Gradio; повторно — ничего не делает."""
    if any(getattr(route, "path", None) == CONVERT_PATH for route in app.router.routes):
        return

    async def convert(request: Request) -> Response:
        if not _signed_in(app, request):
            return JSONResponse({"error": "Not authenticated"}, status_code=401)
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BYTES:
            return JSONResponse({"error": "the file is too large"}, status_code=413)
        data = await request.body()
        if len(data) > MAX_BYTES:
            return JSONResponse({"error": "the file is too large"}, status_code=413)
        name = unquote(request.headers.get("x-file-name", ""))
        try:
            png = await run_in_threadpool(lambda: formats.to_png(formats.decode(data, name)))
        except formats.FormatError as error:
            LOGGER.info("Image import rejected %s: %s", name or "a file", error)
            return JSONResponse({"error": str(error)}, status_code=415)
        LOGGER.info("Imported %s as PNG (%d KB)", name or "an image", len(png) // 1024)
        return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})

    # В начало списка: у Gradio есть маршруты с шаблонами пути, и порядок
    # решает, кто ответит первым.
    app.router.routes.insert(0, Route(CONVERT_PATH, convert, methods=["POST"]))


def script() -> str:
    """Скрипт страницы: списки форматов и подписи подставлены из Python — источник один."""
    text = {key: list(T[key]) for key in TEXT_KEYS}
    return (
        (HERE / "image_import.js").read_text(encoding="utf-8")
        .replace("__NATIVE__", json.dumps(list(formats.NATIVE_EXTENSIONS)))
        .replace("__CONVERTED__", json.dumps(list(formats.CONVERTED_EXTENSIONS)))
        .replace("__PATH__", json.dumps(CONVERT_PATH.lstrip("/")))
        .replace("__TEXT__", json.dumps(text, ensure_ascii=False))
    )

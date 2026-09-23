"""Значение кисти маски: разбор того, что прислал браузер, и сборка того, что ему отдать.

Значение — строка JSON. Картинки в ней не лежат, в ней лежат ссылки:

* ``source`` — путь к исходному изображению на сервере. Браузер загружает
  файл штатным эндпоинтом Gradio (``/gradio_api/upload``) один раз, при
  выборе, и дальше пересылает только путь. Гонять многомегабайтную картинку
  в base64 с каждым нажатием «Применить» незачем.
* ``layer`` — слой пометок размером с исходник: PNG в data URL от браузера
  (он маленький — почти весь прозрачный) или путь к файлу, если слой
  подготовил сервер (расширение холста помечает новую площадь).

Путь пришёл из браузера, то есть от кого угодно, кто достучался до порта.
Поэтому принимается он только внутри каталога загрузок Gradio: иначе поле
``source`` превратилось бы в чтение произвольного файла с диска. Это не
паранойя — ту же границу держит сам Gradio при раздаче файлов, и опыт с
прототипом это показал: чужой файл по той же дороге получает 403.

На выходе — словарь той же формы, что отдавал ``gr.ImageEditor``
(``background``, ``layers``, ``composite``), поэтому разбор маски, склейка и
режимы области ниже по течению не заметили замены редактора.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

PROTOCOL = 1
ORIGIN_SERVER = "server"
ORIGIN_CLIENT = "client"

_DATA_URL_PREFIX = "data:image/png;base64,"

# Потолок для слоя в data URL. Слой 4K в RGBA без сжатия — 33 МБ, в PNG с
# почти сплошной прозрачностью он на порядки меньше; сто мегабайт текста —
# это уже не слой, а попытка забить память сервера.
MAX_LAYER_CHARS = 100 * 1024 * 1024

# Подкаталог для файлов, которые готовит сервер. Внутри каталога загрузок,
# потому что только оттуда Gradio согласится их раздать браузеру.
SERVER_SUBDIR = "fooocus-qwen-painter"


class PayloadError(ValueError):
    """Значение кисти не удалось разобрать или оно указывает не туда."""


@dataclass(frozen=True)
class Canvas:
    """Исходник и слой пометок, уже прочитанные с диска."""

    background: Image.Image | None = None
    layer: Image.Image | None = None

    def as_editor_value(self) -> dict[str, Any] | None:
        """Словарь в форме значения ``gr.ImageEditor`` — или ``None``, если картинки нет."""
        if self.background is None:
            return None
        return {"background": self.background, "layers": [self.layer] if self.layer else [], "composite": None}


def upload_root() -> Path:
    """Каталог загрузок Gradio — единственное место, откуда принимаются пути."""
    from gradio import utils

    return Path(utils.get_upload_folder())


def decode(raw: str | None, roots: Iterable[Path] | None = None) -> Canvas:
    """Разбирает значение кисти. Пустое значение — пустой холст, а не ошибка."""
    if raw is None or not str(raw).strip():
        return Canvas()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise PayloadError(f"значение кисти не JSON: {error.msg}") from error
    if not isinstance(data, dict):
        raise PayloadError("значение кисти должно быть объектом JSON")
    if data.get("v") != PROTOCOL:
        raise PayloadError(f"неизвестная версия значения кисти: {data.get('v')!r}")

    allowed = [Path(root).resolve() for root in (roots if roots is not None else [upload_root()])]
    source = data.get("source")
    if not source:
        return Canvas()

    # Поворот из EXIF применяется так же, как его применяет браузер
    # (createImageBitmap с imageOrientation: 'from-image'). Иначе снимок с
    # телефона, повёрнутый только метаданными, на экране стоял бы прямо, а
    # на сервере — боком, и маска легла бы не туда, где её рисовали.
    background = ImageOps.exif_transpose(_open(_inside(str(source), allowed))).convert("RGBA")
    layer = _read_layer(data.get("layer"), allowed, background.size)
    return Canvas(background=background, layer=layer)


def encode(
    background: Image.Image,
    layer: Image.Image | None = None,
    directory: Path | None = None,
) -> str:
    """Готовит значение, которым сервер загружает картинку в кисть.

    Картинки сохраняются в PNG под каталогом загрузок — оттуда их отдаст
    браузеру сам Gradio. Имя каталога случайное: ревизия заодно служит
    браузеру знаком «это новая загрузка, а не эхо моего же значения».
    """
    revision = uuid.uuid4().hex
    target = (directory or upload_root() / SERVER_SUBDIR) / revision
    target.mkdir(parents=True, exist_ok=True)

    source_path = target / "source.png"
    background.save(source_path, format="PNG")
    layer_path = None
    if layer is not None:
        layer_path = target / "layer.png"
        layer.convert("RGBA").resize(background.size, Image.NEAREST).save(layer_path, format="PNG")

    return json.dumps(
        {
            "v": PROTOCOL,
            "origin": ORIGIN_SERVER,
            "rev": revision,
            "source": str(source_path),
            "layer": str(layer_path) if layer_path else None,
            "width": background.width,
            "height": background.height,
        },
        ensure_ascii=False,
    # Значение ложится в шаблон внутри <script>: «<» экранируется, чтобы
    # никакой путь не закрыл тег раньше времени. JSON от этого не меняется.
    ).replace("<", "\\u003c")


def _inside(path: str, roots: list[Path]) -> Path:
    """Путь, если он внутри разрешённых каталогов; иначе ошибка."""
    try:
        resolved = Path(path).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PayloadError(f"файл кисти не найден: {Path(path).name}") from error
    for root in roots:
        if resolved == root or root in resolved.parents:
            return resolved
    raise PayloadError("путь вне каталога загрузок отклонён")


def _open(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.load()
            return image.copy()
    except (UnidentifiedImageError, OSError) as error:
        raise PayloadError(f"не удалось прочитать изображение {path.name}: {error}") from error


def _read_layer(value: Any, roots: list[Path], size: tuple[int, int]) -> Image.Image | None:
    """Слой из data URL или из файла; размер приводится к исходнику.

    Размер обязан совпадать с исходником, и браузер присылает именно такой.
    Приведение — страховка, а не норма: без неё рассинхрон, откуда бы он ни
    взялся, превратился бы в маску, сдвинутую относительно картинки.
    Ближайший сосед — потому что маска бинарная по смыслу и сглаживанию
    взяться неоткуда.
    """
    if not value:
        return None
    if not isinstance(value, str):
        raise PayloadError("слой кисти должен быть строкой")

    if value.startswith("data:"):
        if not value.startswith(_DATA_URL_PREFIX):
            raise PayloadError("слой кисти принимается только в PNG")
        if len(value) > MAX_LAYER_CHARS:
            raise PayloadError("слой кисти слишком велик")
        try:
            raw = base64.b64decode(value[len(_DATA_URL_PREFIX):], validate=True)
        except (binascii.Error, ValueError) as error:
            raise PayloadError("слой кисти повреждён: base64 не читается") from error
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                layer = image.convert("RGBA")
        except (UnidentifiedImageError, OSError) as error:
            raise PayloadError(f"слой кисти не читается как PNG: {error}") from error
    else:
        layer = _open(_inside(value, roots)).convert("RGBA")

    if layer.size != size:
        layer = layer.resize(size, Image.NEAREST)
    return layer

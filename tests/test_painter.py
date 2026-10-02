"""Кисть маски: значение, его разбор и проводка во вкладке правки.

Своя кисть заменила ``gr.ImageEditor`` из-за запаздывания: у того стоимость
движения мыши растёт с длиной мазка (замер — с 16.7 до 44 мс за мазок на
кадре 3840×2160). Замена обязана была ничего не сломать ниже по течению,
поэтому значение кисти разбирается в словарь той же формы, что отдавал
редактор, — это здесь и проверяется.

Отдельно проверяется граница безопасности. Значение приходит из браузера, то
есть от любого, кто достучался до порта, а в нём — путь к файлу. Без проверки
поле ``source`` стало бы чтением произвольного файла с диска сервера.
"""

from __future__ import annotations

import base64
import io
import json
import re
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.ui import i18n, tab_edit  # noqa: E402
from fooocus_qwen.ui.painter import component, payload  # noqa: E402


def _png_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _client_value(source_path, layer=None, **extra) -> str:
    """Значение в том виде, в каком его шлёт браузер."""
    value = {"v": 1, "origin": "client", "rev": None, "source": str(source_path),
             "layer": layer, "width": 0, "height": 0}
    value.update(extra)
    return json.dumps(value)


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    root = tmp_path / "uploads"
    root.mkdir()
    monkeypatch.setattr(payload, "upload_root", lambda: root)
    return root


# --- разбор и сборка значения --------------------------------------------------

def test_an_empty_value_is_an_empty_canvas_not_an_error(uploads):
    for raw in (None, "", "   "):
        canvas = payload.decode(raw)
        assert canvas.background is None and canvas.layer is None
        assert canvas.as_editor_value() is None


def test_what_the_server_encodes_it_reads_back_unchanged(uploads):
    background = Image.new("RGBA", (40, 30), (10, 20, 30, 255))
    layer = Image.new("RGBA", (40, 30), (0, 0, 0, 0))
    layer.paste((255, 0, 0, 255), (5, 5, 15, 15))

    value = payload.decode(payload.encode(background, layer)).as_editor_value()

    assert value["background"].size == (40, 30)
    assert np.array_equal(np.asarray(value["background"]), np.asarray(background))
    assert np.array_equal(np.asarray(value["layers"][0]), np.asarray(layer))
    assert value["composite"] is None


def test_the_server_files_land_inside_the_upload_folder(uploads):
    """Иначе Gradio откажется раздать их браузеру (403)."""
    data = json.loads(payload.encode(Image.new("RGB", (4, 4)), Image.new("RGBA", (4, 4))))
    for key in ("source", "layer"):
        assert (uploads / payload.SERVER_SUBDIR) in payload.Path(data[key]).parents
    assert data["origin"] == payload.ORIGIN_SERVER and data["rev"]


def test_every_server_load_has_a_new_revision(uploads):
    """По ревизии браузер отличает новую загрузку от эха своего же значения."""
    image = Image.new("RGB", (4, 4))
    first, second = (json.loads(payload.encode(image))["rev"] for _ in range(2))
    assert first != second


def test_a_client_layer_arrives_as_a_png_data_url(uploads):
    source = uploads / "photo.png"
    Image.new("RGB", (32, 32), "white").save(source)
    layer = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    layer.paste((255, 45, 85, 255), (0, 0, 8, 8))

    canvas = payload.decode(_client_value(source, _png_data_url(layer)))

    assert canvas.layer is not None
    assert np.asarray(canvas.layer)[..., 3][4, 4] == 255
    assert np.asarray(canvas.layer)[..., 3][20, 20] == 0


def test_a_layer_of_the_wrong_size_is_brought_to_the_source_size(uploads):
    """Страховка: сдвинутая относительно картинки маска хуже любой ошибки."""
    source = uploads / "photo.png"
    Image.new("RGB", (64, 32)).save(source)
    small = Image.new("RGBA", (32, 16), (255, 0, 0, 255))

    canvas = payload.decode(_client_value(source, _png_data_url(small)))

    assert canvas.layer.size == (64, 32)
    # Ближайший сосед: маска остаётся бинарной, промежуточных значений нет.
    assert set(np.unique(np.asarray(canvas.layer)[..., 3])) <= {0, 255}


def test_exif_rotation_is_applied_like_the_browser_does(uploads):
    """Снимок, повёрнутый только метаданными, браузер показывает прямо.

    Сервер обязан прочитать его так же, иначе маска ляжет боком.
    """
    source = uploads / "phone.jpg"
    image = Image.new("RGB", (40, 20), "white")
    exif = Image.Exif()
    exif[0x0112] = 6  # повернуть на 90° по часовой
    image.save(source, exif=exif.tobytes())

    canvas = payload.decode(_client_value(source))

    assert canvas.background.size == (20, 40)


# --- граница безопасности ------------------------------------------------------

def test_a_path_outside_the_upload_folder_is_refused(uploads, tmp_path):
    outside = tmp_path / "secret.png"
    Image.new("RGB", (4, 4)).save(outside)
    with pytest.raises(payload.PayloadError, match="outside the uploads directory"):
        payload.decode(_client_value(outside))


def test_climbing_out_through_dot_dot_is_refused(uploads, tmp_path):
    outside = tmp_path / "secret.png"
    Image.new("RGB", (4, 4)).save(outside)
    sneaky = uploads / ".." / "secret.png"
    with pytest.raises(payload.PayloadError):
        payload.decode(_client_value(sneaky))


def test_a_layer_path_outside_the_upload_folder_is_refused(uploads, tmp_path):
    source = uploads / "photo.png"
    Image.new("RGB", (4, 4)).save(source)
    outside = tmp_path / "layer.png"
    Image.new("RGBA", (4, 4)).save(outside)
    with pytest.raises(payload.PayloadError, match="outside the uploads directory"):
        payload.decode(_client_value(source, str(outside)))


def test_a_missing_file_is_a_readable_error(uploads):
    with pytest.raises(payload.PayloadError, match="not found"):
        payload.decode(_client_value(uploads / "нет.png"))


@pytest.mark.parametrize("raw, fragment", [
    ("не json", "not JSON"),
    ("[1, 2]", "JSON object"),
    ('{"v": 99, "source": "x"}', "version"),
])
def test_a_malformed_value_is_a_readable_error(uploads, raw, fragment):
    with pytest.raises(payload.PayloadError, match=fragment):
        payload.decode(raw)


@pytest.mark.parametrize("layer, fragment", [
    ("data:image/jpeg;base64,AAAA", "only as PNG"),
    ("data:image/png;base64,@@@@", "base64"),
    ("data:image/png;base64," + base64.b64encode(b"not a png").decode(), "not a readable PNG"),
    (12345, "must be a string"),
])
def test_a_broken_layer_is_a_readable_error(uploads, layer, fragment):
    source = uploads / "photo.png"
    Image.new("RGB", (4, 4)).save(source)
    with pytest.raises(payload.PayloadError, match=fragment):
        payload.decode(_client_value(source, layer))


def test_an_oversized_layer_is_refused_before_decoding(uploads, monkeypatch):
    source = uploads / "photo.png"
    Image.new("RGB", (4, 4)).save(source)
    monkeypatch.setattr(payload, "MAX_LAYER_CHARS", 64)
    with pytest.raises(payload.PayloadError, match="too large"):
        payload.decode(_client_value(source, _png_data_url(Image.new("RGBA", (64, 64)))))


# --- проводка во вкладке правки ------------------------------------------------

def _edit_blocks():
    from fooocus_qwen.ui.i18n import Localizer
    from fooocus_qwen.ui.state import Studio

    studio = Studio(config.AppConfig())
    with gr.Blocks() as demo:
        tab_edit.build(studio, Localizer("ru"))
    return demo


def _event(demo, name):
    for block_fn in demo.fns.values():
        if getattr(block_fn.fn, "__name__", None) == name:
            return block_fn
    raise AssertionError(f"обработчик {name} не найден")


@pytest.mark.parametrize("name", ["run", "describe", "extend"])
def test_every_event_that_needs_the_mask_takes_it_fresh_and_first(name):
    """Кисть синхронизирует значение с задержкой; кнопка обязана забрать свежее.

    Скрипт подставляет значение первым аргументом — поэтому кисть обязана
    стоять в списке входов первой, иначе свежая маска ляжет не в тот вход.
    """
    demo = _edit_blocks()
    event = _event(demo, name)
    assert isinstance(event.inputs[0], component.MaskPainter), f"{name}: кисть не первый вход"
    assert event.js and "flush" in event.js and tab_edit.PAINTER_ID in event.js


def test_the_region_mode_reaches_the_painter():
    demo = _edit_blocks()
    event = _event(demo, "show_region")
    assert isinstance(event.outputs[0], component.MaskPainter)
    assert event.fn("annotation")["region"] == "annotation"


def test_the_painter_starts_in_the_language_and_mode_of_the_tab():
    demo = _edit_blocks()
    painter = next(block for block in demo.blocks.values() if isinstance(block, component.MaskPainter))
    assert painter.props["lang"] == config.AppConfig().lang, "язык кисти — язык запуска"
    assert painter.props["region"] == tab_edit.MASK_MASK
    assert painter.props["palette"] == list(tab_edit.ANNOTATION_COLOURS)


def test_a_bad_value_from_the_browser_becomes_a_status_line_not_a_crash(uploads, tmp_path):
    from fooocus_qwen.ui.i18n import Localizer
    from fooocus_qwen.ui.state import Studio

    outside = tmp_path / "secret.png"
    Image.new("RGB", (4, 4)).save(outside)
    studio = Studio(config.AppConfig())
    with gr.Blocks() as demo:
        tab_edit.build(studio, Localizer("ru"))
    describe = _event(demo, "describe").fn

    update, message = describe(_client_value(outside), "ru")
    assert "outside the uploads directory" in message


# --- клиентская часть ----------------------------------------------------------

SCRIPT = (component.HERE / "painter.js").read_text(encoding="utf-8")


def test_every_text_the_script_shows_is_translated():
    used = set(re.findall(r"data-(?:text|tip)=\"(painter_[a-z_]+)\"", SCRIPT))
    used |= set(re.findall(r"say\('(painter_[a-z_]+)'\)", SCRIPT))
    missing = used - set(i18n.PAINTER)
    assert not missing, f"в скрипте есть тексты без перевода: {sorted(missing)}"


def test_no_translated_text_is_left_unused():
    unused = [key for key in i18n.PAINTER if key not in SCRIPT]
    assert not unused, f"переводы, которых скрипт не показывает: {unused}"


def test_every_painter_text_has_both_languages():
    for key, pair in i18n.PAINTER.items():
        assert len(pair) == 2 and all(pair), key


def test_the_styles_are_built_into_the_script():
    script = component.client_script()
    assert script.startswith("const QP_CSS = ")
    assert ".qp-stage" in script


@pytest.mark.skipif(shutil.which("node") is None, reason="нет Node.js")
def test_the_client_script_is_valid_javascript(tmp_path):
    """Gradio исполняет скрипт как тело функции — так его и проверяем."""
    path = tmp_path / "painter.js"
    path.write_text("(function(element, trigger, props){\n" + component.client_script() + "\n});\n",
                    encoding="utf-8")
    result = subprocess.run(["node", "--check", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_each_stroke_draws_only_its_new_segment():
    """Суть замены редактора — стоимость движения не растёт с длиной мазка.

    Проверяется по устройству кода: обработчик движения рисует отрезок от
    последней точки до новой и не перерисовывает мазок целиком. Перерисовка
    всего слоя допустима только в отмене — там, где она и нужна.
    """
    extend = re.search(r"function extendStroke\(event\) \{(.*?)\n\}", SCRIPT, re.S).group(1)
    assert "segment(target, stroke, stroke.last, p)" in extend
    assert "const target = strokeTarget(stroke)" in extend
    assert "redraw(" not in extend and "replayStroke(" not in extend


def test_a_translucent_stroke_goes_through_the_live_canvas():
    """Полупрозрачный мазок рисуется непрозрачно на живом холсте и сливается в
    слой один раз — иначе стыки отрезков темнели бы внутри одного мазка."""
    target = re.search(r"function strokeTarget\(stroke\) \{(.*?)\n\}", SCRIPT, re.S).group(1)
    assert "stroke.alpha < 1 ? liveCtx : layerCtx" in target
    end = re.search(r"function endStroke\(\) \{(.*?)\n\}", SCRIPT, re.S).group(1)
    assert "layerCtx.globalAlpha = stroke.alpha" in end and "layerCtx.drawImage(ui.live, 0, 0)" in end
    replay = re.search(r"function replayStroke\(ctx, stroke\) \{(.*?)\n\}", SCRIPT, re.S).group(1)
    assert "ctx.globalAlpha = alpha" in replay, "отмена перерисовывает мазок с его прозрачностью"


def test_free_colour_is_only_for_the_sketch():
    """Свой цвет и непрозрачность — только в окне эскиза; маска их не получает."""
    begin = re.search(r"function beginStroke\(event, erase\) \{(.*?)\n\}", SCRIPT, re.S).group(1)
    assert "FREE_COLOUR && state.region === 'annotation' && !erase ? state.alpha : 1" in begin
    assert component.MaskPainter(region="mask").props["free_colour"] is False
    assert component.MaskPainter(free_colour=True).props["free_colour"] is True

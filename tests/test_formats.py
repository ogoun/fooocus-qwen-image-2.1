"""Картинки любого формата: декодер (imaging/formats.py) и маршрут перевода в PNG (ui/image_import.py).

Образцы делаются здесь же, кодировщиками тех же библиотек. RAW так не
сделать — LibRaw только читает; реальные снимки восьми марок проверены
живым прогоном (CHANGELOG), здесь — что RAW по расширению уходит в LibRaw и
что мусор с расширением RAW не роняет разбор.
"""

from __future__ import annotations

import base64
import io
import socket

import numpy as np
import pytest
from PIL import Image, ImageCms

from fooocus_qwen.imaging import formats

formats.register()


def _gradient(alpha: bool = False) -> Image.Image:
    array = np.zeros((48, 64, 3), np.uint8)
    array[..., 0] = np.linspace(0, 255, 64)[None, :]
    array[..., 1] = np.linspace(0, 255, 48)[:, None]
    array[..., 2] = 90
    image = Image.fromarray(array)
    if alpha:
        image.putalpha(Image.linear_gradient("L").resize(image.size))
    return image


def _encode(image: Image.Image, fmt: str, **options) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **options)
    return buffer.getvalue()


def _close(a: Image.Image, b: Image.Image, tolerance: float) -> bool:
    difference = np.abs(np.asarray(a, np.int16) - np.asarray(b.convert(a.mode), np.int16))
    return float(difference.mean()) <= tolerance


@pytest.mark.parametrize(
    ("name", "fmt", "tolerance"),
    [
        ("photo.heic", "HEIF", 6),
        ("photo.avif", "AVIF", 6),
        ("photo.jxl", "JXL", 6),
        ("scan.tif", "TIFF", 0),
        ("art.tga", "TGA", 0),
        ("art.qoi", "QOI", 0),
        ("art.jp2", "JPEG2000", 1),
        ("art.ppm", "PPM", 0),
        ("art.pcx", "PCX", 0),
        ("art.sgi", "SGI", 0),
    ],
)
def test_formats_decode_to_the_same_picture(name, fmt, tolerance):
    source = _gradient()
    image = formats.decode(_encode(source, fmt), name)
    assert image.mode == "RGB"
    assert image.size == source.size
    assert _close(image, source, tolerance)


def test_heic_is_recognised_by_content_not_name():
    image = formats.decode(_encode(_gradient(), "HEIF"), "renamed.bin")
    assert image.size == (64, 48)


def test_transparency_survives_and_opaque_alpha_is_dropped():
    with_alpha = formats.decode(_encode(_gradient(alpha=True), "TIFF"), "a.tiff")
    assert with_alpha.mode == "RGBA"
    opaque = _gradient()
    opaque.putalpha(255)
    assert formats.decode(_encode(opaque, "TIFF"), "b.tiff").mode == "RGB"


def test_sixteen_bit_scales_by_depth_not_by_the_brightest_pixel():
    # Тёмный 16-битный кадр не должен «осветлиться»: 0x2000 из 0xFFFF — это 32 из 255.
    dark = np.full((8, 8), 0x2000, np.uint16)
    image = formats.decode(_encode(Image.fromarray(dark), "PNG"), "dark.png")
    assert image.mode == "RGB"
    assert abs(int(np.asarray(image)[0, 0, 0]) - 32) <= 1


def test_cmyk_and_icc_end_in_srgb():
    source = _gradient()
    cmyk = formats.decode(_encode(source.convert("CMYK"), "JPEG", quality=95), "print.jpg")
    assert cmyk.mode == "RGB"
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    tagged = formats.decode(_encode(source, "JPEG", quality=95, icc_profile=srgb), "tagged.jpg")
    assert tagged.mode == "RGB"
    assert _close(tagged, source, 4)


def test_exif_orientation_is_applied():
    source = _gradient()
    exif = Image.Exif()
    exif[0x0112] = 6  # повернуть на 90° по часовой
    image = formats.decode(_encode(source, "JPEG", exif=exif), "phone.jpg")
    assert image.size == (48, 64)


def test_first_frame_of_an_animation():
    frames = [_gradient().convert("P"), _gradient().rotate(180).convert("P")]
    data = _encode(frames[0], "GIF", save_all=True, append_images=frames[1:])
    assert formats.decode(data, "anim.gif").size == (64, 48)


def test_ico_takes_the_largest_size():
    data = _encode(_gradient().resize((64, 64)), "ICO", sizes=[(16, 16), (64, 64)])
    assert formats.decode(data, "app.ico").size == (64, 64)


def test_exr_linear_light_becomes_srgb(tmp_path):
    openexr = pytest.importorskip("OpenEXR")
    linear = np.full((4, 4, 3), 0.2140, np.float32)  # sRGB 0.5 → линейный 0.214
    linear[0, 0] = 5.0  # ярче белого — обрезается, а не ломает шкалу
    header = {"compression": openexr.ZIP_COMPRESSION, "type": openexr.scanlineimage}
    path = tmp_path / "x.exr"  # OpenEXR пишет только в файл
    with openexr.File(header, {"RGB": linear}) as exr:
        exr.write(str(path))
    image = np.asarray(formats.decode(path.read_bytes(), "render.exr"))
    assert abs(int(image[1, 1, 0]) - 128) <= 1
    assert tuple(image[0, 0]) == (255, 255, 255)


def test_radiance_hdr_via_opencv():
    cv2 = pytest.importorskip("cv2")
    linear = np.full((4, 4, 3), 0.2140, np.float32)
    ok, data = cv2.imencode(".hdr", linear)
    assert ok
    image = np.asarray(formats.decode(data.tobytes(), "sky.hdr"))
    assert abs(int(image[1, 1, 0]) - 128) <= 3


def test_garbage_is_a_clear_error_not_a_crash():
    with pytest.raises(formats.FormatError, match="cannot read .heic"):
        formats.decode(b"not an image at all" * 20, "x.heic")
    with pytest.raises(formats.FormatError):
        formats.decode(b"not a raw file" * 20, "x.cr2")
    with pytest.raises(formats.FormatError, match="empty"):
        formats.decode(b"", "x.png")


def test_raw_extension_goes_to_libraw(monkeypatch):
    calls = []

    def fake(data):
        calls.append(len(data))
        return Image.new("RGB", (6, 4))

    monkeypatch.setattr(formats, "_decode_raw", fake)
    # Внутри RAW чаще всего TIFF — Pillow открыл бы миниатюру; расширение решает.
    tiff = _encode(_gradient(), "TIFF")
    assert formats.decode(tiff, "IMG_0001.CR2").size == (6, 4)
    assert calls == [len(tiff)]


def test_missing_package_is_named(monkeypatch):
    monkeypatch.setitem(formats._missing, ".jxl", "pillow-jxl-plugin")
    with pytest.raises(formats.FormatError, match="pillow-jxl-plugin"):
        formats.decode(b"\xff\x0a garbage" * 4, "x.jxl")


def test_missing_format_packages_are_reported(monkeypatch):
    assert formats.missing_packages() == []  # в окружении проекта всё стоит
    real = formats.importlib.util.find_spec
    monkeypatch.setattr(formats.importlib.util, "find_spec",
                        lambda name: None if name == "rawpy" else real(name))
    assert formats.missing_packages() == [("rawpy", "camera RAW")]
    assert "requirements.txt" in formats.INSTALL_HINT


def test_too_many_pixels_is_refused(monkeypatch):
    monkeypatch.setattr(formats, "MAX_PIXELS", 100)
    with pytest.raises(formats.FormatError, match="too large"):
        formats.decode(_encode(_gradient(), "PNG"), "big.png")


def test_png_round_trip_is_lossless():
    source = _gradient(alpha=True)
    again = Image.open(io.BytesIO(formats.to_png(source)))
    assert again.mode == "RGBA"
    assert np.array_equal(np.asarray(again), np.asarray(source))


def test_needs_conversion_only_for_what_browsers_do_not_show():
    for name in ("a.png", "b.JPG", "c.webp", "d.gif", "e.bmp"):
        assert not formats.needs_conversion(name)
    for name in ("a.heic", "b.HEIF", "c.cr3", "d.tiff", "e.psd", "f.jxl", "g.exr", "h.avif", "i.svg"):
        assert formats.needs_conversion(name)


def test_extension_lists_do_not_overlap():
    assert not set(formats.NATIVE_EXTENSIONS) & set(formats.CONVERTED_EXTENSIONS)
    assert all(ext == ext.lower() and ext.startswith(".") for ext in formats.CONVERTED_EXTENSIONS)


# --- API: base64 любого формата ---


def test_api_accepts_heic_as_base64():
    from fooocus_qwen.api import images

    data = base64.b64encode(_encode(_gradient(), "HEIF")).decode()
    image = images.decode_base64(f"data:image/heic;base64,{data}")
    assert image.size == (64, 48)
    with pytest.raises(images.ImageError):
        images.decode_base64(base64.b64encode(b"junk" * 50).decode())


# --- маршрут перевода в интерфейсе ---


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def gradio_servers():
    gr = pytest.importorskip("gradio")
    httpx = pytest.importorskip("httpx")
    from fooocus_qwen.ui import image_import

    servers = {}
    for key, auth in (("open", None), ("locked", ("anna", "secret"))):
        with gr.Blocks() as demo:
            gr.Markdown("x")
        port = _free_port()
        demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, auth=auth)
        image_import.install(demo.app)
        image_import.install(demo.app)  # повторно — без второго маршрута
        servers[key] = (demo, f"http://127.0.0.1:{port}")
    yield servers, httpx, image_import
    for demo, _ in servers.values():
        demo.close()


def test_route_converts_heic_to_png(gradio_servers):
    servers, httpx, image_import = gradio_servers
    demo, url = servers["open"]
    assert sum(getattr(r, "path", "") == image_import.CONVERT_PATH for r in demo.app.router.routes) == 1
    heic = _encode(_gradient(), "HEIF")
    response = httpx.post(f"{url}{image_import.CONVERT_PATH}", content=heic, headers={"X-File-Name": "IMG_0001.HEIC"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).size == (64, 48)

    bad = httpx.post(f"{url}{image_import.CONVERT_PATH}", content=b"junk" * 40, headers={"X-File-Name": "x.heic"})
    assert bad.status_code == 415
    assert "cannot read" in bad.json()["error"]


def test_route_is_behind_the_ui_login(gradio_servers):
    servers, httpx, image_import = gradio_servers
    _, url = servers["locked"]
    heic = _encode(_gradient(), "HEIF")
    # Контроль: собственный маршрут Gradio без входа тоже закрыт — значит, вход включён.
    assert httpx.get(f"{url}/config").status_code == 401
    assert httpx.post(f"{url}{image_import.CONVERT_PATH}", content=heic).status_code == 401
    with httpx.Client(base_url=url) as client:
        login = client.post("/login", data={"username": "anna", "password": "secret"})
        assert login.status_code == 200
        response = client.post(image_import.CONVERT_PATH, content=heic, headers={"X-File-Name": "a.heic"})
        assert response.status_code == 200


def test_route_refuses_oversized_files(gradio_servers, monkeypatch):
    servers, httpx, image_import = gradio_servers
    _, url = servers["open"]
    monkeypatch.setattr(image_import, "MAX_BYTES", 10)
    response = httpx.post(f"{url}{image_import.CONVERT_PATH}", content=b"x" * 100)
    assert response.status_code == 413


def test_page_script_gets_lists_and_texts_from_python():
    from fooocus_qwen.ui import image_import

    script = image_import.script()
    for marker in ("__NATIVE__", "__CONVERTED__", "__PATH__", "__TEXT__"):
        assert marker not in script
    assert '".heic"' in script and '".cr3"' in script
    assert "fooocus/convert-image" in script
    assert "Converting {name} to PNG" in script

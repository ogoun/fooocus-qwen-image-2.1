"""Работа с готовыми роликами: проба, кадры, склейка, листы.

Ролики синтетические и крошечные (320×192, 1–2 с): их делает сам ffmpeg
(``testsrc2`` + ``sine``) или PyAV, когда нужен заранее известный цвет
каждого кадра. Прогон — секунды.
"""

from __future__ import annotations

import subprocess
import sys

import av
import numpy as np
import pytest
from PIL import Image

from fooocus_qwen.video import assemble

FPS = 24
_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        [assemble.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args],
        check=True, capture_output=True, creationflags=_FLAGS,
    )


def make_clip(path, seconds=1.0, size=(320, 192), audio=True, rate=48_000):
    """Тестовая таблица с тоном 440 Гц (или без звука)."""
    width, height = size
    args = ["-f", "lavfi", "-i", f"testsrc2=s={width}x{height}:r={FPS}:d={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=f=440:r={rate}:d={seconds}", "-ac", "2", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    _ffmpeg(*args)
    return path


def make_colored(path, colors):
    """Ролик из кадров заданных цветов; один ключевой кадр в начале и B-кадры.

    GOP длиннее ролика: «ближайший ключевой» к концу — первый кадр, и любая
    попытка вернуть его вместо последнего видна по цвету.
    """
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 320, 192, "yuv420p"
        stream.options = {"g": "250", "bf": "3", "crf": "10"}
        for color in colors:
            array = np.empty((192, 320, 3), np.uint8)
            array[:] = color
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def mean_color(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), np.float32).reshape(-1, 3).mean(axis=0)


def assert_color(image, expected):
    assert np.abs(mean_color(image) - np.asarray(expected, np.float32)).max() < 12


RED, GREEN, BLUE = (220, 30, 30), (30, 200, 30), (30, 30, 220)


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    folder = tmp_path_factory.mktemp("clips")
    return {
        "a": make_clip(folder / "a.mp4", 1.0),
        "b": make_clip(folder / "b.mp4", 2.0, rate=32_000),
        "silent": make_clip(folder / "silent.mp4", 1.0, audio=False),
        "tall": make_clip(folder / "tall.mp4", 1.0, size=(192, 320)),
    }


@pytest.fixture(scope="module")
def colored(tmp_path_factory):
    # 1 с красного, почти секунда зелёного, последний кадр синий.
    colors = [RED] * 24 + [GREEN] * 23 + [BLUE]
    return make_colored(tmp_path_factory.mktemp("colored") / "colored.mp4", colors)


def test_probe(clips):
    info = assemble.probe(clips["b"])
    assert (info.width, info.height) == (320, 192)
    assert info.fps == pytest.approx(24)
    assert info.frames == 48
    assert info.seconds == pytest.approx(2.0, abs=1 / FPS)
    assert info.has_audio
    assert not assemble.probe(clips["silent"]).has_audio


def test_last_frame_is_the_last_not_a_keyframe(colored, tmp_path):
    dest = tmp_path / "last.png"
    image = assemble.last_frame(colored, dest)
    assert_color(image, BLUE)
    assert dest.is_file()


def test_first_frame_and_frame_at(colored):
    assert_color(assemble.first_frame(colored), RED)
    assert_color(assemble.frame_at(colored, 0.5), RED)
    assert_color(assemble.frame_at(colored, 1.5), GREEN)
    assert_color(assemble.frame_at(colored, 99), BLUE)


def _audio_seconds(path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        return float(stream.duration * stream.time_base)


def test_concat_cut(clips, tmp_path):
    out = assemble.concat([clips["a"], clips["b"]], tmp_path / "cut.mp4", ["cut"])
    info = assemble.probe(out)
    assert info.seconds == pytest.approx(3.0, abs=1 / FPS)
    assert info.has_audio and (info.width, info.height) == (320, 192)
    assert _audio_seconds(out) == pytest.approx(3.0, abs=0.05)
    assert not (tmp_path / "cut.part.mp4").exists()


def test_concat_dissolve_shortens_by_transition(clips, tmp_path):
    out = assemble.concat(
        [clips["a"], clips["b"], clips["a"]], tmp_path / "dissolve.mp4",
        ["dissolve", "wipeleft"], transition_seconds=0.25,
    )
    info = assemble.probe(out)
    assert info.seconds == pytest.approx(1 + 2 + 1 - 2 * 0.25, abs=1 / FPS)
    assert info.has_audio
    assert _audio_seconds(out) == pytest.approx(info.seconds, abs=0.05)


def test_transition_capped_at_half_of_shortest(clips, tmp_path):
    out = assemble.concat([clips["a"], clips["b"]], tmp_path / "cap.mp4", ["fade"], transition_seconds=5)
    assert assemble.probe(out).seconds == pytest.approx(3.0 - 0.5, abs=1 / FPS)


def test_silent_clip_in_the_middle_and_size_normalised(clips, tmp_path):
    out = assemble.concat(
        [clips["a"], clips["silent"], clips["tall"]], tmp_path / "mixed.mp4",
        ["dissolve", "cut"], transition_seconds=0.25, loudnorm=False,
    )
    info = assemble.probe(out)
    assert (info.width, info.height) == (320, 192)
    assert info.has_audio
    assert info.seconds == pytest.approx(3.0 - 0.25, abs=1 / FPS)
    # Портретный клип вписан с полями по бокам, а не растянут.
    tail = np.asarray(assemble.last_frame(out).convert("L"), np.float32)
    assert tail[:, :40].mean() < 20 and tail[:, 140:180].mean() > 40


def test_concat_rejects_unknown_transition(clips, tmp_path):
    with pytest.raises(ValueError, match="Unknown transition"):
        assemble.concat([clips["a"], clips["a"]], tmp_path / "x.mp4", ["spin"])


def test_contact_sheet_and_poster(clips, tmp_path):
    sheet = assemble.contact_sheet([clips["a"], clips["b"], clips["silent"]], tmp_path / "sheet.png", columns=2)
    with Image.open(sheet) as image:
        # Два шота в ряду по три кадра 256×154 и подпись; два ряда.
        assert image.width == 8 + 2 * (3 * 256 + 2 * 8 + 8)
        assert image.height > 2 * 154
    poster = assemble.poster(clips["a"], tmp_path / "poster.jpg")
    with Image.open(poster) as image:
        assert image.format == "JPEG" and image.size == (320, 192)
    webp = assemble.poster(clips["a"], tmp_path / "poster.webp", max_side=160)
    with Image.open(webp) as image:
        assert image.format == "WEBP" and max(image.size) == 160


def test_trim_and_extract_audio(clips, tmp_path):
    cut = assemble.trim(clips["b"], tmp_path / "trim.mp4", 0.5, 1.5)
    assert assemble.probe(cut).seconds == pytest.approx(1.0, abs=1 / FPS)
    wav = assemble.extract_audio(clips["a"], tmp_path / "a.wav")
    assert wav.stat().st_size > 1000
    with pytest.raises(ValueError, match="No audio"):
        assemble.extract_audio(clips["silent"], tmp_path / "s.wav")


def test_ffmpeg_error_carries_stderr(tmp_path):
    with pytest.raises(assemble.FfmpegError, match="ffmpeg failed"):
        assemble.trim(tmp_path / "missing.mp4", tmp_path / "o.mp4", 0, 1)


def test_missing_ffmpeg_is_explained(monkeypatch):
    import imageio_ffmpeg

    monkeypatch.delenv(assemble.FFMPEG_ENV, raising=False)
    monkeypatch.setattr(assemble.shutil, "which", lambda name: None)

    def nothing():
        raise RuntimeError("no binary")

    monkeypatch.setattr(imageio_ffmpeg, "get_ffmpeg_exe", nothing)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        assemble.ffmpeg_exe()


def test_env_override(monkeypatch, tmp_path):
    fake = tmp_path / "ffmpeg.exe"
    fake.write_bytes(b"")
    monkeypatch.setenv(assemble.FFMPEG_ENV, str(fake))
    assert assemble.ffmpeg_exe() == str(fake)
    monkeypatch.setenv(assemble.FFMPEG_ENV, str(tmp_path / "nope.exe"))
    with pytest.raises(RuntimeError, match="no such file"):
        assemble.ffmpeg_exe()

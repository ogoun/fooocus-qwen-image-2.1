"""Готовые ролики: проба, кадры, склейка, контактный лист, постер.

Генерация отдаёт MP4 (H.264, 24 кадра в секунду, стерео AAC 32 или 48 кГц);
дальше с файлом работают без модели: берут последний кадр шота как первый
кадр следующего, клеят шоты в фильм, показывают раскадровку и миниатюру в
галерее. Всё это — здесь.

Два инструмента и почему именно они:

* PyAV (``av``) — чтение: длительность, размер, кадры. Он ставится колесом
  со своими библиотеками FFmpeg, ffprobe не нужен, а кадр приходит сразу
  картинкой PIL — без промежуточных файлов и разбора текстового вывода.
* ffmpeg — запись и склейка: граф фильтров (``xfade``, ``acrossfade``,
  ``loudnorm``) из PyAV собирать долго и хрупко, а ffmpeg делает это одной
  командой. Берётся системный, если есть, иначе — из ``imageio-ffmpeg``
  (колесо с libx264), см. :func:`ffmpeg_exe`.

Последний кадр — именно последний, а не ближайший ключевой: перемотка
встаёт на ключевой кадр перед концом, и хвост декодируется до конца потока.
Ключевой кадр отстоит от конца на секунды (у x264 — до 250 кадров), и
продолжение шота от него начиналось бы с прыжка назад.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FFMPEG_ENV = "FOOOCUS_FFMPEG"

# Переход → эффект xfade. «dissolve» монтажёра — это наплыв (crossfade), а у
# xfade одноимённый эффект — зернистое рассыпание, поэтому он отображается в
# ``fade``; «fade» монтажёра — через чёрное.
TRANSITIONS: dict[str, str | None] = {
    "cut": None,
    "dissolve": "fade",
    "fade": "fadeblack",
    "wipeleft": "wipeleft",
    "wiperight": "wiperight",
}
CUT = "cut"

# Аудиофейд на склейке встык: 25 мс убирают щелчок от разрыва волны и ещё не
# слышны как провал. Правило «Вайб-монтажа»: короткие фейды убирают щелчки,
# длинный кроссфейд накладывает слова.
CUT_FADE_SECONDS = 0.025
# Громкость для соцсетей: −14 LUFS интегрально, пик не выше −1.5 dBTP.
LOUDNESS_LUFS = -14.0
TRUE_PEAK_DB = -1.5
SAMPLE_RATE = 48_000
AUDIO_BITRATE = "192k"
CRF = 18
# Длиннее — граф уходит в файл: командная строка Windows ограничена 32 767 символами.
_INLINE_GRAPH_LIMIT = 8_000
_ERROR_TAIL = 2_000


class FfmpegError(RuntimeError):
    """ffmpeg завершился с ошибкой; в тексте — хвост его stderr."""


@dataclass(frozen=True)
class MediaInfo:
    """Что известно о ролике по его видеопотоку."""

    width: int
    height: int
    fps: float
    frames: int
    seconds: float
    has_audio: bool


# --- ffmpeg -------------------------------------------------------------------


def ffmpeg_exe() -> str:
    """Путь к ffmpeg: ``FOOOCUS_FFMPEG`` → ``PATH`` → колесо ``imageio-ffmpeg``.

    Явно заданный в окружении путь не подменяется молча другим: если файла
    нет, это ошибка настройки, и о ней лучше сказать.
    """
    configured = os.environ.get(FFMPEG_ENV, "").strip().strip('"')
    if configured:
        if Path(configured).is_file():
            return configured
        raise RuntimeError(f"{FFMPEG_ENV} points to {configured!r}, but there is no such file")
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass
    raise RuntimeError(
        "ffmpeg not found: install it into PATH, set "
        f"{FFMPEG_ENV} to ffmpeg.exe, or run: pip install imageio-ffmpeg"
    )


def _run(args: Sequence[str]) -> None:
    """Запускает ffmpeg без окна консоли; ошибка — исключением с хвостом stderr."""
    command = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args]
    # Без флага каждый вызов из оконного приложения мигал бы чёрным окном.
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    result = subprocess.run(command, capture_output=True, creationflags=flags, check=False)
    if result.returncode != 0:
        tail = result.stderr.decode("utf-8", errors="replace").strip()[-_ERROR_TAIL:]
        raise FfmpegError(f"ffmpeg failed (exit {result.returncode}): {tail or 'no output'}")


def _part_path(out: Path) -> Path:
    """Имя для записи: готовый файл появляется целиком или не появляется вовсе."""
    return out.with_name(f"{out.stem}.part{out.suffix}")


def _finish(part: Path, out: Path) -> Path:
    os.replace(part, out)
    return out


# --- чтение -------------------------------------------------------------------


def _open(path: Path):
    import av

    return av.open(str(path))


def probe(path: str | Path) -> MediaInfo:
    """Размер, частота, число кадров и длительность видеопотока; есть ли звук."""
    with _open(Path(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        fps = float(rate) if rate else 0.0
        if stream.duration is not None and stream.time_base is not None:
            stream_seconds = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            stream_seconds = container.duration / 1_000_000
        else:
            stream_seconds = 0.0
        frames = stream.frames or (round(stream_seconds * fps) if fps else 0)
        if not frames:
            # Заголовок молчит (бывает у потоков без индекса) — счёт по пакетам.
            frames = sum(1 for packet in container.demux(stream) if packet.size)
        # Длительность по кадрам точнее заголовка: её и получит склейка после fps=.
        seconds = frames / fps if frames and fps else stream_seconds
        return MediaInfo(
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            fps=fps,
            frames=int(frames),
            seconds=seconds,
            has_audio=bool(container.streams.audio),
        )


def _save(image: Image.Image, dest: str | Path | None) -> Image.Image:
    if dest is not None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest)
    return image


def first_frame(video: str | Path, dest_png: str | Path | None = None) -> Image.Image:
    """Первый кадр в порядке показа."""
    with _open(Path(video)) as container:
        for frame in container.decode(video=0):
            return _save(frame.to_image(), dest_png)
    raise ValueError(f"No frames in {video}")


def last_frame(video: str | Path, dest_png: str | Path | None = None) -> Image.Image:
    """Ровно последний кадр: от ключевого кадра перед хвостом — декодирование до конца.

    Декодер отдаёт кадры в порядке показа, и после сброса буфера (конец
    потока) последним приходит последний показываемый — B-кадры не мешают.
    Если перемотка ничего не дала (бывает у файлов без индекса), поток
    декодируется целиком.
    """
    path = Path(video)
    with _open(path) as container:
        stream = container.streams.video[0]
        last = None
        if stream.duration and stream.time_base:
            # Секунда до конца; seek(backward) сам отступит до ключевого кадра.
            target = max(0, int(stream.duration - 1 / stream.time_base) + (stream.start_time or 0))
            try:
                container.seek(target, stream=stream, backward=True, any_frame=False)
                for frame in container.decode(stream):
                    last = frame
            except Exception:  # noqa: BLE001 — перемотка не обязана работать; есть запасной путь
                last = None
        if last is None:
            container.seek(0)
            for frame in container.decode(stream):
                last = frame
    if last is None:
        raise ValueError(f"No frames in {video}")
    return _save(last.to_image(), dest_png)


def frame_at(video: str | Path, seconds: float, dest_png: str | Path | None = None) -> Image.Image:
    """Кадр, показываемый в момент ``seconds`` (последний с началом не позже него)."""
    with _open(Path(video)) as container:
        stream = container.streams.video[0]
        start = float((stream.start_time or 0) * stream.time_base) if stream.time_base else 0.0
        moment = start + max(float(seconds), 0.0)
        # Полкадра допуска: время кадра в контейнере округлено до time_base.
        rate = stream.average_rate or stream.guessed_rate
        slack = 0.5 / float(rate) if rate else 0.0
        if moment > start and stream.time_base:
            container.seek(int(moment / stream.time_base), stream=stream, backward=True, any_frame=False)
        chosen = None
        for frame in container.decode(stream):
            if chosen is not None and frame.time is not None and frame.time > moment + slack:
                break
            chosen = frame
    if chosen is None:
        raise ValueError(f"No frames in {video}")
    return _save(chosen.to_image(), dest_png)


# --- запись -------------------------------------------------------------------


def _even(value: int) -> int:
    # yuv420p хранит цвет на блоках 2×2: нечётная сторона кодеру не годится.
    return max(2, value - value % 2)


def _video_chain(index: int, width: int, height: int, fps: int, seconds: float) -> str:
    """Кадр клипа → общий холст: вписать без искажения, поля чёрным.

    ``fps`` дважды не по ошибке: первый выравнивает кадры по общей сетке до
    ``trim``, второй заново объявляет частоту — ``setpts`` её сбрасывает в
    «переменную» (1/0), а ``xfade`` с такой частотой не строится.
    """
    return (
        f"[{index}:v:0]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,settb=AVTB,fps={fps},"
        f"trim=duration={seconds:.6f},setpts=PTS-STARTPTS,format=yuv420p,fps={fps}[v{index}]"
    )


def _audio_chain(index: int, info: MediaInfo, seconds: float, fade_in: bool, fade_out: bool) -> str:
    """Звук клипа той же длины, что картинка; без звука — тишина.

    Длина подгоняется (``apad`` + ``atrim``) к картинке: иначе каждый шов
    сдвигал бы звук относительно кадра, и к концу фильма речь уезжала бы.
    """
    if info.has_audio:
        source = f"[{index}:a:0]aresample={SAMPLE_RATE},"
    else:
        source = f"anullsrc=r={SAMPLE_RATE}:cl=stereo,"
    chain = (
        f"{source}aformat=sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts=stereo,"
        f"apad=whole_dur={seconds:.6f},atrim=duration={seconds:.6f},asetpts=PTS-STARTPTS"
    )
    fade = min(CUT_FADE_SECONDS, seconds / 4)
    if fade_in:
        chain += f",afade=t=in:st=0:d={fade:.6f}"
    if fade_out:
        chain += f",afade=t=out:st={seconds - fade:.6f}:d={fade:.6f}"
    return chain + f"[a{index}]"


def _normalize_transitions(transitions: Sequence[str], joins: int) -> list[str]:
    names = [str(name).strip().lower() or CUT for name in transitions][:joins]
    names += [CUT] * (joins - len(names))
    for name in names:
        if name not in TRANSITIONS:
            raise ValueError(f"Unknown transition {name!r}; expected one of: {', '.join(TRANSITIONS)}")
    return names


def concat(
    clips: Sequence[str | Path],
    out: str | Path,
    transitions: Sequence[str] = (),
    transition_seconds: float = 0.5,
    loudnorm: bool = True,
    fps: int = 24,
) -> Path:
    """Склеивает ролики в один MP4 (H.264 yuv420p + AAC, ``+faststart``).

    ``transitions[i]`` — шов между ``clips[i]`` и ``clips[i + 1]``: ``cut``
    (встык, с короткими аудиофейдами против щелчков) или наплыв ``dissolve``,
    ``fade`` (через чёрное), ``wipeleft``, ``wiperight`` — картинка через
    ``xfade``, звук через ``acrossfade`` той же длины. Недостающие швы — ``cut``.

    Холст — размер первого клипа; остальные вписываются в него с полями.
    Сдвиг каждого ``xfade`` считается от фактических длительностей (проба)
    с накоплением: шов отнимает у общей длины свою длительность. Переход не
    длиннее половины самого короткого клипа, иначе соседние швы наложились бы.
    """
    paths = [Path(clip) for clip in clips]
    if not paths:
        raise ValueError("Nothing to concatenate: no clips given")
    out = Path(out)
    infos = [probe(path) for path in paths]
    durations = [info.seconds for info in infos]
    if min(durations) <= 0:
        raise ValueError("A clip has zero duration")
    names = _normalize_transitions(transitions, len(paths) - 1)
    overlap = max(0.0, min(float(transition_seconds), min(durations) / 2))
    # Шаг сетки кадров: переход короче кадра xfade не нарисует — это просто стык.
    if overlap < 1 / fps:
        names = [CUT] * len(names)

    width, height = _even(infos[0].width), _even(infos[0].height)
    graph: list[str] = []
    for index, info in enumerate(infos):
        graph.append(_video_chain(index, width, height, fps, durations[index]))
        fade_in = index > 0 and names[index - 1] == CUT
        fade_out = index < len(names) and names[index] == CUT
        graph.append(_audio_chain(index, info, durations[index], fade_in, fade_out))

    video, audio, length = "[v0]", "[a0]", durations[0]
    for index, name in enumerate(names, start=1):
        joined_video, joined_audio = f"[vj{index}]", f"[aj{index}]"
        effect = TRANSITIONS[name]
        if effect is None:
            # concat теряет частоту кадров (1/0), а следующий xfade без неё не
            # строится — fps= возвращает её.
            graph.append(f"{video}{audio}[v{index}][a{index}]concat=n=2:v=1:a=1[vc{index}]{joined_audio}")
            graph.append(f"[vc{index}]fps={fps}{joined_video}")
            length += durations[index]
        else:
            offset = length - overlap
            graph.append(
                f"{video}[v{index}]xfade=transition={effect}:duration={overlap:.6f}:offset={offset:.6f}{joined_video}"
            )
            graph.append(f"{audio}[a{index}]acrossfade=d={overlap:.6f}:c1=tri:c2=tri{joined_audio}")
            length += durations[index] - overlap
        video, audio = joined_video, joined_audio

    if loudnorm:
        # loudnorm внутри работает на 192 кГц — обратно к рабочей частоте.
        graph.append(
            f"{audio}loudnorm=I={LOUDNESS_LUFS}:TP={TRUE_PEAK_DB}:LRA=11,"
            f"aresample={SAMPLE_RATE}[aout]"
        )
        audio = "[aout]"

    args: list[str] = []
    for path in paths:
        args += ["-i", str(path)]
    out.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(out)
    script = out.with_name(f"{out.stem}.filtergraph.txt")
    text = ";\n".join(graph)
    if len(text) > _INLINE_GRAPH_LIMIT:
        script.write_text(text, encoding="utf-8", newline="\n")
        args += ["-filter_complex_script", str(script)]
    else:
        args += ["-filter_complex", text]
    args += [
        "-map", video, "-map", audio,
        "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF), "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", str(SAMPLE_RATE), "-ac", "2",
        "-movflags", "+faststart",
        str(part),
    ]
    try:
        _run(args)
    finally:
        script.unlink(missing_ok=True)
    return _finish(part, out)


def trim(video: str | Path, out: str | Path, start: float, end: float | None = None) -> Path:
    """Отрезок ``[start, end)`` с перекодированием — точно по кадру, а не по ключевому."""
    if start < 0 or (end is not None and end <= start):
        raise ValueError("Trim range must satisfy 0 <= start < end")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(out)
    args = ["-i", str(video), "-ss", f"{start:.6f}"]
    if end is not None:
        args += ["-to", f"{end:.6f}"]
    args += [
        "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-movflags", "+faststart", str(part),
    ]
    _run(args)
    return _finish(part, out)


def extract_audio(video: str | Path, dest_wav: str | Path) -> Path:
    """Звуковая дорожка в WAV (16 бит, частота и каналы — как в ролике)."""
    if not probe(video).has_audio:
        raise ValueError(f"No audio stream in {video}")
    dest = Path(dest_wav)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = _part_path(dest)
    _run(["-i", str(video), "-vn", "-c:a", "pcm_s16le", str(part)])
    return _finish(part, dest)


# --- картинки -----------------------------------------------------------------


def poster(video: str | Path, dest: str | Path, max_side: int | None = None) -> Path:
    """Первый кадр миниатюрой для галереи; формат — по расширению (JPEG, WEBP, PNG)."""
    image = first_frame(video)
    if max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    suffix = dest.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        image.convert("RGB").save(dest, "JPEG", quality=90, optimize=True)
    elif suffix == ".webp":
        image.save(dest, "WEBP", quality=88, method=4)
    else:
        image.save(dest)
    return dest


def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: шрифт без размера
        return ImageFont.load_default()


def contact_sheet(
    clips: Sequence[str | Path],
    dest_png: str | Path,
    columns: int = 3,
    thumb_width: int = 256,
) -> Path:
    """Контактный лист раскадровки: у каждого шота первый, средний и последний кадр.

    Ячейка — тройка кадров одного шота с подписью номера и длительности;
    ``columns`` — сколько шотов в ряду листа. Глазом сразу видно, стыкуется
    ли конец шота с началом следующего и не уехал ли персонаж.
    """
    paths = [Path(clip) for clip in clips]
    if not paths:
        raise ValueError("No clips for a contact sheet")
    columns = max(1, int(columns))
    gap, label_height = 8, 28
    font = _font(18)

    cells: list[tuple[str, list[Image.Image]]] = []
    for number, path in enumerate(paths, start=1):
        info = probe(path)
        middle = info.seconds / 2
        frames = [first_frame(path), frame_at(path, middle), last_frame(path)]
        thumb_height = max(1, round(thumb_width * info.height / max(info.width, 1)))
        frames = [frame.convert("RGB").resize((thumb_width, thumb_height), Image.Resampling.LANCZOS) for frame in frames]
        cells.append((f"Shot {number} · {info.seconds:.2f} s", frames))

    cell_width = 3 * thumb_width + 2 * gap
    cell_height = label_height + max(frames[0].height for _, frames in cells)
    rows = math.ceil(len(cells) / columns)
    used_columns = min(columns, len(cells))
    sheet = Image.new(
        "RGB",
        (gap + used_columns * (cell_width + gap), gap + rows * (cell_height + gap)),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(sheet)
    for index, (label, frames) in enumerate(cells):
        row, column = divmod(index, columns)
        left = gap + column * (cell_width + gap)
        top = gap + row * (cell_height + gap)
        draw.text((left + 2, top + 4), label, fill=(235, 235, 235), font=font)
        for position, frame in enumerate(frames):
            sheet.paste(frame, (left + position * (thumb_width + gap), top + label_height))

    dest = Path(dest_png)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)
    return dest

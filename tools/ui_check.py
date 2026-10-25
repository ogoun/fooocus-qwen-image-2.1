"""Проверка интерфейса в настоящем браузере — без модели и без видеокарты.

Юнит-тесты проверяют функции, а дефекты интерфейса этого проекта раз за
разом жили в доставке результата до экрана: переключатель языка менял
надпись на кнопке и больше ничего, холст не держал высоту, кисть отставала
от руки. Ни один тест этого не видел, потому что ни один не открывал
браузер. Этот инструмент открывает.

Приложение поднимается той же дорогой, что и боевой запуск
(``ui.app.start``), но генератор подменён: он запоминает запрос и
возвращает исходник как результат. Так проверяется всё, что лежит между
рукой человека и запросом к модели — вплоть до того, в каком месте кадра
оказалась нарисованная маска, — за минуту и без тридцати трёх гигабайт весов.

Нужен Playwright с Chromium (``requirements-dev.txt``):
    .venv\\Scripts\\python -m pip install -r requirements-dev.txt
    .venv\\Scripts\\python -m playwright install chromium

Запуск:
    .venv\\Scripts\\python tools\\ui_check.py
    .venv\\Scripts\\python tools\\ui_check.py --only painter latency

Код возврата — число проваленных проверок.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fooocus_qwen.logging_setup import use_utf8_console

use_utf8_console()

import argparse
import shutil
import statistics
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from PIL import Image

from fooocus_qwen import config
from fooocus_qwen.engine.generator import GeneratedImage

PAINTER = "qs-edit-painter"
API = f"window.__qsPainters['{PAINTER}']"
SIZES = [(3840, 2160), (2560, 1440), (1920, 1080), (1600, 900), (1366, 768), (1100, 800)]
# Номера вкладок — по порядку в полосе (ui/app.py). Именами, а не числами:
# вкладка «Видео» встала третьей, и числа сдвинулись бы во всех сценариях.
TAB_GENERATE, TAB_EDIT, TAB_VIDEO, TAB_GALLERY, TAB_SETTINGS = range(5)
TABS = 5


# --- подставной генератор ------------------------------------------------------

class FakeGenerator:
    """Запоминает запросы и возвращает исходник вместо генерации."""

    def __init__(self) -> None:
        self.requests: list = []
        self.lock = threading.Lock()

    def generate(self, request, progress=None):
        with self.lock:
            self.requests.append(request)
        image = request.source if request.source is not None else Image.new("RGB", (256, 256), "gray")
        return [GeneratedImage(image=image.convert("RGB"), seed=1,
                               parameters={"seconds": 0.0, "clipped_outside_pct": 0.0})]

    def interrupt(self) -> None:
        pass

    @property
    def pipe(self):
        """Пайплайн-заглушка: переключатель внимания ставит механизм трансформеру."""
        return self

    @property
    def transformer(self):
        return self

    def set_attention_backend(self, name: str) -> None:
        self.attention = name

    @contextmanager
    def exclusive(self):
        with self.lock:
            yield

    def wait(self, count: int, timeout: float = 30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if len(self.requests) >= count:
                    return self.requests[count - 1]
            time.sleep(0.1)
        raise TimeoutError(f"the generator did not receive request #{count} within {timeout} s")


class FakeVideoEngine:
    """Видеодвижок-заглушка: события прогресса и готовый маленький MP4 вместо H3."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.jobs: list[dict] = []
        self.busy = False
        self.cancelled = False

    def generate(self, job: dict, on_event=None) -> dict:
        from fooocus_qwen.video import assemble

        self.jobs.append(job)
        if on_event is not None:
            for done in range(1, 4):
                on_event({"event": "progress", "part": 0, "parts": 1,
                          "value": {"label": "Sampling", "timing_phase": "sampling", "done": done, "total": 3,
                                    "overall": {"fraction": done / 4, "remaining_seconds": 3 - done}}})
                time.sleep(0.1)
        output = self.directory / f"fake{len(self.jobs)}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        import subprocess

        color = ("red", "green", "blue")[len(self.jobs) % 3]
        subprocess.run(
            [assemble.ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
             f"color=c={color}:s={job['width'] // 4}x{job['height'] // 4}:r=24:d=1", "-f", "lavfi", "-i",
             "sine=frequency=440:duration=1", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", str(output)],
            check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return {"event": "done", "output": str(output), "parts": [str(output)]}

    def cancel(self) -> None:
        self.cancelled = True

    def release(self) -> None:
        pass

    def close(self) -> None:
        pass


def prepare_studio(studio, fake, video) -> None:
    """Подмены перед подъёмом сервера: генератор картинок и видеодвижок."""
    from fooocus_qwen.video.engine import EngineStatus

    studio._generator = fake
    studio._video = video
    studio.video_status = lambda: EngineStatus(True, True)
    # Перед роликом студия выгружает модель картинок — здесь это выгрузило бы
    # подставной генератор, и следующая генерация загрузила бы настоящие
    # 33 ГБ на видеокарту (так и случилось при первом прогоне сценария video).
    studio.claim_gpu_for_video = lambda: None
    # И любой другой путь выгрузки (смена точности, профиля): подставной
    # генератор обязан пережить весь прогон.
    studio.unload = lambda: None


# --- учёт результатов ----------------------------------------------------------

@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def check(self, ok: bool, text: str) -> bool:
        (self.passed if ok else self.failed).append(text)
        print(f"  {'ok  ' if ok else 'FAIL'} {text}")
        return ok


# --- вспомогательное -----------------------------------------------------------

def sample_image(directory: Path, width: int, height: int) -> Path:
    """Картинка с узнаваемым содержимым: градиент, чтобы было что смотреть."""
    path = directory / f"sample_{width}x{height}.png"
    if not path.exists():
        x = np.linspace(0, 255, width, dtype=np.uint8)
        y = np.linspace(0, 255, height, dtype=np.uint8)
        rgb = np.stack([np.tile(x, (height, 1)), np.tile(y[:, None], (1, width)),
                        np.full((height, width), 128, np.uint8)], axis=2)
        Image.fromarray(rgb).save(path)
    return path


def open_tab(page, index: int) -> None:
    """Открывает вкладку по номеру — и тогда, когда она в меню переполнения.

    На узком экране Gradio прячет вкладки, не влезшие в полосу, в меню «…»
    (``.overflow-menu``): с пятью вкладками на телефоне туда уходят «Галерея»
    и «Настройки». В полосе их кнопок тогда нет вовсе — они только в меню,
    в том же порядке, следом за видимыми.
    """
    tabs = page.locator('button[role="tab"]')
    shown = [tab for tab in tabs.all() if tab.is_visible()]
    if index < len(shown):
        shown[index].click()
    else:
        page.locator(".overflow-menu button").first.click()
        page.wait_for_timeout(300)
        page.locator(".overflow-dropdown button").nth(index - len(shown)).click()
    page.wait_for_timeout(700)


def fresh_page(browser, url: str, width=1920, height=1080, scale=1):
    page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=scale)
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)[:200]))
    # Упавший обработчик без очереди не показывает уведомления — только ответ
    # 500 на /gradio_api/run/predict и трассировку в консоли сервера. Так
    # прошла незамеченной ошибка выбора в галерее (KeyError: 'value').
    page.on("response", lambda response: errors.append(f"HTTP {response.status} {response.url[-40:]}")
            if response.status >= 500 else None)
    page.goto(url)
    page.wait_for_selector('button[role="tab"]', timeout=60000)
    page.wait_for_timeout(2500)
    return page, errors


def painter_state(page) -> dict:
    return page.evaluate(f"() => {API}.state()")


def load_into_painter(page, path: Path) -> dict:
    page.locator(f"#{PAINTER} input.qp-file").set_input_files(str(path))
    page.wait_for_function(f"() => {API}.state().width > 0 && !{API}.state().uploading", timeout=20000)
    return painter_state(page)


def stage_box(page) -> dict:
    return page.locator(f"#{PAINTER} .qp-stage").bounding_box()


def image_rect(page, painter: str = PAINTER) -> dict:
    """Где на экране лежит картинка внутри холста кисти."""
    return page.evaluate(f"""() => {{
        const host = [...document.querySelectorAll('#{painter} *')].find(n => n.shadowRoot);
        const r = host.shadowRoot.querySelector('.qp-image').getBoundingClientRect();
        return {{x: r.left, y: r.top, width: r.width, height: r.height}};
    }}""")


def stroke(page, points: list[tuple[float, float]], steps: int = 25, button: str = "left",
           painter: str = PAINTER) -> None:
    """Мазок по долям картинки: (0,0) — левый верхний угол, (1,1) — правый нижний."""
    rect = image_rect(page, painter)

    def to_screen(fx: float, fy: float) -> tuple[float, float]:
        return rect["x"] + rect["width"] * fx, rect["y"] + rect["height"] * fy

    page.mouse.move(*to_screen(*points[0]))
    page.mouse.down(button=button)
    for point in points[1:]:
        page.mouse.move(*to_screen(*point), steps=steps)
    page.mouse.up(button=button)
    page.wait_for_timeout(300)


def click_text(page, text: str) -> None:
    page.get_by_role("button", name=text, exact=True).first.click()


def apply_edit(page, prompt: str = "a small red boat") -> None:
    """«Применить правку» с промтом: без него правка не запускается вовсе."""
    for box in page.locator("textarea").all():
        if box.is_visible():
            if not box.input_value().strip():
                box.fill(prompt)
            break
    click_text(page, "Apply edit")


# --- сценарии ------------------------------------------------------------------

def scenario_layout(browser, url, report: Report) -> None:
    """Все вкладки на шести размерах окна: прокрутка вбок, выход за край, холст выше окна."""
    print("layout:")
    for width, height in SIZES:
        page, errors = fresh_page(browser, url, width, height)
        for tab in range(TABS):
            open_tab(page, tab)
            problems = page.evaluate("""() => {
                const doc = document.documentElement, out = [];
                if (doc.scrollWidth > doc.clientWidth + 1) out.push(`horizontal scroll ${doc.scrollWidth}>${doc.clientWidth}`);
                document.querySelectorAll('.qs-board, .qs-browse, .qs-side, .qs-canvas, .qs-refs').forEach(el => {
                    const r = el.getBoundingClientRect();
                    if (!r.width) return;
                    const name = (el.className.toString().match(/qs-[a-z-]+/) || ['?'])[0];
                    if (r.right > doc.clientWidth + 1) out.push(`${name} past the edge by ${Math.round(r.right - doc.clientWidth)}px`);
                    if ((name === 'qs-board' || name === 'qs-browse') && r.height > innerHeight) out.push(`${name} taller than the window`);
                });
                return out;
            }""")
            if tab == 1:
                # Кисть делит строку с сеткой референсов и обязана занять всё,
                # что сетке не нужно (прежний дефект: 567 px в колонке 1351).
                painter_width, free_width = page.evaluate(f"""() => {{
                    const painter = document.getElementById('{PAINTER}');
                    const row = painter.parentElement;
                    const refs = row.querySelector('.qs-refs');
                    const gap = parseFloat(getComputedStyle(row).columnGap) || 0;
                    const taken = refs ? refs.getBoundingClientRect().width + gap : 0;
                    return [painter.getBoundingClientRect().width, row.getBoundingClientRect().width - taken];
                }}""")
                if painter_width < free_width * 0.95:
                    problems.append(f"painter narrower than the free space: {painter_width:.0f} of {free_width:.0f}px")
                # В раскладке «рядом» (шире 2000 px) кисть и результат —
                # одного размера: сетка слева не отнимает ширину у кисти.
                if width > 2000:
                    result_width = page.evaluate(
                        "() => document.querySelector('.qs-slot-result .qs-board').getBoundingClientRect().width"
                    )
                    if abs(painter_width - result_width) > 2:
                        problems.append(f"painter and result differ in width: {painter_width:.0f} and {result_width:.0f}px")
            report.check(not problems, f"{width}x{height} tab {tab + 1}" + (f": {problems}" if problems else ""))
        report.check(not errors, f"{width}x{height} no page errors" + (f": {errors[:2]}" if errors else ""))
        page.close()


def scenario_language(browser, url, report: Report) -> None:
    print("language:")
    page, errors = fresh_page(browser, url)
    open_tab(page, TAB_EDIT)
    tabs = [t.inner_text() for t in page.locator('button[role="tab"]').all()]
    report.check(tabs == ["Generate", "Edit", "Video", "Gallery", "Settings"], f"English by default: {tabs}")
    page.locator("button.qs-lang").click()
    page.wait_for_timeout(2000)
    tabs = [t.inner_text() for t in page.locator('button[role="tab"]').all()]
    report.check(tabs == ["\u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f",
                          "\u0420\u0435\u0434\u0430\u043a\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435",
                          "\u0412\u0438\u0434\u0435\u043e",
                          "\u0413\u0430\u043b\u0435\u0440\u0435\u044f",
                          "\u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438"],
                 "RU button: all five tabs switched to Russian")
    hint = page.evaluate(f"""() => {{
        const host = [...document.querySelectorAll('#{PAINTER} *')].find(n => n.shadowRoot);
        return host.shadowRoot.querySelector('.qp-empty-title').textContent;
    }}""")
    report.check(hint != "Drop an image here" and bool(hint), "the painter switched to Russian too")
    title = page.evaluate("() => document.querySelector('.qs-refpose').title")
    report.check(bool(title) and not title.startswith("Pose:"), "cell icon tooltips switched to Russian too")
    page.locator("button.qs-lang").click()
    page.wait_for_timeout(2000)
    tabs = [t.inner_text() for t in page.locator('button[role="tab"]').all()]
    report.check(tabs[0] == "Generate", f"and back to English: {tabs}")
    title = page.evaluate("() => document.querySelector('.qs-refpose').title")
    report.check(title.startswith("Pose:"), f"tooltips back in English: {title!r}")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def mask_at(mask: Image.Image, fx: float, fy: float) -> int:
    array = np.asarray(mask.convert("L"))
    h, w = array.shape
    return int(array[min(int(fy * h), h - 1), min(int(fx * w), w - 1)])


def scenario_painter(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """Кисть от загрузки до маски в запросе: маска там, где её нарисовали."""
    print("painter:")
    page, errors = fresh_page(browser, url)
    open_tab(page, TAB_EDIT)
    state = load_into_painter(page, sample_image(samples, 1888, 1280))
    report.check(state["width"] == 1888 and state["height"] == 1280, f"image loaded: {state['width']}x{state['height']}")
    report.check(bool(state["source"]), "source sent to the server")

    stroke(page, [(0.3, 0.5), (0.7, 0.5)])
    report.check(painter_state(page)["strokes"] == 1, "stroke recorded in the history")

    before = len(fake.requests)
    click_text(page, "Apply edit")
    page.wait_for_timeout(1500)
    status = " ".join(box.input_value() for box in page.locator("textarea").all())
    report.check(
        len(fake.requests) == before and "Describe the edit" in status,
        "without a prompt the edit does not start and asks for a description",
    )

    apply_edit(page)
    request = fake.wait(before + 1)
    report.check(request.source is not None and request.source.size == (1888, 1280), "server received the whole source")
    # Результат — крупным просмотром, а не сеткой: в сетке единственная
    # картинка становилась квадратом выше окна, и видна была полоса кадра.
    page.wait_for_function("() => document.querySelector('.qs-slot-result .preview img')", timeout=20000)
    report.check(True, "result opened in the large preview")
    report.check(request.mask is not None, "server received the mask")
    if request.mask is not None:
        centre, corner = mask_at(request.mask, 0.5, 0.5), mask_at(request.mask, 0.05, 0.05)
        report.check(centre == 255 and corner == 0, f"mask is where it was drawn: centre {centre}, corner {corner}")
    report.check(request.mask_mode == "mask", f"mask mode: {request.mask_mode}")

    # Отмена: без пометок правится весь кадр.
    page.locator(f"#{PAINTER}").hover()
    page.keyboard.press("Control+z")
    page.wait_for_timeout(300)
    report.check(painter_state(page)["strokes"] == 0, "Ctrl+Z undoes the stroke")
    before = len(fake.requests)
    apply_edit(page)
    request = fake.wait(before + 1)
    report.check(request.mask is None and request.mask_mode == "none", f"no marks means the whole frame: {request.mask_mode}")

    # Ластик правой кнопкой: мазок кистью и поверх — стирание.
    stroke(page, [(0.2, 0.3), (0.8, 0.3)])
    stroke(page, [(0.2, 0.3), (0.8, 0.3)], button="right")
    before = len(fake.requests)
    apply_edit(page)
    request = fake.wait(before + 1)
    report.check(request.mask is None, "right button erases")

    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_annotation(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    print("annotation:")
    page, errors = fresh_page(browser, url)
    open_tab(page, TAB_EDIT)
    load_into_painter(page, sample_image(samples, 1024, 768))
    page.get_by_label("Annotation", exact=True).check()
    page.wait_for_timeout(800)
    report.check(painter_state(page)["region"] == "annotation", "painter picked up annotation mode")
    page.evaluate(f"""() => {{
        const host = [...document.querySelectorAll('#{PAINTER} *')].find(n => n.shadowRoot);
        host.shadowRoot.querySelector('.qp-swatch[data-color="#0000ff"]').click();
    }}""")
    stroke(page, [(0.4, 0.4), (0.6, 0.6)])
    before = len(fake.requests)
    apply_edit(page)
    request = fake.wait(before + 1)
    pixel = np.asarray(request.source.convert("RGB"))[int(0.5 * 768), int(0.5 * 1024)]
    report.check(request.mask is None and request.mask_mode == "annotation", f"annotation mode: {request.mask_mode}")
    report.check(pixel[2] > 200 and pixel[0] < 60, f"blue mark merged into the source: {tuple(int(v) for v in pixel)}")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_outpaint(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """«Расширить»: одна кнопка — расширенный кадр, без маски и второго шага."""
    print("outpaint and result round trip:")
    page, errors = fresh_page(browser, url)
    open_tab(page, TAB_EDIT)
    load_into_painter(page, sample_image(samples, 1024, 768))
    page.get_by_text("Outpaint", exact=True).first.click()
    page.wait_for_timeout(500)
    box = page.locator("input[aria-label='To an aspect ratio']:visible").first
    box.click()
    page.get_by_role("option", name="16:9", exact=True).first.click()
    page.wait_for_timeout(300)

    before = len(fake.requests)
    click_text(page, "Extend")
    request = fake.wait(before + 1)
    report.check(request.mask is None and request.size == request.source.size
                 and request.prompt.startswith("Outpaint the image"),
                 f"one click: no mask, the frame is the gray canvas {request.size}")
    canvas = np.asarray(request.source.convert("RGB"))
    report.check(tuple(canvas[canvas.shape[0] // 2, 3]) == (128, 128, 128),
                 "the new area goes to the model as flat gray")
    page.wait_for_function(
        "() => [...document.querySelectorAll('textarea')].some(t => t.value.includes('1366×768'))", timeout=20000,
    )
    report.check(painter_state(page)["width"] == 1024, "the painter keeps the source; the result is separate")

    page.wait_for_timeout(1000)
    old_source = painter_state(page)["source"]
    click_text(page, "Send to editor")
    page.wait_for_function(f"() => {API}.state().source !== {old_source!r}".replace("'", '"'), timeout=20000)
    report.check(painter_state(page)["width"] == 1366, "the extended picture goes back to the editor for another round")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_crop(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """Кадрирование в кисти правки: рамка мышью, обрезка, новый исходник, Ctrl+Z."""
    print("crop in the editor:")
    page, errors = fresh_page(browser, url)
    open_tab(page, TAB_EDIT)
    load_into_painter(page, sample_image(samples, 1200, 800))
    stroke(page, [(0.3, 0.3), (0.4, 0.4)])  # разметка обязана пережить обрезку
    page.locator(f"#{PAINTER} [data-act='crop']").first.click()
    page.wait_for_timeout(300)
    state = painter_state(page)
    report.check(state["crop"] == {"x": 0, "y": 0, "w": 1200, "h": 800}, f"the frame starts as the whole picture: {state['crop']}")

    # Правый нижний угол — к середине: рамка 600×400 от левого верхнего.
    handle = page.locator(f"#{PAINTER} .qp-crop-handle[data-h='se']").first.bounding_box()
    rect = image_rect(page)
    page.mouse.move(handle["x"] + handle["width"] / 2, handle["y"] + handle["height"] / 2)
    page.mouse.down()
    page.mouse.move(rect["x"] + rect["width"] * 0.5, rect["y"] + rect["height"] * 0.5, steps=15)
    page.mouse.up()
    crop = painter_state(page)["crop"]
    report.check(abs(crop["w"] - 600) <= 4 and abs(crop["h"] - 400) <= 4, f"dragging the corner resizes the frame: {crop}")
    shot = samples.parent / "crop.png"
    page.screenshot(path=str(shot))
    print(f"  screenshot: {shot}")

    page.locator(f"#{PAINTER} [data-act='cropApply']").first.click()
    page.wait_for_function(f"() => {API}.state().width < 1200 && !{API}.state().uploading && {API}.state().source", timeout=20000)
    state = painter_state(page)
    report.check(abs(state["width"] - 600) <= 4 and abs(state["height"] - 400) <= 4 and state["crop"] is None,
                 f"the picture is cropped: {state['width']}×{state['height']}")

    before = len(fake.requests)
    apply_edit(page)
    request = fake.wait(before + 1)
    report.check(request.source.size == (state["width"], state["height"]) and request.mask is not None,
                 f"the cropped picture is what the model gets, marks kept: {request.source.size}")

    page.locator(f"#{PAINTER} .qp-stage").first.hover()
    page.keyboard.press("Control+z")
    page.wait_for_function(f"() => {API}.state().width === 1200", timeout=10000)
    report.check(True, "Ctrl+Z brings the original back")

    # Пропорция: 16:9 из кадра 3:2 — наибольшая рамка по центру.
    page.evaluate(f"() => {API}.crop(null, '16:9')")
    page.wait_for_function(f"() => {API}.state().height === 675 && !{API}.state().uploading", timeout=20000)
    report.check(painter_state(page)["width"] == 1200, "a 16:9 frame takes the largest centred area")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_latency(browser, url, report: Report, samples: Path) -> None:
    """Длинный мазок на кадре 4K при плотности экрана 2x: стоимость движения не растёт.

    Тот самый замер, которым было поймано запаздывание gr.ImageEditor: у него
    на этом кадре движение дорожало по ходу мазка с 16.7 до 44 мс.
    """
    print("painter latency:")
    page, errors = fresh_page(browser, url, scale=2)
    open_tab(page, TAB_EDIT)
    load_into_painter(page, sample_image(samples, 3840, 2160))
    box = stage_box(page)
    x0, y0, w, h = box["x"], box["y"], box["width"], box["height"]
    page.mouse.move(x0 + w * 0.1, y0 + h * 0.1)
    page.mouse.down()
    durations, n = [], 2400
    for i in range(n):
        t = i / n
        row, frac = int(t * 12), (t * 12) % 1
        x = x0 + w * (0.1 + 0.8 * (frac if row % 2 == 0 else 1 - frac))
        y = y0 + h * (0.1 + 0.8 * row / 12)
        started = time.perf_counter()
        page.mouse.move(x, y)
        durations.append((time.perf_counter() - started) * 1000)
    page.mouse.up()
    quarters = [statistics.mean(durations[i * n // 4:(i + 1) * n // 4]) for i in range(4)]
    growth = quarters[3] / quarters[0]
    text = " ".join(f"{q:5.1f}" for q in quarters)
    report.check(growth < 1.3, f"move cost by stroke quarter, ms: {text} (growth {growth:.2f}x)")
    report.check(statistics.mean(durations) < 20, f"mean move {statistics.mean(durations):.1f} ms (frame budget 16.7)")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_paste(browser, url, report: Report) -> None:
    """Ctrl+V над кистью открывает картинку из буфера обмена.

    Пустой холст обещает это подсказкой, так что обещание проверяется.
    Событие создаётся из скрипта: доверенность ему не нужна, в отличие от
    движений мыши, — обработчик читает только ``clipboardData``.
    """
    print("clipboard paste:")
    page, errors = fresh_page(browser, url)
    open_tab(page, TAB_EDIT)
    page.locator(f"#{PAINTER}").hover()
    page.evaluate("""async () => {
        const canvas = document.createElement('canvas');
        canvas.width = 320; canvas.height = 200;
        canvas.getContext('2d').fillRect(0, 0, 320, 200);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
        const data = new DataTransfer();
        data.items.add(new File([blob], 'pasted.png', { type: 'image/png' }));
        document.dispatchEvent(new ClipboardEvent('paste', { clipboardData: data, bubbles: true }));
    }""")
    page.wait_for_function(f"() => {API}.state().width === 320 && !{API}.state().uploading", timeout=20000)
    state = painter_state(page)
    report.check(state["height"] == 200 and bool(state["source"]), f"image pasted and sent: {state['width']}x{state['height']}")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_gallery(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """Карточка вместо JSON и действия с переходом на нужную вкладку."""
    print("gallery:")
    from fooocus_qwen.imaging import metadata
    from fooocus_qwen.storage import gallery

    # Галерея читает каталоги дат, как их пишет генерация; дата из будущего
    # ставит образец первым, что бы ни сохранили сценарии до этого.
    target = gallery.next_path(config.OUTPUT_DIR, datetime(2099, 1, 1, 23, 59, 59))
    target.parent.mkdir(parents=True, exist_ok=True)

    metadata.save_png(Image.open(sample_image(samples, 640, 480)), target,
                      {"prompt": "a lighthouse at dawn", "seed": 777, "preset": "LowQuality",
                       "width": 640, "height": 480, "aspect": "4:3", "true_cfg_scale": 1.0})
    page, errors = fresh_page(browser, url)
    open_tab(page, TAB_GALLERY)
    page.wait_for_timeout(800)
    thumbs = page.locator(".qs-browse img")
    report.check(thumbs.count() >= 1, f"gallery refreshed on open: {thumbs.count()} images")
    # Тег на месте — ещё не картинка: сервер может отказать в файле. Так и
    # было — обновление галереи без очереди получало ссылки вида
    # /gradio_api/run/predict/gradio_api/file=… с ответом 404, и галерея
    # показывала битые значки при зелёной проверке.
    page.wait_for_timeout(1500)
    loaded = page.evaluate(
        "() => [...document.querySelectorAll('.qs-browse img')].filter(i => i.complete && i.naturalWidth > 0).length"
    )
    report.check(loaded == thumbs.count(), f"gallery images loaded: {loaded} of {thumbs.count()}")
    thumbs.first.click()
    page.wait_for_timeout(1200)
    card = page.locator(".block.qs-card").inner_text()
    report.check("a lighthouse at dawn" in card and "777" in card and "{" not in card, "the card is readable text")

    click_text(page, "Open in editor")
    page.wait_for_function(f"() => {API}.state().width === 640", timeout=20000)
    active = page.locator('button[role="tab"][aria-selected="true"]').inner_text()
    report.check(active == "Edit", f"switched to the edit tab: {active}")

    # Выбор переживает уход с вкладки: карточка на месте, кликать снова не
    # нужно. И нельзя: галерея открыта в режиме просмотра, где клик по
    # крупному кадру листает на следующий, — сценарий выбирал бы чужой файл.
    open_tab(page, TAB_GALLERY)
    card = page.locator(".block.qs-card").inner_text()
    report.check("a lighthouse at dawn" in card, "selection survived leaving the tab")
    click_text(page, "Reuse parameters")
    page.wait_for_timeout(1500)
    active = page.locator('button[role="tab"][aria-selected="true"]').inner_text()
    prompt = page.locator("textarea").first.input_value()
    report.check(active == "Generate" and prompt == "a lighthouse at dawn", f"reuse: tab {active}, prompt {prompt!r}")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_send(browser, url, report: Report, fake: FakeGenerator) -> None:
    """Генерация → «Отправить в редактор»: картинка в кисти, вкладка правки открыта."""
    print("sending a generation result to the editor:")
    page, errors = fresh_page(browser, url)
    for box in page.locator("textarea").all():
        if box.is_visible():
            box.fill("a gray square")
            break
    before = len(fake.requests)
    click_text(page, "Generate")
    fake.wait(before + 1)
    page.wait_for_function("() => document.querySelector('.preview img')", timeout=20000)
    # Подпись кнопки локализуется под язык браузера: проверяются обе. Что
    # селектор её вообще находит, проверено на голой галерее Gradio с кнопками
    # по умолчанию — там «Поделиться» есть.
    share = 'button[aria-label="Share"], button[aria-label="Поделиться"]'
    report.check(page.locator(share).count() == 0, "galleries have no share button")
    click_text(page, "Send to editor")
    page.wait_for_function(f"() => {API}.state().width === 256", timeout=20000)
    active = page.locator('button[role="tab"][aria-selected="true"]').inner_text()
    report.check(active == "Edit", f"edit tab is open: {active}")
    report.check(painter_state(page)["height"] == 256, "generated image is in the painter")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


# По ячейке: загружена ли картинка, что написано в строке тега и влезает ли
# тег в свою строку. Последнее проверяется шириной, а не текстом: подпись
# «<image1>», обрезанная до «<im», в innerText остаётся целой — так первая
# редакция сетки и прошла текстовую проверку с нечитаемыми тегами.
SLOTS_JS = """() => [...document.querySelectorAll('.qs-refcell')].filter(cell => cell.getBoundingClientRect().width > 0).map(cell => {
    const img = cell.querySelector('.qs-refslot img');
    const tag = cell.querySelector('.qs-reftag');
    // Ширина самого текста (Range), а не контейнера: у контейнера с
    // overflow: hidden scrollWidth совпадал с шириной, и обрезанная подпись
    // проходила проверку.
    const inner = tag.querySelector('code, [title]') || tag;
    const range = document.createRange();
    range.selectNodeContents(inner);
    const text = range.getBoundingClientRect(), box = tag.getBoundingClientRect();
    return {
        loaded: !!(img && img.complete && img.naturalWidth > 0),
        text: tag.innerText.trim(),
        clipped: text.width > 0 && (text.right > box.right + 1 || text.left < box.left - 1),
    };
})"""


def reference_slots(page) -> list[dict]:
    """Состояние десяти слотов: загружена ли картинка и что написано в ячейке.

    «Загружена» — не «тег img на месте»: битая ссылка тоже оставляет тег.
    Смотрится, что браузер картинку действительно получил и разобрал.
    """
    return page.evaluate(SLOTS_JS)


def wait_loaded(page, indices: list[int], timeout: int = 20000) -> None:
    """Ждёт, пока загруженными станут ровно эти слоты.

    При неудаче сообщает, что в слотах на самом деле: голый таймаут не
    говорит, картинка не пришла, пришла не туда или пришла битой.
    """
    expected = sorted(indices)
    try:
        page.wait_for_function(
            f"""() => JSON.stringify(({SLOTS_JS})().flatMap((s, i) => s.loaded ? [i] : []))
                    === '{expected}'.replace(/ /g, '')""",
            timeout=timeout,
        )
    except Exception as error:
        actual = [i for i, s in enumerate(reference_slots(page)) if s["loaded"]]
        raise AssertionError(f"expected slots {expected}, loaded {actual}") from error


def wait_label(page, index: int, fragment: str, timeout: int = 20000) -> str:
    """Ждёт подпись слота: она приходит ответом обработчика, позже картинки."""
    try:
        page.wait_for_function(
            f"() => ({SLOTS_JS})()[{index}].text.includes({fragment!r})", timeout=timeout
        )
    except Exception as error:
        raise AssertionError(
            f"slot {index} has no '{fragment}': '{reference_slots(page)[index]['text']}'"
        ) from error
    return reference_slots(page)[index]["text"]


def scenario_references(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """Сетка референсов: два столбца по пять слева от результата и в его рост;
    клик и перетаскивание.

    Проверяется вся дорога: ячейка в сетке → подпись тегом → запрос к модели.
    Подпись и запрос обязаны сходиться: тег считается по порядку заполненных
    слотов, и слоты с дырами (заполнены первый, второй и третий, но положены
    в разном порядке) — ровно тот случай, где позиционная подпись соврала бы.
    """
    print("reference grid:")
    page, errors = fresh_page(browser, url)

    # Геометрия — на трёх окнах: высота сетки привязана к полю результата,
    # а поле меняет высоту с окном (пропорция и потолок), и совпадение на
    # одном размере ещё не значит, что сетка следует за полем.
    for width, height in ((1920, 1080), (2560, 1440), (1280, 800)):
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_timeout(300)
        geometry = page.evaluate("""() => {
            const visible = el => el && el.getBoundingClientRect().width > 0;
            const refs = [...document.querySelectorAll('.qs-refs')].find(visible).getBoundingClientRect();
            const board = [...document.querySelectorAll('.qs-resultrow > .qs-board')].find(visible).getBoundingClientRect();
            const cells = [...document.querySelectorAll('.qs-refslot')]
                .map(s => s.getBoundingClientRect()).filter(r => r.width > 0);
            return {
                refs: [refs.top, refs.bottom, refs.right].map(Math.round),
                board: [board.top, board.bottom, board.left].map(Math.round),
                bottomCell: Math.round(Math.max(...cells.map(r => r.bottom))),
                rows: [...new Set(cells.map(r => Math.round(r.top)))].length,
                columns: [...new Set(cells.map(r => Math.round(r.left)))].length,
                count: cells.length,
                ratio: Math.max(...cells.map(r => Math.max(r.width / r.height, r.height / r.width))),
                cell: [Math.round(cells[0].width), Math.round(cells[0].height)],
                // Цепочка от поля картинки вверх до колонки сетки — для
                // диагноза, если ячейки не заполняют высоту.
                chain: (() => {
                    const out = [];
                    let el = [...document.querySelectorAll('.qs-refslot')].find(visible);
                    while (el && !el.classList.contains('qs-resultrow')) {
                        const s = getComputedStyle(el);
                        out.push(`${el.className.split(' ').slice(0, 3).join('.')}: ${Math.round(el.getBoundingClientRect().height)} ` +
                                 `[${s.display} ${s.flexDirection} flex=${s.flex} minh=${s.minHeight} h=${s.height} gap=${s.rowGap}]`);
                        el = el.parentElement;
                    }
                    const refsCol = [...document.querySelectorAll('.qs-refs')].find(visible);
                    for (const child of refsCol.children) {
                        const s = getComputedStyle(child);
                        out.push(`  ↳ ${child.className.split(' ').slice(0, 3).join('.')}: ${Math.round(child.getBoundingClientRect().height)} [flex=${s.flex}]`);
                    }
                    return out;
                })(),
            };
        }""")
        size = f"{width}×{height}"
        (refs_top, refs_bottom, refs_right), (board_top, board_bottom, board_left) = geometry["refs"], geometry["board"]
        report.check(geometry["count"] == 10, f"{size}: ten cells: {geometry['count']}")
        report.check(geometry["rows"] == 5 and geometry["columns"] == 2,
                     f"{size}: two columns of five: {geometry['columns']}×{geometry['rows']}")
        report.check(abs(refs_top - board_top) <= 2 and abs(refs_bottom - board_bottom) <= 2,
                     f"{size}: grid matches the result board height: {refs_top}–{refs_bottom} and {board_top}–{board_bottom}")
        report.check(geometry["bottomCell"] <= refs_bottom + 1,
                     f"{size}: cells stay inside the grid: {geometry['bottomCell']} ≤ {refs_bottom}")
        report.check(refs_right <= board_left + 1, f"{size}: grid is left of the result: {refs_right} ≤ {board_left}")
        report.check(geometry["ratio"] <= 1.5,
                     f"{size}: cells are close to square: {geometry['cell']} px, elongation {geometry['ratio']:.2f}")
        if geometry["ratio"] > 1.5:
            print("    chain:")
            for link in geometry["chain"]:
                print(f"      {link}")
    page.set_viewport_size({"width": 1920, "height": 1080})
    page.wait_for_timeout(300)
    report.check(not any(slot["loaded"] for slot in reference_slots(page)), "all empty by default")

    # Картинка кладётся прямо в третью ячейку — как это сделал бы человек,
    # перетащив файл или выбрав его кликом (оба пути ведут в тот же input).
    page.locator(".qs-refslot").nth(2).locator('input[type="file"]').set_input_files(
        str(sample_image(samples, 300, 200))
    )
    wait_loaded(page, [2])
    # Не «тега нет», а «подпись о том, что тег не нужен, есть»: отсутствие
    # тега выполняется и тогда, когда подписи нет вовсе.
    label = wait_label(page, 2, "no tag needed")
    report.check("<image" not in label, f"single reference is labelled 'tag not needed': '{label}'")
    clipped = [slot["text"] for slot in reference_slots(page) if slot["clipped"]]
    report.check(not clipped, "the 'tag not needed' label is fully visible" + (f": {clipped}" if clipped else ""))
    why = page.evaluate(
        "() => [...document.querySelectorAll('.qs-refcell')].filter(c => c.getBoundingClientRect().width > 0)[2]"
        ".querySelector('.qs-reftag [title]')?.title || ''"
    )
    report.check("in words" in why, f"the label has a tooltip with the reason: '{why[:50]}…'")

    # Результат генерации — в первый свободный слот, то есть в первый.
    for box in page.locator("textarea").all():
        if box.is_visible():
            box.fill("a gray square")
            break
    before = len(fake.requests)
    click_text(page, "Generate")
    fake.wait(before + 1)
    page.wait_for_function("() => document.querySelector('.preview img')", timeout=20000)
    click_text(page, "Send to references")
    wait_loaded(page, [0, 2])
    first, third = wait_label(page, 0, "<image1>"), wait_label(page, 2, "<image2>")
    report.check(True, f"tags follow the order of filled slots: '{first}', '{third}'")

    # С правки — снова в первый свободный, то есть во второй, и переход сюда.
    open_tab(page, TAB_EDIT)
    load_into_painter(page, sample_image(samples, 1024, 768))
    apply_edit(page)
    fake.wait(before + 2)
    page.wait_for_function("() => document.querySelector('.qs-slot-result .preview img')", timeout=20000)
    click_text(page, "Send to references")
    try:
        wait_loaded(page, [0, 1, 2])
    except AssertionError as error:
        statuses = [box.input_value() for box in page.locator(".qs-status textarea").all()]
        tab = page.locator('button[role="tab"][aria-selected="true"]').inner_text()
        slot = page.evaluate("""() => {
            const s = document.querySelectorAll('.qs-refslot')[1];
            const img = s.querySelector('img');
            return {text: s.innerText.trim(), img: img ? img.getAttribute('src') : null,
                    html: s.innerHTML.replace(/\\s+/g, ' ').slice(0, 300)};
        }""")
        raise AssertionError(f"{error}; tab '{tab}'; status lines: {statuses[-3:]}; slot 2: {slot}") from error
    active = page.locator('button[role="tab"][aria-selected="true"]').inner_text()
    report.check(active == "Generate", f"from the edit tab it switches to generation: {active}")
    tags = [wait_label(page, index, f"<image{index + 1}>") for index in (0, 1, 2)]
    report.check(True, f"tags of the three slots shifted in order: {tags}")
    clipped = [slot["text"] for slot in reference_slots(page) if slot["clipped"]]
    report.check(not clipped, "tags are fully visible, not clipped" + (f": {clipped}" if clipped else ""))
    # И на маленьком окне: ячейка там 57 px, и при прежних 12 px теги не влезали.
    page.set_viewport_size({"width": 1280, "height": 800})
    page.wait_for_timeout(400)
    clipped = [slot["text"] for slot in reference_slots(page) if slot["clipped"]]
    report.check(not clipped, "1280×800: tags are fully visible" + (f": {clipped}" if clipped else ""))
    page.set_viewport_size({"width": 1920, "height": 1080})
    page.wait_for_timeout(400)
    # Снимок заполненной сетки — глазам: текстом проверено, что подписи
    # есть, но не то, как подпись ложится поверх картинки в ячейке.
    shot = samples.parent / "references-grid.png"
    page.locator(".qs-refs").first.screenshot(path=str(shot))
    print(f"  grid screenshot: {shot}")

    # Модель получает ровно заполненные слоты и ровно в порядке тегов.
    click_text(page, "Generate")
    request = fake.wait(before + 3)
    sizes = [image.size for image in request.references]
    report.check(sizes == [(256, 256), (1024, 768), (300, 200)],
                 f"slots 1, 2, 3 went into the request in order: {sizes}")

    click_text(page, "Clear references")
    wait_loaded(page, [])
    report.check(True, "'Clear references' empties the grid")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


VISIBLE_MODAL_JS = """() => [...document.querySelectorAll('.qs-modal')]
    .find(m => m.getBoundingClientRect().width > 0 && getComputedStyle(m).display !== 'none')"""


def wait_modal(page, opened: bool, timeout: int = 20000) -> None:
    """Ждёт, пока окно откроется или закроется. Окон на странице четыре (поза и
    эскиз на двух вкладках), Gradio скрытое не рисует — смотрится видимое."""
    page.wait_for_function(f"() => !!({VISIBLE_MODAL_JS})() === {str(opened).lower()}", timeout=timeout)


def pose_tiles_loaded(page) -> int:
    return page.evaluate(
        f"() => [...(({VISIBLE_MODAL_JS})()?.querySelectorAll('.qs-posegrid img') || [])]"
        ".filter(i => i.complete && i.naturalWidth > 0).length"
    )


def wait_pose_tiles(page, count: int, timeout: int = 60000) -> None:
    page.wait_for_function(
        f"() => [...(({VISIBLE_MODAL_JS})()?.querySelectorAll('.qs-posegrid img') || [])]"
        f".filter(i => i.complete && i.naturalWidth > 0).length >= {count}",
        timeout=timeout,
    )


def modal_message(page) -> str:
    return page.evaluate(
        f"() => (({VISIBLE_MODAL_JS})()?.querySelector('.qs-modalmessage')?.innerText || '').trim()"
    )


def visible_icons(page) -> list[list[dict]]:
    """Значки видимых ячеек (открытой вкладки): лежат ли поверх картинки, есть ли подсказка."""
    return page.evaluate("""() => [...document.querySelectorAll('.qs-refcell')]
        .filter(cell => cell.getBoundingClientRect().width > 0).map(cell => {
        const slot = cell.querySelector('.qs-refslot').getBoundingClientRect();
        return [...cell.querySelectorAll('.qs-refpose, .qs-refsketch')].map(b => {
            const r = b.getBoundingClientRect();
            return {inside: r.left >= slot.left - 1 && r.right <= slot.right + 1
                            && r.top >= slot.top - 1 && r.bottom <= slot.bottom + 1,
                    title: b.title, size: Math.round(r.width)};
        });
    })""")


def scenario_tools(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """Значки ячейки: окно поз (каталог, «Добавить позу» по фото) и окно эскиза.

    Проверяется вся дорога до модели: скелет каталога, скелет с фото и эскиз
    оказываются в запросе генерации ровно в тех ячейках, куда их положили.
    Распознавание позы — настоящее (DWPose на процессоре), генератор —
    подставной: плитка новой позы — его ответ, важно лишь, что запрос пришёл
    с тем промтом и единственным референсом-скелетом. Вторая часть — те же
    ячейки на вкладке правки: сетка в рост кисти, теги со сдвигом на исходник
    и маску, референсы в запросе правки.
    """
    from fooocus_qwen.poses import library, tile

    print("pose and sketch in a cell:")
    page, errors = fresh_page(browser, url)

    icons = visible_icons(page)
    flat = [icon for cell in icons for icon in cell]
    report.check(len(icons) == 10 and all(len(cell) == 2 for cell in icons),
                 f"each of the ten cells has two icons: {[len(cell) for cell in icons]}")
    report.check(all(icon["inside"] for icon in flat), "icons lie on top of the cell image")
    report.check(all(icon["title"] for icon in flat),
                 f"icons have tooltips: '{flat[0]['title']}', '{flat[1]['title']}'")
    # На языке интерфейса: язык приложения живёт в gr.State, в скрипт он не
    # передаётся, и подсказки английского интерфейса были русскими.
    report.check(flat[0]["title"].startswith("Pose:") and flat[1]["title"].startswith("Sketch:"),
                 "tooltips are in the interface language (English)")

    # --- поза из каталога — в третью ячейку ---
    page.locator(".qs-refpose:visible").nth(2).click()
    wait_modal(page, True)
    catalog = len(library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir()))
    wait_pose_tiles(page, catalog + 1)
    report.check(pose_tiles_loaded(page) == catalog + 1,
                 f"the window has {catalog} poses and the 'Add pose' tile: {pose_tiles_loaded(page)}")
    box = page.evaluate(f"() => {{ const r = ({VISIBLE_MODAL_JS})().getBoundingClientRect();"
                        " return [r.left, r.top, r.width, r.height].map(Math.round); }")
    report.check(box[0] == 0 and box[1] == 0 and box[2] >= 1900 and box[3] >= 1070,
                 f"the window covers the whole page: {box}")
    shot = samples.parent / "pose-window.png"
    page.screenshot(path=str(shot))
    print(f"  pose window screenshot: {shot}")
    page.locator(".qs-posegrid:visible img").first.click()
    wait_loaded(page, [2])
    wait_modal(page, False)
    report.check(True, "catalog tile: skeleton in cell 3, window closed")

    # --- «Добавить позу» по фото — в первую ячейку ---
    page.locator(".qs-refpose:visible").nth(0).click()
    wait_modal(page, True)
    wait_pose_tiles(page, catalog + 1)
    page.locator(".qs-posegrid:visible img").last.click()
    page.wait_for_selector(".qs-posephoto:visible input[type=file]", state="attached", timeout=20000)
    report.check("recognis" in modal_message(page), f"photo hint: '{modal_message(page)[:60]}…'")
    photo = samples / "pose_photo.jpg"
    shutil.copy(config.POSE_LIBRARY_DIR / "dance_02.jpg", photo)
    before = len(fake.requests)
    page.locator(".qs-posephoto:visible input[type=file]").set_input_files(str(photo))
    wait_loaded(page, [0, 2], timeout=60000)
    request = fake.wait(before + 1, timeout=60)
    report.check(request.prompt.startswith(tile.PROMPT) and "Her " in request.prompt[len(tile.PROMPT):]
                 and len(request.references) == 1 and request.image_number == tile.CANDIDATES,
                 f"new pose tile requested from the model: skeleton reference {request.references[0].size}")
    page.wait_for_function(
        f"() => (({VISIBLE_MODAL_JS})()?.innerText || '').includes('Tile ready')", timeout=60000,
    )
    custom = library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir())[catalog:]
    report.check(len(custom) == 1 and custom[0].tile.exists() and custom[0].thumb.exists(),
                 f"custom pose saved with its tile: {[entry.name for entry in custom]}")
    wait_pose_tiles(page, catalog + 2, timeout=20000)
    report.check(True, "the new pose appeared in the window before 'Add pose'")
    click_text(page, "Close")
    wait_modal(page, False)

    # --- эскиз — во вторую ячейку; «Отмена» — пятая остаётся пустой ---
    page.locator(".qs-refsketch:visible").nth(1).click()
    wait_modal(page, True)
    page.wait_for_function(
        "() => (window.__qsPainters || {})['qs-sketch-painter']?.state().width > 0", timeout=20000
    )
    shot = samples.parent / "sketch-window.png"
    stroke(page, [(0.2, 0.2), (0.8, 0.8)], painter="qs-sketch-painter")
    stroke(page, [(0.2, 0.8), (0.8, 0.2)], painter="qs-sketch-painter")
    page.screenshot(path=str(shot))
    print(f"  sketch window screenshot: {shot}")
    click_text(page, "Accept")
    wait_loaded(page, [0, 1, 2])
    wait_modal(page, False)
    report.check(True, "'Accept': sketch in cell 2, window closed")

    page.locator(".qs-refsketch:visible").nth(4).click()
    wait_modal(page, True)
    click_text(page, "Cancel")
    wait_modal(page, False)
    report.check(not reference_slots(page)[4]["loaded"], "'Cancel' closes the window, cell 5 is empty")

    # --- модель получает ровно это и в этом порядке ---
    for box in page.locator("textarea").all():
        if box.is_visible():
            box.fill("a dancer <image1>")
            break
    count = len(fake.requests)
    click_text(page, "Generate")
    request = fake.wait(count + 1)
    sizes = [image.size for image in request.references]
    report.check(sizes == [(768, 768), (1024, 1024), (768, 768)],
                 f"the request got the photo pose, the sketch, the catalog pose: {sizes}")
    pose_img, sketch_img = np.asarray(request.references[0]), np.asarray(request.references[1].convert("L"))
    report.check(pose_img.mean() < 40, f"pose is a skeleton on black: mean brightness {pose_img.mean():.0f}")
    centre, corner = sketch_img[500:524, 500:524].mean(), sketch_img[40:80, 900:980].mean()
    report.check(centre < 128 < corner,
                 f"sketch: stroke in the centre is dark ({centre:.0f}), canvas is white ({corner:.0f})")

    # --- вкладка правки: своя сетка слева от кисти ---
    open_tab(page, TAB_EDIT)
    load_into_painter(page, sample_image(samples, 1024, 768))
    stroke(page, [(0.6, 0.6), (0.8, 0.8)])
    geometry = page.evaluate(f"""() => {{
        const painter = document.getElementById('{PAINTER}').getBoundingClientRect();
        const refs = [...document.querySelectorAll('.qs-refs')].find(r => r.getBoundingClientRect().width > 0)
            .getBoundingClientRect();
        return {{refs: [refs.top, refs.bottom, refs.right].map(Math.round),
                 painter: [painter.top, painter.bottom, painter.left].map(Math.round)}};
    }}""")
    (refs_top, refs_bottom, refs_right), (painter_top, painter_bottom, painter_left) = (
        geometry["refs"], geometry["painter"])
    report.check(abs(refs_top - painter_top) <= 2 and abs(refs_bottom - painter_bottom) <= 2,
                 f"edit: grid matches the painter height: {refs_top}–{refs_bottom} and {painter_top}–{painter_bottom}")
    report.check(refs_right <= painter_left + 1, f"edit: grid is left of the painter: {refs_right} ≤ {painter_left}")
    icons = visible_icons(page)
    report.check(len(icons) == 10 and all(icon["title"] for cell in icons for icon in cell),
                 "edit: ten cells, icons have tooltips (tab rendered after loading)")
    report.check(not any(slot["loaded"] for slot in reference_slots(page)),
                 "edit: its own grid, generation references did not leak here")

    page.locator(".qs-refpose:visible").nth(0).click()
    wait_modal(page, True)
    wait_pose_tiles(page, catalog + 2)
    page.locator(".qs-posegrid:visible img").first.click()
    wait_loaded(page, [0])
    wait_modal(page, False)
    tag = wait_label(page, 0, "<image3>")
    report.check(True, f"edit, 'Mask' mode: source and mask come first, reference is '{tag}'")
    page.get_by_label("No region — edit the whole frame", exact=True).check()
    tag = wait_label(page, 0, "<image2>")
    report.check(True, f"edit, 'No area': no mask, reference is '{tag}'")
    page.get_by_label("Mask", exact=True).check()
    wait_label(page, 0, "<image3>")

    count = len(fake.requests)
    apply_edit(page, "put the dancer from <image3> into the marked area")
    request = fake.wait(count + 1)
    sizes = [image.size for image in request.references]
    report.check(sizes == [(768, 768)] and request.mask is not None,
                 f"edit: the request got the mask and the pose reference: {sizes}")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))


VIEWER = "window.__qsViewer"


def viewer_state(page) -> dict:
    return page.evaluate(f"() => {VIEWER} ? {VIEWER}.state() : null")


def wait_viewer(page, opened: bool, timeout: int = 10000) -> None:
    page.wait_for_function(f"() => {VIEWER} && {VIEWER}.state().open === {str(opened).lower()}", timeout=timeout)


def scenario_viewer(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """Просмотр в полном размере: результат, референс, результат правки, галерея.

    Клик по большой картинке результата раньше листал галерею, по картинке в
    ячейке референса — открывал выбор файла. Теперь оба открывают просмотр;
    окно выбора файла при этом открываться не должно.
    """
    print("full-size viewer:")
    page, errors = fresh_page(browser, url)
    choosers: list = []
    page.on("filechooser", lambda chooser: choosers.append(chooser))

    for box in page.locator("textarea").all():
        if box.is_visible():
            box.fill("a gray square")
            break
    before = len(fake.requests)
    click_text(page, "Generate")
    fake.wait(before + 1)
    page.wait_for_function("() => document.querySelector('.qs-board .preview .media-button img')", timeout=20000)
    page.wait_for_timeout(500)
    page.locator(".qs-board .preview .media-button img").first.click()
    wait_viewer(page, True)
    state = viewer_state(page)
    report.check(state["mode"] == "fit" and state["natural"] == [256, 256] and state["count"] == 1,
                 f"result click opens the viewer, fitted, full file: {state['natural']}")
    page.locator(".qv-image").click()
    page.wait_for_function(f"() => {VIEWER}.state().mode === 'actual'", timeout=5000)
    page.locator(".qv-image").click()
    page.wait_for_function(f"() => {VIEWER}.state().mode === 'fit'", timeout=5000)
    report.check(True, "click on the image toggles 100% and fit")
    shot = samples.parent / "viewer.png"
    page.screenshot(path=str(shot))
    print(f"  viewer screenshot: {shot}")
    page.keyboard.press("Escape")
    wait_viewer(page, False)
    report.check(page.locator(".qs-board .preview .media-button img").count() == 1,
                 "Esc closes it; the result stays in preview (the click did not flip the gallery)")

    # Референс: просмотр вместо выбора файла.
    page.locator(".qs-refslot").nth(0).locator('input[type="file"]').set_input_files(
        str(sample_image(samples, 300, 200)))
    wait_loaded(page, [0])
    choosers.clear()
    page.locator(".qs-refslot img").first.click()
    wait_viewer(page, True)
    page.wait_for_timeout(500)
    report.check(viewer_state(page)["natural"] == [300, 200] and not choosers,
                 f"reference click opens the viewer, no file dialog: {viewer_state(page)['natural']}, dialogs {len(choosers)}")
    page.mouse.click(8, 400)  # фон
    wait_viewer(page, False)
    report.check(True, "clicking the backdrop closes it")

    # Результат правки.
    open_tab(page, TAB_EDIT)
    load_into_painter(page, sample_image(samples, 640, 480))
    before = len(fake.requests)
    apply_edit(page)
    fake.wait(before + 1)
    page.wait_for_function("() => document.querySelector('.qs-slot-result .preview .media-button img')", timeout=20000)
    page.wait_for_timeout(500)
    page.locator(".qs-slot-result .preview .media-button img").first.click()
    wait_viewer(page, True)
    report.check(viewer_state(page)["natural"] == [640, 480], f"edit result opens too: {viewer_state(page)['natural']}")
    page.locator(".qv-close").click()
    wait_viewer(page, False)

    # Галерея: одиночный клик — карточка, двойной — просмотр.
    open_tab(page, TAB_GALLERY)
    page.wait_for_timeout(1200)
    thumb = page.locator(".qs-browse img").first
    thumb.click()
    page.wait_for_timeout(600)
    report.check(not viewer_state(page)["open"], "gallery: a single click still opens the card, not the viewer")
    thumb.dblclick()
    wait_viewer(page, True)
    distinct = page.evaluate(
        "() => new Set([...document.querySelectorAll('.qs-browse .thumbnail-item img')].map(i => i.src)).size")
    report.check(viewer_state(page)["count"] == distinct > 0,
                 f"gallery: double click opens the viewer, each image listed once: "
                 f"{viewer_state(page)['count']} of {distinct}")
    page.keyboard.press("Escape")
    wait_viewer(page, False)
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


SKETCH_HOST_JS = """(id) => [...document.querySelectorAll('#' + id + ' *')].find(n => n.shadowRoot).shadowRoot"""


def painter_controls(page, painter_id: str) -> dict:
    """Видны ли у кисти свой цвет и непрозрачность."""
    return page.evaluate(f"""(id) => {{
        const root = ({SKETCH_HOST_JS})(id);
        return {{colour: !root.querySelector('.qp-colour').hidden, alpha: !root.querySelector('.qp-alpha').hidden}};
    }}""", painter_id)


def scenario_sketch_colour(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """Эскиз: свой цвет и полупрозрачная кисть; у маски правки их нет.

    Мазок идёт туда и обратно по одному месту: полупрозрачный мазок обязан
    лечь ровно — на стыках своих отрезков и на самоперекрытии не темнеть.
    """
    print("sketch colour and opacity:")
    page, errors = fresh_page(browser, url)
    page.locator(".qs-refsketch:visible").nth(0).click()
    page.wait_for_function(
        "() => (window.__qsPainters || {})['qs-sketch-painter']?.state().width > 0", timeout=20000
    )
    controls = painter_controls(page, "qs-sketch-painter")
    report.check(controls == {"colour": True, "alpha": True}, f"sketch has a colour picker and opacity: {controls}")
    page.evaluate(f"""(id) => {{
        const root = ({SKETCH_HOST_JS})(id);
        const colour = root.querySelector('.qp-colour-input');
        colour.value = '#ff0000';
        colour.dispatchEvent(new Event('input', {{bubbles: true}}));
        const alpha = root.querySelector('.qp-alpha-range');
        alpha.value = '50';
        alpha.dispatchEvent(new Event('input', {{bubbles: true}}));
    }}""", "qs-sketch-painter")
    state = page.evaluate("() => window.__qsPainters['qs-sketch-painter'].state()")
    report.check(state["color"] == "#ff0000" and state["alpha"] == 0.5,
                 f"custom colour and 50% opacity chosen: {state['color']}, {state['alpha']}")
    stroke(page, [(0.2, 0.5), (0.8, 0.5), (0.2, 0.5)], painter="qs-sketch-painter")
    click_text(page, "Accept")
    wait_loaded(page, [0])

    for box in page.locator("textarea").all():
        if box.is_visible():
            box.fill("a sketch")
            break
    count = len(fake.requests)
    click_text(page, "Generate")
    request = fake.wait(count + 1)
    pixels = np.asarray(request.references[0].convert("RGB")).astype(int)
    middle, corner = pixels[512, 512], pixels[100, 100]
    report.check(abs(middle[0] - 255) <= 6 and all(abs(c - 128) <= 12 for c in middle[1:]),
                 f"50% red over white, even where the stroke overlaps itself: {middle.tolist()}")
    report.check(corner.tolist() == [255, 255, 255], f"canvas stays white: {corner.tolist()}")

    # У кисти маски на правке — ни своего цвета, ни прозрачности.
    open_tab(page, TAB_EDIT)
    load_into_painter(page, sample_image(samples, 320, 240))
    page.get_by_label("Annotation", exact=True).check()
    page.wait_for_timeout(500)
    controls = painter_controls(page, PAINTER)
    report.check(controls == {"colour": False, "alpha": False},
                 f"the edit brush (annotation mode) has neither: {controls}")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def visible_input(page, label: str):
    for item in page.get_by_label(label, exact=True).all():
        if item.is_visible():
            return item
    raise LookupError(f"no visible field {label!r}")


def pose_grid_text(page) -> str:
    return page.evaluate(
        f"() => (({VISIBLE_MODAL_JS})()?.querySelector('.qs-posegrid')?.innerText || '')"
    )


def scenario_pose_edit(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """Имена поз и карандаш на плитках: правка имени и перерисовка обложки."""
    from fooocus_qwen.poses import library, tile

    print("pose names and the edit pencil:")
    page, errors = fresh_page(browser, url)
    page.locator(".qs-refpose:visible").nth(0).click()
    wait_modal(page, True)
    total = len(library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir()))
    wait_pose_tiles(page, total + 1)
    page.wait_for_function(
        f"() => ({VISIBLE_MODAL_JS})().querySelectorAll('.qs-poseedit').length === {total}", timeout=10000,
    )
    last_has_pen = page.evaluate(f"""() => {{
        const items = [...({VISIBLE_MODAL_JS})().querySelectorAll('.qs-posegrid .thumbnail-item')];
        return !!items[items.length - 1].querySelector('.qs-poseedit');
    }}""")
    report.check(not last_has_pen, f"a pencil on each of {total} poses, none on “Add a pose”")
    text = pose_grid_text(page)
    report.check("Dance 1" in text and "Standing 1" in text, "tiles are captioned with default names")
    shot = samples.parent / "pose-edit.png"
    page.screenshot(path=str(shot))
    print(f"  screenshot: {shot}")

    # Карандаш: правка, ячейки не трогаются.
    page.locator(".qs-posegrid:visible .qs-poseedit").nth(1).click()
    page.wait_for_function(
        f"() => (({VISIBLE_MODAL_JS})()?.innerText || '').includes('Pose “Dance 2”')", timeout=10000,
    )
    page.screenshot(path=str(samples.parent / "pose-edit-panel.png"))
    name = visible_input(page, "Pose name")
    report.check(name.input_value() == "Dance 2" and not reference_slots(page)[0]["loaded"],
                 f"the pencil opens the edit panel, name {name.input_value()!r}, cells untouched")
    name.fill("Hands up")
    click_text(page, "Save name")
    page.wait_for_function(
        f"() => (({VISIBLE_MODAL_JS})()?.innerText || '').includes('Name saved')", timeout=10000,
    )
    report.check("Hands up" in pose_grid_text(page), "the new name is under the tile")
    titles = library.load_titles(config.user_pose_dir())
    report.check(titles.get("dance_02") == "Hands up", f"stored with the user's data: {titles}")

    # Схематичный вид: скелет вместо обложки в окне и в правке.
    def tile_src(index: int) -> str:
        return page.evaluate(
            f"() => [...({VISIBLE_MODAL_JS})().querySelectorAll('.qs-posegrid img')][{index}]?.src || ''"
        )

    schematic = visible_input(page, "Schematic view")
    report.check(not schematic.is_checked(), "a catalogue pose is not schematic by default")
    schematic.check()
    page.wait_for_function(
        f"() => (({VISIBLE_MODAL_JS})()?.innerText || '').includes('shown as a skeleton')", timeout=10000,
    )
    page.wait_for_function(
        f"() => ([...({VISIBLE_MODAL_JS})().querySelectorAll('.qs-posegrid img')][1]?.src || '')"
        ".includes('dance_02.png')", timeout=10000,
    )
    report.check("dance_02" in library.load_schematic(config.user_pose_dir()),
                 f"“Schematic view” shows the skeleton on the tile: {tile_src(1)[-40:]!r}")

    before = len(fake.requests)
    click_text(page, "Redraw cover")
    request = fake.wait(before + 1)
    page.wait_for_function(
        f"() => (({VISIBLE_MODAL_JS})()?.innerText || '').includes('Cover redrawn')", timeout=20000,
    )
    cover = config.user_pose_dir() / library.META_DIR / library.COVERS_DIR / "dance_02.thumb.jpg"
    report.check(request.prompt.startswith(tile.PROMPT) and cover.exists(),
                 "“Redraw cover” asks the model and stores the cover over the catalogue")
    page.wait_for_timeout(300)
    report.check(not schematic.is_checked() and "dance_02" not in library.load_schematic(config.user_pose_dir())
                 and "dance_02.thumb" in tile_src(1),
                 "a redrawn cover lifts the schematic view")

    click_text(page, "Back to poses")
    page.wait_for_timeout(400)
    page.locator(".qs-posegrid:visible img").nth(1).click()
    wait_loaded(page, [0])
    wait_modal(page, False)
    report.check(True, "after editing, a plain click on the same tile still picks it")

    # Новая поза с именем.
    page.locator(".qs-refpose:visible").nth(2).click()
    wait_modal(page, True)
    wait_pose_tiles(page, total + 1)
    page.locator(".qs-posegrid:visible img").last.click()
    page.wait_for_selector(".qs-posephoto:visible input[type=file]", state="attached", timeout=20000)
    visible_input(page, "Pose name").fill("Victory")
    photo = samples / "pose_named.jpg"
    shutil.copy(config.POSE_LIBRARY_DIR / "tpose_01.jpg", photo)
    page.locator(".qs-posephoto:visible input[type=file]").set_input_files(str(photo))
    wait_loaded(page, [0, 2], timeout=60000)
    page.wait_for_function(
        f"() => (({VISIBLE_MODAL_JS})()?.querySelector('.qs-posegrid')?.innerText || '').includes('Victory')",
        timeout=60000,
    )
    report.check(True, "a pose added with a name shows it under its tile")

    # Удаление: сначала вопрос, потом поза уходит с диска и из окна.
    victory = next(entry for entry in library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir())
                   if entry.title == "Victory")
    page.wait_for_timeout(1500)  # плитка ещё рисуется: подделка отвечает быстро, но ответ идёт очередью
    page.locator(".qs-posegrid:visible .qs-poseedit").nth(total).click()
    page.wait_for_function(
        f"() => (({VISIBLE_MODAL_JS})()?.innerText || '').includes('Pose “Victory”')", timeout=10000,
    )
    click_text(page, "Delete pose")
    page.wait_for_function(
        f"() => (({VISIBLE_MODAL_JS})()?.innerText || '').includes('cannot be undone')", timeout=10000,
    )
    page.screenshot(path=str(samples.parent / "pose-delete.png"))
    report.check(victory.keypoints.exists(), "“Delete pose” asks first and deletes nothing yet")
    click_text(page, "Yes, delete")
    page.wait_for_function(
        f"() => (({VISIBLE_MODAL_JS})()?.innerText || '').includes('Pose “Victory” deleted')", timeout=10000,
    )
    left = list(victory.folder.glob(f"{victory.name}*"))
    report.check(not left and "Victory" not in pose_grid_text(page),
                 f"after confirming, the pose is gone from the disk and the window: {left}")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def choices_are_stacked(page, texts) -> bool:
    """Варианты стоят столбцом: левый край общий, каждый ниже предыдущего."""
    boxes = [visible_label_box(page, text) for text in texts]
    if None in boxes or len(boxes) < 2:
        return False
    lefts = {round(box["x"]) for box in boxes}
    tops = [box["y"] for box in boxes]
    return len(lefts) == 1 and tops == sorted(tops) and len(set(tops)) == len(tops)


def visible_label_box(page, text: str):
    for item in page.get_by_label(text, exact=True).all():
        if item.is_visible():
            return item.locator("xpath=ancestor::label[1]").bounding_box()
    return None


def make_loras(directory: Path) -> Path:
    """Две крошечные LoRA: годная для 2.1 (с триггером) и от Qwen-Image 1."""
    import torch
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    good = "transformer.transformer_blocks.0.attn.to_q"
    save_file({f"{good}.lora_A.weight": torch.zeros(4, 4096), f"{good}.lora_B.weight": torch.zeros(4096, 4)},
              str(directory / "zzz-style.safetensors"), metadata={"trigger_word": "zenlesszonezero"})
    old = "transformer.transformer_blocks.0.attn.add_q_proj"
    save_file({f"{old}.lora_A.weight": torch.zeros(4, 3072), f"{old}.lora_B.weight": torch.zeros(3072, 4)},
              str(directory / "qwen1-style.safetensors"))
    return directory


def pick_lora(page, slot: int, name: str) -> None:
    box = page.locator(f"input[aria-label='LoRA {slot}']:visible").first
    box.click()
    page.get_by_role("option", name=name, exact=True).first.click()
    page.wait_for_timeout(400)


def scenario_loras(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """Ячейки LoRA: выбор файла включает ячейку, строка о файле, чужая LoRA пропускается."""
    print("LoRA slots:")
    page, errors = fresh_page(browser, url)
    page.get_by_text("LoRA", exact=True).first.click()
    page.wait_for_timeout(500)
    pick_lora(page, 1, "zzz-style")
    page.wait_for_function("() => document.body.innerText.includes('rank 4 · 1 layers')", timeout=10000)
    report.check(visible_input(page, "On").is_checked(), "picking a file ticks the slot")
    report.check("zenlesszonezero" in page.inner_text("body"), "trigger words are shown under the slot")
    pick_lora(page, 2, "qwen1-style")
    page.wait_for_function("() => document.body.innerText.includes('Not for Qwen-Image 2.1')", timeout=10000)
    report.check(True, "a Qwen-Image 1 LoRA is flagged before generating")
    weight = page.locator(".qs-lora:visible input[type=number]").first
    weight.fill("0.75")
    weight.press("Enter")
    page.wait_for_timeout(500)
    shot = samples.parent / "loras.png"
    page.screenshot(path=str(shot))
    print(f"  screenshot: {shot}")

    before = len(fake.requests)
    page.locator("textarea:visible").first.fill("a cat")
    click_text(page, "Generate")
    request = fake.wait(before + 1)
    names = [(item.name, item.weight) for item in request.loras]
    page.wait_for_timeout(800)
    status = " ".join(box.input_value() for box in page.locator("textarea").all())
    report.check(names == [("zzz-style", 0.75)], f"only the fitting LoRA goes to the model, with its weight: {names}")
    report.check("“qwen1-style” skipped" in status, "the status line names the skipped LoRA")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_tall(browser, url, report: Report, samples: Path) -> None:
    """Высокие окна, вкладка правки: кисть и сетка кончаются над строкой промта.

    В раскладке столбцом высота строки «сетка + кисть» раньше выводилась по
    кругу, и на 1920×1500 кисть уезжала под промт (снимок для README).
    """
    print("tall windows on the Edit tab:")
    for width, height in ((1920, 1500), (1280, 1440), (1920, 1080)):
        page, errors = fresh_page(browser, url, width, height)
        open_tab(page, TAB_EDIT)
        load_into_painter(page, sample_image(samples, 1024, 1024))
        page.wait_for_timeout(500)
        boxes = page.evaluate("""() => {
            const r = s => { const n = [...document.querySelectorAll(s)].find(e => e.offsetParent); return n && n.getBoundingClientRect(); };
            return {bar: r('.qs-bar').top, row: r('.qs-slot-editor > .qs-resultrow').bottom, refs: r('.qs-refs').bottom};
        }""")
        report.check(boxes["row"] <= boxes["bar"] and boxes["refs"] <= boxes["bar"],
                     f"{width}×{height}: brush and grid end above the prompt ({boxes})")
        report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
        page.close()


def scenario_quality(browser, url, report: Report, samples: Path) -> None:
    """«Качество и скорость» на генерации и правке: столбцом и словами, значение — имя пресета."""
    from fooocus_qwen.ui import quality

    print("quality choices:")
    page, errors = fresh_page(browser, url)
    texts = [text for text, _name in quality.choices("en")]
    report.check(choices_are_stacked(page, texts), "Generate: quality options stand in a column, fast to best")
    report.check(visible_input(page, quality.label(config.AppConfig().preset, "en")).is_checked(),
                 "the default preset is selected")
    shot = samples.parent / "quality.png"
    page.screenshot(path=str(shot))
    print(f"  screenshot: {shot}")
    open_tab(page, TAB_EDIT)
    page.wait_for_timeout(500)
    report.check(choices_are_stacked(page, texts), "Edit: quality options stand in a column too")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_performance(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """Секция «Производительность»: состояние, точность, SageAttention на лету."""
    print("performance:")
    from fooocus_qwen import settings
    from fooocus_qwen.engine import attention

    page, errors = fresh_page(browser, url)
    open_tab(page, TAB_SETTINGS)
    page.wait_for_timeout(800)
    status = " ".join(box.input_value() for box in page.locator("textarea").all())
    report.check("Precision: BF16" in status and "Memory profile:" in status,
                 "status shows precision, memory profile and Turbo")
    # Точность — столбец радиокнопок (bf16, INT8 и шесть GGUF), подписи из
    # общего каталога; подходящая карте помечена.
    from fooocus_qwen.engine import hardware

    labels = {name: settings.precision_label(name, "en", recommended=name == settings.recommended_precision(
        hardware.vram_gib())) for name in settings.PRECISIONS}
    report.check(page.get_by_label(labels["bf16"], exact=True).is_checked(),
                 f"current precision is selected: {labels['bf16']!r}")
    offered = [name for name, text in labels.items() if page.get_by_label(text, exact=True).count() == 1]
    report.check(offered == list(settings.PRECISIONS), f"all precisions are offered, one per line: {offered}")
    report.check(page.get_by_label("Auto — by the video card", exact=False).first.is_checked(),
                 "memory profile Auto is selected")
    report.check(choices_are_stacked(page, labels.values()), "precision options stand in a column")
    page.screenshot(path=str(samples.parent / "settings.png"))

    sage = page.get_by_label("SageAttention — fast attention")
    if not attention.sage_available():
        report.check(sage.is_disabled(), "without the package the SageAttention checkbox is disabled")
    else:
        sage.check()
        # Первое включение подгружает модуль трансформера diffusers — секунды.
        try:
            page.wait_for_function(
                "() => [...document.querySelectorAll('textarea')].some(t => t.value.includes('Attention: SageAttention'))",
                timeout=30000,
            )
        except Exception:  # noqa: BLE001 — проверка ниже скажет, что не так
            pass
        status = " ".join(box.input_value() for box in page.locator("textarea").all())
        report.check(settings.load().sage_attention and "Attention: SageAttention" in status,
                     f"SageAttention turned on and saved: {getattr(fake, 'attention', None)}; {status[-120:]!r}")
        sage.uncheck()
        try:
            page.wait_for_function(
                "() => [...document.querySelectorAll('textarea')].some(t => t.value.includes('Attention: standard'))",
                timeout=30000,
            )
        except Exception:  # noqa: BLE001
            pass
        report.check(not settings.load().sage_attention and fake.attention == "native",
                     "and turned back off")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_secret(browser, url, report: Report) -> None:
    """Токен языковой модели не попадает на страницу ни в каком виде."""
    print("secret:")
    config.ENDPOINT_FILE.write_text(
        "\n".join(["llama.cpp", "192.0.2.10:8000", "token=ui-check-secret", ""]), encoding="utf-8"
    )
    page, errors = fresh_page(browser, url)
    open_tab(page, TAB_SETTINGS)
    page.wait_for_timeout(800)
    html = page.content()
    fields = " ".join(t.input_value() for t in page.locator("textarea, input").all() if t.is_visible())
    report.check("ui-check-secret" not in html and "ui-check-secret" not in fields, "the token is neither in the markup nor in the fields")
    report.check("192.0.2.10" in fields, "while the address is visible")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


# --- запуск --------------------------------------------------------------------

# --- форматы на входе: HEIC, TIFF, SVG ------------------------------------------

# Chrome под Windows отдаёт .heic с пустым типом — так и кладётся файл в
# сценарии: Playwright по расширению подставил бы image/heic, и опыт проверял
# бы не тот случай, на котором всё сломалось.
DROP_JS = """async ([selector, index, name, base64, type]) => {
    const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0));
    const target = [...document.querySelectorAll(selector)].filter(e => e.getBoundingClientRect().width > 0)[index];
    const box = target.getBoundingClientRect();
    const x = box.left + box.width / 2, y = box.top + box.height / 2;
    // Цель — то, что под курсором, как при настоящем перетаскивании.
    let hit = document.elementFromPoint(x, y);
    while (hit && hit.shadowRoot && hit.shadowRoot.elementFromPoint(x, y) && hit.shadowRoot.elementFromPoint(x, y) !== hit) {
        hit = hit.shadowRoot.elementFromPoint(x, y);
    }
    const data = new DataTransfer();
    data.items.add(new File([bytes], name, { type }));
    for (const kind of ['dragenter', 'dragover', 'drop']) {
        hit.dispatchEvent(new DragEvent(kind, { bubbles: true, cancelable: true, composed: true, clientX: x, clientY: y, dataTransfer: data }));
    }
}"""

# Контроль: страница без перехватчика — его capture-обработчики на window не
# регистрируются. Так видно, что без перевода HEIC действительно терялся, и
# зелёный результат — заслуга перевода, а не опыта, который ничего не ловит.
NO_IMPORT_JS = """(() => {
    const original = window.addEventListener;
    window.addEventListener = function (type, listener, options) {
        if (this === window && (type === 'drop' || type === 'change') && options === true) return;
        return original.call(this, type, listener, options);
    };
})()"""


def heic_sample(samples: Path, width: int = 300, height: int = 200) -> Path:
    from fooocus_qwen.imaging import formats

    formats.register()
    path = samples / f"IMG_{width}x{height}.HEIC"
    Image.open(sample_image(samples, width, height)).save(path, format="HEIF", quality=90)
    return path


def drop_file(page, selector: str, index: int, path: Path, mime: str = "") -> None:
    import base64

    page.evaluate(DROP_JS, [selector, index, path.name, base64.b64encode(path.read_bytes()).decode(), mime])


def toast(page) -> dict:
    return page.evaluate("""() => {
        const t = document.querySelector('.fq-import-toast');
        return t ? { text: t.textContent, visible: t.dataset.visible === 'true', error: t.dataset.error === 'true' } : {};
    }""")


def scenario_formats(browser, url, report: Report, samples: Path) -> None:
    """HEIC, TIFF и SVG доходят до ячейки референса, кисти правки и окна поз.

    Дорога та же, что у человека: перетаскивание (файл без MIME-типа, как
    его отдаёт Chrome под Windows) и выбор в диалоге. Проверяется, что
    браузер картинку получил и разобрал, а не что тег на месте.
    """
    print("input formats:")
    heic = heic_sample(samples)
    tiff = samples / "scan16.tiff"
    sixteen = (np.tile(np.linspace(0, 65535, 240, dtype=np.uint16), (160, 1)))
    Image.fromarray(sixteen).save(tiff)
    svg = samples / "logo.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100">'
                   '<rect width="200" height="100" fill="#3a7"/></svg>', encoding="utf-8")
    broken = samples / "broken.heic"
    broken.write_bytes(b"not a picture" * 50)

    # Контроль: без перехватчика перетащенный HEIC ячейка молча теряет.
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.add_init_script(NO_IMPORT_JS)
    page.goto(url)
    page.wait_for_selector('button[role="tab"]', timeout=60000)
    page.wait_for_timeout(2500)
    drop_file(page, ".qs-refslot", 0, heic)
    page.wait_for_timeout(3000)
    report.check(not reference_slots(page)[0]["loaded"],
                 "control: without the importer a dropped HEIC is lost (the original bug)")
    page.close()

    page, errors = fresh_page(browser, url)
    report.check(page.evaluate("() => Boolean(window.__fqImport)"), "importer is installed on the page")
    accept = page.evaluate("() => document.querySelector('.qs-refslot input[type=file]').accept")
    report.check(".heic" in accept and ".cr3" in accept, f"file dialog lists HEIC and RAW: {accept[:40]}…")

    drop_file(page, ".qs-refslot", 0, heic)
    wait_loaded(page, [0])
    report.check(True, "dropped HEIC (no MIME type) shows in reference cell 1")
    page.locator(".qs-refslot").nth(1).locator('input[type="file"]').set_input_files(
        {"name": heic.name, "mimeType": "", "buffer": heic.read_bytes()}
    )
    wait_loaded(page, [0, 1])
    report.check(True, "HEIC picked in the file dialog shows in reference cell 2")
    drop_file(page, ".qs-refslot", 2, svg, "image/svg+xml")
    wait_loaded(page, [0, 1, 2])
    report.check(True, "dropped SVG is rasterized into reference cell 3")
    natural = page.evaluate("""() => [...document.querySelectorAll('.qs-refslot img')]
        .filter(i => i.getBoundingClientRect().width > 0).map(i => [i.naturalWidth, i.naturalHeight])""")
    report.check(natural[0] == [300, 200] and natural[2][0] == 2048,
                 f"sizes kept: HEIC {natural[0]}, SVG raster {natural[2]}")

    drop_file(page, ".qs-refslot", 3, broken)
    page.wait_for_function("() => document.querySelector('.fq-import-toast')?.dataset.error === 'true'", timeout=20000)
    message = toast(page)
    report.check("broken.heic" in message["text"] and "cannot read" in message["text"],
                 f"broken file: a clear message, not silence: '{message['text'][:70]}'")
    report.check(not reference_slots(page)[3]["loaded"], "broken file: the cell stays empty")

    # Кисть правки: перетаскивание в теневой DOM и выбор файла там же.
    open_tab(page, TAB_EDIT)
    drop_file(page, f"#{PAINTER}", 0, heic)
    page.wait_for_function(f"() => {API}.state().width === 300 && !{API}.state().uploading", timeout=30000)
    state = painter_state(page)
    report.check(state["height"] == 200 and str(state["source"]).endswith(".png"),
                 f"dropped HEIC opens in the edit brush and uploads as PNG: {state['width']}x{state['height']}")
    page.locator(f"#{PAINTER} input.qp-file").set_input_files(
        {"name": tiff.name, "mimeType": "image/tiff", "buffer": tiff.read_bytes()}
    )
    page.wait_for_function(f"() => {API}.state().width === 240 && !{API}.state().uploading", timeout=30000)
    report.check(painter_state(page)["height"] == 160, "16-bit TIFF picked in the edit brush opens")

    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def saved_field(page):
    """Поле сохранённых промтов на видимой вкладке."""
    return page.locator(".qs-libraryfield input:visible").first


def pick_saved_prompt(page, name: str) -> None:
    """Выбор в поле сохранённых промтов; щелчок по полю освежает список."""
    saved_field(page).click()
    page.wait_for_timeout(500)
    page.get_by_role("option", name=name, exact=True).first.click()
    page.wait_for_timeout(800)


def rows_follow(page) -> dict:
    """Где строка промтов относительно поля промта и строки действий (видимая вкладка)."""
    return page.evaluate("""() => {
        const vis = s => [...document.querySelectorAll(s)].find(e => e.offsetParent);
        const lib = vis('.qs-library');
        const bar = vis('.qs-bar').getBoundingClientRect();
        const row = lib.getBoundingClientRect();
        const actions = vis('.qs-slot-actions');
        return {bar_bottom: bar.bottom, lib_top: row.top, lib_bottom: row.bottom,
                actions_top: actions ? actions.getBoundingClientRect().top : null,
                width: row.width, bar_width: bar.width};
    }""")


def scenario_prompts(browser, url, report: Report, samples: Path) -> None:
    """Сохранённые промты под полем промта: Upscale по умолчанию, общий список вкладок."""
    from fooocus_qwen.prompting import library

    print("saved prompts:")
    for width, height in ((1920, 1080), (1100, 800)):
        page, errors = fresh_page(browser, url, width, height)
        for tab, name in ((0, "Generate"), (1, "Edit")):
            open_tab(page, tab)
            box = rows_follow(page)
            below = box["lib_top"] >= box["bar_bottom"] - 1 and box["lib_top"] - box["bar_bottom"] < 40
            if box["actions_top"] is not None:
                below = below and box["lib_bottom"] <= box["actions_top"] + 1
            report.check(below, f"{width}×{height} {name}: the saved-prompts row sits right under the prompt ({box})")
        shot = samples.parent / f"prompts_{width}.png"
        page.screenshot(path=str(shot), full_page=True)
        print(f"  screenshot: {shot}")
        report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
        page.close()

    page, errors = fresh_page(browser, url)
    pick_saved_prompt(page, "Upscale")
    prompt = visible_input(page, "Prompt").input_value()
    report.check(prompt.startswith("LOW-LEVEL PHOTOGRAPHIC RESTORATION ONLY."), "Generate: Upscale is there by default and loads")
    report.check(saved_field(page).input_value() == "Upscale", "the loaded name stays in the field")
    height = rows_follow(page)["lib_bottom"] - rows_follow(page)["lib_top"]
    report.check(height <= 48, f"the row is one line high: {height:.0f} px")

    visible_input(page, "Prompt").fill("a red fox")
    # Посимвольно, как человек: fill() подставляет значение мимо обработчиков
    # ввода Gradio, и вписанное имя до сервера не доходит.
    box = saved_field(page)
    box.click()
    box.fill("")
    box.type("Fox", delay=30)
    click_text(page, "Save prompt")
    page.wait_for_function("() => document.body.innerText.includes('Prompt “Fox” saved') || "
                           "[...document.querySelectorAll('textarea')].some(t => t.value.includes('Prompt “Fox” saved'))",
                           timeout=10000)
    saved = library.load_prompt("Fox", config.PROMPT_DIR)
    report.check(saved.get("prompt") == "a red fox" and "seed" in saved, "Generate saves the whole setup")

    open_tab(page, TAB_EDIT)
    pick_saved_prompt(page, "Fox")
    report.check(visible_input(page, "Prompt").input_value() == "a red fox",
                 "Edit: a prompt saved on Generate shows up without reloading the page")
    visible_input(page, "Prompt").fill("a red fox, sharper")
    click_text(page, "Save prompt")
    page.wait_for_timeout(1000)
    saved = library.load_prompt("Fox", config.PROMPT_DIR)
    report.check(saved.get("prompt") == "a red fox, sharper" and "seed" in saved,
                 "Edit saves the text and keeps the generation setup under the same name")
    click_text(page, "Delete prompt")
    page.wait_for_timeout(1000)
    report.check("Fox" not in library.list_prompts(config.PROMPT_DIR), "Edit: delete removes the prompt")
    saved_field(page).click()
    page.wait_for_timeout(500)
    options = [item.inner_text() for item in page.get_by_role("option").all() if item.is_visible()]
    page.keyboard.press("Escape")
    report.check("Fox" not in options and "Upscale" in options, f"the list follows: {options}")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


NO_OVERFLOW_JS = """() => {
    const width = document.documentElement.clientWidth;
    return [...document.querySelectorAll('*')]
        .filter(e => e.offsetParent !== null && e.getBoundingClientRect().right > width + 1)
        .slice(0, 3).map(e => e.tagName + '.' + String(e.className).slice(0, 40));
}"""


def in_view(page, selector: str) -> bool:
    """Поле на виду: верх выше нижнего края окна больше чем на 120 px."""
    return page.evaluate(f"""() => {{
        const n = [...document.querySelectorAll('{selector}')].find(e => e.offsetParent);
        if (!n) return false;
        const r = n.getBoundingClientRect();
        return r.top < window.innerHeight - 120 && r.bottom > 0;
    }}""")


def scenario_compact(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """Пересмотр интерфейса: лишнего не видно, телефон без вылезания, результат правки на виду."""
    print("compact interface:")
    page, errors = fresh_page(browser, url)
    rewritten = page.get_by_label("Rewritten prompt", exact=True)
    report.check(not any(item.is_visible() for item in rewritten.all()),
                 "Generate: the empty rewritten-prompt field is hidden while AI boost is off")
    visible_input(page, "AI boost").check()
    page.wait_for_timeout(600)
    report.check(any(item.is_visible() for item in rewritten.all()), "turning AI boost on shows it")
    visible_input(page, "AI boost").uncheck()
    page.wait_for_timeout(600)
    report.check(not any(item.is_visible() for item in rewritten.all()), "and turning it off hides it again")
    heights = page.evaluate("""() => [...document.querySelectorAll('button')]
        .filter(b => b.offsetParent && /Send to (editor|references)/.test(b.innerText))
        .map(b => Math.round(b.getBoundingClientRect().height))""")
    report.check(heights and max(heights) <= 36, f"“Send to …” buttons are small: {heights}")

    # Правка на FHD: результат под кистью, после правки страница докручивает до него.
    open_tab(page, TAB_EDIT)
    # Буст правки — тем же набором, что на генерации.
    report.check(not any(item.is_visible() for item in rewritten.all()),
                 "Edit: the empty rewritten-prompt field is hidden while AI boost is off")
    report.check(any(item.is_visible() for item in page.get_by_role("button", name="Rewrite now").all()),
                 "Edit: “Rewrite now” is there, as on Generate")
    visible_input(page, "AI boost").check()
    page.wait_for_timeout(600)
    report.check(any(item.is_visible() for item in rewritten.all()), "Edit: turning AI boost on shows the field")
    visible_input(page, "AI boost").uncheck()
    page.wait_for_timeout(600)
    load_into_painter(page, sample_image(samples, 1200, 800))
    page.evaluate("() => window.scrollTo(0, 0)")
    report.check(not in_view(page, ".qs-preview"), "Edit at 1920×1080: the result starts below the fold")
    before = len(fake.requests)
    apply_edit(page)
    fake.wait(before + 1)
    page.wait_for_function("""() => [...document.querySelectorAll('.qs-preview')].some(e => e.offsetParent && e.querySelector('img'))""",
                           timeout=20000)
    page.wait_for_timeout(1500)
    report.check(in_view(page, ".qs-preview"), "after the edit the page scrolls to the result")

    # Сценарий секрета сохраняет поддельный токен: сверка — с тем, что лежит в файле.
    from fooocus_qwen.ui import tab_settings

    open_tab(page, TAB_SETTINGS)
    shown = page.get_by_role("button", name="Remove the token", exact=True).first.is_visible()
    report.check(shown == tab_settings.has_token(),
                 f"Settings: “Remove the token” is shown only with a saved token (shown {shown})")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()

    page, errors = fresh_page(browser, url, 430, 900)
    for tab, name in enumerate(("Generate", "Edit", "Video", "Gallery", "Settings")):
        open_tab(page, tab)
        if name == "Edit":
            load_into_painter(page, sample_image(samples, 1200, 800))
        overflow = page.evaluate(NO_OVERFLOW_JS)
        report.check(not overflow, f"430 px {name}: nothing sticks out past the screen edge {overflow}")
    open_tab(page, TAB_EDIT)
    zoom = painter_state(page).get("zoom") or 0
    box = page.evaluate("""() => { const n = [...document.querySelectorAll('.qs-refs')].find(e => e.offsetParent);
        const b = [...document.querySelectorAll('.qs-painter')].find(e => e.offsetParent);
        return {refs: n.getBoundingClientRect().bottom, brush: b.getBoundingClientRect().top,
                width: b.getBoundingClientRect().width}; }""")
    report.check(box["refs"] <= box["brush"] + 1 and box["width"] > 300,
                 f"430 px Edit: references above the brush, the brush full width ({box}, zoom {zoom})")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def video_mode(page, label: str) -> None:
    page.get_by_label(label, exact=True).first.check()
    page.wait_for_timeout(600)


def scenario_video(browser, url, report: Report, fake: FakeGenerator, video: FakeVideoEngine, samples: Path) -> None:
    """Вкладка «Видео»: режимы, длительность, отправка картинки, ролик, цепочка, телефон."""
    print("video tab:")
    page, errors = fresh_page(browser, url)
    open_tab(page, TAB_VIDEO)
    note = page.locator(".qs-videonote:visible").all_inner_texts()
    report.check(any("864×480" in text and "124 frames" in text for text in note),
                 f"the size note shows the canvas and the frame grid: {note}")
    first = page.get_by_label("First frame", exact=True)
    report.check(not any(item.is_visible() for item in first.all()), "Text → video: no frame inputs")
    video_mode(page, "First frame → video")
    report.check(any(item.is_visible() for item in page.locator(".qs-videoinputs .image-container").all())
                 or any(item.is_visible() for item in first.all()), "First frame → video: the frame input appears")
    video_mode(page, "From references (characters, style, voice)")
    report.check(page.get_by_text("Up to 9 images").first.is_visible(), "References: the label hint is shown")

    def shown_tiles():
        return page.evaluate("""() => [...document.querySelectorAll('.qs-videorefs .qs-videotile')]
            .filter(e => e.offsetParent).map(e => (e.querySelector('[data-testid=block-label]') || {}).innerText)""")

    report.check(shown_tiles() == ["Image", "Video", "Audio"], f"References: one empty tile per kind: {shown_tiles()}")
    video_mode(page, "Video → video (edit a video)")
    page.wait_for_timeout(800)
    source_shown = any(item.is_visible() for item in page.locator(".qs-videosource").all())
    report.check(source_shown and shown_tiles() == ["Image", "Audio"],
                 f"Video → video: the source video and no video tiles: {shown_tiles()}")
    video_mode(page, "Text → video")
    page.wait_for_timeout(500)
    report.check(not shown_tiles(), "Text → video: no reference tiles")

    # Картинка с генерации — «В видео»: первый кадр, режим «первый кадр».
    open_tab(page, TAB_GENERATE)
    for box in page.locator("textarea").all():
        if box.is_visible():
            box.fill("a gray square")
            break
    before = len(fake.requests)
    click_text(page, "Generate")
    fake.wait(before + 1)
    page.wait_for_function("() => document.querySelector('.preview img')", timeout=20000)
    click_text(page, "To video")
    page.wait_for_timeout(2500)
    active = page.locator('button[role="tab"][aria-selected="true"]').inner_text()
    report.check(active == "Video", f"“To video” opens the Video tab: {active}")
    loaded = page.evaluate("""() => [...document.querySelectorAll('.qs-videoinputs img')]
        .some(img => img.offsetParent && img.complete && img.naturalWidth > 0)""")
    report.check(loaded, "the generated image is the first frame")
    checked = page.get_by_label("First frame → video", exact=True).first.is_checked()
    report.check(checked, "the mode switched to First frame → video")

    # В режиме референсов «В видео» кладёт картинку в первую свободную плитку.
    video_mode(page, "From references (characters, style, voice)")
    open_tab(page, TAB_GENERATE)
    click_text(page, "To video")
    page.wait_for_timeout(2500)
    report.check(shown_tiles()[:2] == ["<Picture 1>", "Image"],
                 f"“To video” in references mode fills the first image tile: {shown_tiles()}")
    video_mode(page, "First frame → video")

    # Upscale: галочка открывает параметры, подпись называет выходной кадр,
    # режим меняет поля, задание несёт размер для движка.
    page.get_by_label("Upscale", exact=True).first.check()
    page.wait_for_timeout(1200)
    multiplier = page.get_by_label("Multiplier of the frame (1-4)")
    report.check(any(item.is_visible() for item in multiplier.all()), "Upscale: the multiplier appears")
    note = " ".join(page.locator(".qs-videonote:visible").all_inner_texts())
    report.check("1728×960" in note, f"Upscale: the size note names the output frame: {note[-80:]}")
    page.get_by_label("Target dimensions", exact=True).first.check()
    page.wait_for_timeout(1200)
    widths = page.get_by_label("Width, px")
    report.check(any(item.is_visible() for item in widths.all()),
                 f"Upscale: target dimensions show width and height ({widths.count()} found)")
    # Slider прятался не до конца (поле числа и подпись оставались) — отсюда
    # перечень видимых частей в сообщении.
    shown = [item.evaluate("e => e.tagName + ' ' + e.type") for item in multiplier.all() if item.is_visible()]
    report.check(not shown, f"Upscale: target dimensions hide the multiplier {shown}")
    page.get_by_label("Megapixels", exact=True).first.check()
    page.wait_for_timeout(1200)
    area = page.get_by_label("Frame area, MP")
    report.check(any(item.is_visible() for item in area.all())
                 and not any(item.is_visible() for item in widths.all()),
                 "Upscale: megapixels show the area slider only")
    page.get_by_label("Scale by multiplier", exact=True).first.check()
    page.wait_for_timeout(1200)

    # Ролик: задание ушло, плеер показал MP4, строка состояния — сид и размер.
    jobs = len(video.jobs)
    for box in page.locator("textarea").all():
        if box.is_visible():
            box.fill("The camera slowly pushes in.")
            break
    click_text(page, "Make video")
    deadline = time.time() + 30
    while len(video.jobs) <= jobs and time.time() < deadline:
        time.sleep(0.2)
    job = video.jobs[-1] if len(video.jobs) > jobs else {}
    report.check(bool(job) and (job.get("media") or {}).get("first"), f"the job carries the first frame: {list(job)}")
    report.check("For the target video, at 0.00 seconds" in (job.get("prompts") or [""])[0],
                 "the H3 prompt starts with the I2VA alignment line")
    page.wait_for_function("""() => [...document.querySelectorAll('.qs-videoboard video')]
        .some(v => v.offsetParent && v.currentSrc)""", timeout=30000)
    status = page.locator(".qs-status textarea:visible").first.input_value()
    report.check(job.get("final_upscale") == [1728, 960], f"the job asks for the upscale: {job.get('final_upscale')}")
    report.check("Done" in status and "1728×960" in status, f"status reports the result: {status[:120]}")

    # Цепочка: последний кадр — первым кадром следующего шота.
    click_text(page, "Last frame → next shot")
    page.wait_for_timeout(2500)
    status = page.locator(".qs-status textarea:visible").first.input_value()
    report.check("next shot" in status, f"“Last frame → next shot” fills the first frame: {status[:100]}")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()

    # Галерея показывает ролик, карточка — параметры ролика.
    page, errors = fresh_page(browser, url)
    open_tab(page, TAB_GALLERY)
    page.wait_for_timeout(1500)
    report.check(page.locator(".qs-browse video, .qs-browse [data-testid*='video']").count() > 0
                 or page.evaluate("() => document.querySelector('.qs-browse').innerHTML.includes('.mp4')"),
                 "the gallery lists the video")
    report.check(not errors, "gallery: no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()

    page, errors = fresh_page(browser, url, 430, 900)
    open_tab(page, TAB_VIDEO)
    overflow = page.evaluate(NO_OVERFLOW_JS)
    report.check(not overflow, f"430 px Video: nothing sticks out past the screen edge {overflow}")
    report.check(not errors, "phone: no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Browser check of the interface")
    parser.add_argument("--port", type=int, default=7899)
    parser.add_argument("--only", nargs="*", default=None,
                        help="layout language painter annotation outpaint latency paste gallery send references tools viewer sketch poses performance secret formats prompts compact video")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    from fooocus_qwen.ui import app

    # Рабочий каталог — внутри проекта (tmp/ под .gitignore), не в системном
    # %TEMP%: всё, что порождает прогон, остаётся рядом с проектом.
    scratch = Path(__file__).resolve().parents[1] / "tmp" / "ui_check"
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=scratch))
    # Результаты подставного генератора не должны попасть в галерею человека.
    config.OUTPUT_DIR = work / "outputs"
    config.PROMPT_DIR = work / "prompts"
    # Адрес языковой модели тоже подменяется: сценарий секрета пишет туда
    # поддельный токен, и настоящий файл человека трогать нельзя.
    config.ENDPOINT_FILE = work / "llm_endpoint.txt"
    # Настройки производительности и дополнительные веса — тоже свои: сценарий
    # производительности переключает внимание, и выбор человека трогать нельзя.
    config.SETTINGS_FILE = work / "settings.json"
    config.INT8_DIR = work / "int8"
    config.TURBO_DIR = work / "turbo"
    # Лора расширения кадра — заглушка: подделке генератора веса не нужны,
    # а качать 159 МБ ради проверки интерфейса незачем.
    config.OUTPAINT_DIR = work / "outpaint"
    config.OUTPAINT_DIR.mkdir()
    from fooocus_qwen.engine import fetch as fetch_module

    (config.OUTPAINT_DIR / fetch_module.OUTPAINT_FILE).write_bytes(b"stub")
    # Свои позы живут в каталоге генераций (config.user_pose_dir) и уезжают во
    # временный вместе с ним. Каталог поз и веса DWPose — настоящие, для чтения.
    config.ensure_directories()
    # Промты по умолчанию — как при запуске (__main__): в свой каталог прогона.
    from fooocus_qwen.prompting import library as prompt_library

    prompt_library.install_defaults(config.DEFAULT_PROMPT_DIR, config.PROMPT_DIR)

    fake = FakeGenerator()
    # Ролики подставного движка и рабочие каталоги заданий — тоже в прогоне.
    config.VIDEO_RUNS_DIR = work / "video_runs"
    video = FakeVideoEngine(work / "fake_video")
    # Сценарии ищут на странице подписи английского интерфейса — он по
    # умолчанию; русский проверяет сценарий языка переключением.
    lora_dir = make_loras(work / "loras")
    cfg = config.AppConfig(host="127.0.0.1", port=args.port, preload=False, lang="en", lora_dir=lora_dir)
    app.start(cfg, prepare=lambda studio: prepare_studio(studio, fake, video))
    url = f"http://127.0.0.1:{args.port}/"
    samples = work / "samples"
    samples.mkdir()

    scenarios = {
        "layout": lambda b, r: scenario_layout(b, url, r),
        "language": lambda b, r: scenario_language(b, url, r),
        "painter": lambda b, r: scenario_painter(b, url, r, fake, samples),
        "annotation": lambda b, r: scenario_annotation(b, url, r, fake, samples),
        "outpaint": lambda b, r: scenario_outpaint(b, url, r, fake, samples),
        "crop": lambda b, r: scenario_crop(b, url, r, fake, samples),
        "latency": lambda b, r: scenario_latency(b, url, r, samples),
        "paste": lambda b, r: scenario_paste(b, url, r),
        "gallery": lambda b, r: scenario_gallery(b, url, r, fake, samples),
        "send": lambda b, r: scenario_send(b, url, r, fake),
        "references": lambda b, r: scenario_references(b, url, r, fake, samples),
        "tools": lambda b, r: scenario_tools(b, url, r, fake, samples),
        "viewer": lambda b, r: scenario_viewer(b, url, r, fake, samples),
        "sketch": lambda b, r: scenario_sketch_colour(b, url, r, fake, samples),
        "poses": lambda b, r: scenario_pose_edit(b, url, r, fake, samples),
        "quality": lambda b, r: scenario_quality(b, url, r, samples),
        "tall": lambda b, r: scenario_tall(b, url, r, samples),
        "loras": lambda b, r: scenario_loras(b, url, r, fake, samples),
        "performance": lambda b, r: scenario_performance(b, url, r, fake, samples),
        "secret": lambda b, r: scenario_secret(b, url, r),
        "formats": lambda b, r: scenario_formats(b, url, r, samples),
        "prompts": lambda b, r: scenario_prompts(b, url, r, samples),
        "compact": lambda b, r: scenario_compact(b, url, r, fake, samples),
        "video": lambda b, r: scenario_video(b, url, r, fake, video, samples),
    }
    chosen = args.only or list(scenarios)
    report = Report()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--use-angle=d3d11", "--ignore-gpu-blocklist"])
        for name in chosen:
            try:
                scenarios[name](browser, report)
            except Exception as error:  # noqa: BLE001 — сценарий обязан не ронять остальные
                report.check(False, f"{name}: scenario aborted: {type(error).__name__}: {error}")
        browser.close()

    print(f"\ntotal {len(report.passed) + len(report.failed)}, failed {len(report.failed)}")
    for item in report.failed:
        print(f"  ✗ {item}")
    return len(report.failed)


if __name__ == "__main__":
    raise SystemExit(main())

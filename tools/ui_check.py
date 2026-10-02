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
TABS = 4


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
    page.locator('button[role="tab"]').nth(index).click()
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
    open_tab(page, 1)
    tabs = [t.inner_text() for t in page.locator('button[role="tab"]').all()]
    report.check(tabs == ["Generate", "Edit", "Gallery", "Settings"], f"English by default: {tabs}")
    page.locator("button.qs-lang").click()
    page.wait_for_timeout(2000)
    tabs = [t.inner_text() for t in page.locator('button[role="tab"]').all()]
    report.check(tabs == ["\u0413\u0435\u043d\u0435\u0440\u0430\u0446\u0438\u044f",
                          "\u0420\u0435\u0434\u0430\u043a\u0442\u0438\u0440\u043e\u0432\u0430\u043d\u0438\u0435",
                          "\u0413\u0430\u043b\u0435\u0440\u0435\u044f",
                          "\u041d\u0430\u0441\u0442\u0440\u043e\u0439\u043a\u0438"],
                 "RU button: all four tabs switched to Russian")
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
    open_tab(page, 1)
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
    open_tab(page, 1)
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
    print("outpaint and result round trip:")
    page, errors = fresh_page(browser, url)
    open_tab(page, 1)
    load_into_painter(page, sample_image(samples, 1024, 768))
    page.get_by_text("Outpaint", exact=True).first.click()
    page.wait_for_timeout(500)
    page.get_by_label("→", exact=True).check()
    page.get_by_role("button", name="Outpaint", exact=True).last.click()
    page.wait_for_function(f"() => {API}.state().width > 1024", timeout=20000)
    page.wait_for_timeout(800)
    state = painter_state(page)
    report.check(state["width"] > 1024 and state["height"] == 768, f"canvas widened: {state['width']}x{state['height']}")

    before = len(fake.requests)
    apply_edit(page)
    request = fake.wait(before + 1)
    ok = request.mask is not None and mask_at(request.mask, 0.97, 0.5) == 255 and mask_at(request.mask, 0.1, 0.5) == 0
    report.check(ok, "new area is marked, old area is not")

    page.wait_for_timeout(1500)
    old_source = painter_state(page)["source"]
    click_text(page, "Send to editor")
    page.wait_for_function(f"() => {API}.state().source !== {old_source!r}".replace("'", '"'), timeout=20000)
    report.check(painter_state(page)["strokes"] == 0, "result came back to the painter without old marks")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_latency(browser, url, report: Report, samples: Path) -> None:
    """Длинный мазок на кадре 4K при плотности экрана 2x: стоимость движения не растёт.

    Тот самый замер, которым было поймано запаздывание gr.ImageEditor: у него
    на этом кадре движение дорожало по ходу мазка с 16.7 до 44 мс.
    """
    print("painter latency:")
    page, errors = fresh_page(browser, url, scale=2)
    open_tab(page, 1)
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
    open_tab(page, 1)
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
    open_tab(page, 2)
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
    open_tab(page, 2)
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
    open_tab(page, 1)
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
    open_tab(page, 1)
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
    open_tab(page, 1)
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
    open_tab(page, 2)
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
    open_tab(page, 1)
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
    open_tab(page, 1)
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
    open_tab(page, 3)
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
    open_tab(page, 3)
    page.wait_for_timeout(800)
    html = page.content()
    fields = " ".join(t.input_value() for t in page.locator("textarea, input").all() if t.is_visible())
    report.check("ui-check-secret" not in html and "ui-check-secret" not in fields, "the token is neither in the markup nor in the fields")
    report.check("192.0.2.10" in fields, "while the address is visible")
    report.check(not errors, "no page errors" + (f": {errors[:2]}" if errors else ""))
    page.close()


# --- запуск --------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Browser check of the interface")
    parser.add_argument("--port", type=int, default=7899)
    parser.add_argument("--only", nargs="*", default=None,
                        help="layout language painter annotation outpaint latency paste gallery send references tools viewer sketch poses performance secret")
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
    # Свои позы живут в каталоге генераций (config.user_pose_dir) и уезжают во
    # временный вместе с ним. Каталог поз и веса DWPose — настоящие, для чтения.
    config.ensure_directories()

    fake = FakeGenerator()
    # Сценарии ищут на странице подписи английского интерфейса — он по
    # умолчанию; русский проверяет сценарий языка переключением.
    lora_dir = make_loras(work / "loras")
    cfg = config.AppConfig(host="127.0.0.1", port=args.port, preload=False, lang="en", lora_dir=lora_dir)
    app.start(cfg, prepare=lambda studio: setattr(studio, "_generator", fake))
    url = f"http://127.0.0.1:{args.port}/"
    samples = work / "samples"
    samples.mkdir()

    scenarios = {
        "layout": lambda b, r: scenario_layout(b, url, r),
        "language": lambda b, r: scenario_language(b, url, r),
        "painter": lambda b, r: scenario_painter(b, url, r, fake, samples),
        "annotation": lambda b, r: scenario_annotation(b, url, r, fake, samples),
        "outpaint": lambda b, r: scenario_outpaint(b, url, r, fake, samples),
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
        "loras": lambda b, r: scenario_loras(b, url, r, fake, samples),
        "performance": lambda b, r: scenario_performance(b, url, r, fake, samples),
        "secret": lambda b, r: scenario_secret(b, url, r),
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

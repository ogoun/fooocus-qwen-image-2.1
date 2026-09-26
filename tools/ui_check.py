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
        raise TimeoutError(f"генератор не получил запрос №{count} за {timeout} с")


# --- учёт результатов ----------------------------------------------------------

@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def check(self, ok: bool, text: str) -> bool:
        (self.passed if ok else self.failed).append(text)
        print(f"  {'ок  ' if ok else 'ПРОВАЛ'} {text}")
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
    click_text(page, "Применить правку")


# --- сценарии ------------------------------------------------------------------

def scenario_layout(browser, url, report: Report) -> None:
    """Все вкладки на шести размерах окна: прокрутка вбок, выход за край, холст выше окна."""
    print("раскладка:")
    for width, height in SIZES:
        page, errors = fresh_page(browser, url, width, height)
        for tab in range(TABS):
            open_tab(page, tab)
            problems = page.evaluate("""() => {
                const doc = document.documentElement, out = [];
                if (doc.scrollWidth > doc.clientWidth + 1) out.push(`прокрутка вбок ${doc.scrollWidth}>${doc.clientWidth}`);
                document.querySelectorAll('.qs-board, .qs-browse, .qs-side, .qs-canvas, .qs-refs').forEach(el => {
                    const r = el.getBoundingClientRect();
                    if (!r.width) return;
                    const name = (el.className.toString().match(/qs-[a-z-]+/) || ['?'])[0];
                    if (r.right > doc.clientWidth + 1) out.push(`${name} за краем на ${Math.round(r.right - doc.clientWidth)}px`);
                    if ((name === 'qs-board' || name === 'qs-browse') && r.height > innerHeight) out.push(`${name} выше окна`);
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
                    problems.append(f"кисть уже свободного места: {painter_width:.0f} из {free_width:.0f}px")
                # В раскладке «рядом» (шире 2000 px) кисть и результат —
                # одного размера: сетка слева не отнимает ширину у кисти.
                if width > 2000:
                    result_width = page.evaluate(
                        "() => document.querySelector('.qs-slot-result .qs-board').getBoundingClientRect().width"
                    )
                    if abs(painter_width - result_width) > 2:
                        problems.append(f"кисть и результат разной ширины: {painter_width:.0f} и {result_width:.0f}px")
            report.check(not problems, f"{width}x{height} вкладка {tab + 1}" + (f": {problems}" if problems else ""))
        report.check(not errors, f"{width}x{height} без ошибок страницы" + (f": {errors[:2]}" if errors else ""))
        page.close()


def scenario_language(browser, url, report: Report) -> None:
    print("язык:")
    page, errors = fresh_page(browser, url)
    open_tab(page, 1)
    page.locator("button.qs-lang").click()
    page.wait_for_timeout(2000)
    tabs = [t.inner_text() for t in page.locator('button[role="tab"]').all()]
    report.check(tabs == ["Generate", "Edit", "Gallery", "Settings"], f"вкладки по-английски: {tabs}")
    hint = page.evaluate(f"""() => {{
        const host = [...document.querySelectorAll('#{PAINTER} *')].find(n => n.shadowRoot);
        return host.shadowRoot.querySelector('.qp-empty-title').textContent;
    }}""")
    report.check(hint == "Drop an image here", f"кисть по-английски: {hint!r}")
    page.locator("button.qs-lang").click()
    page.wait_for_timeout(2000)
    tabs = [t.inner_text() for t in page.locator('button[role="tab"]').all()]
    report.check(tabs[0] == "Генерация", f"и обратно: {tabs}")
    report.check(not errors, "без ошибок страницы" + (f": {errors[:2]}" if errors else ""))
    page.close()


def mask_at(mask: Image.Image, fx: float, fy: float) -> int:
    array = np.asarray(mask.convert("L"))
    h, w = array.shape
    return int(array[min(int(fy * h), h - 1), min(int(fx * w), w - 1)])


def scenario_painter(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """Кисть от загрузки до маски в запросе: маска там, где её нарисовали."""
    print("кисть:")
    page, errors = fresh_page(browser, url)
    open_tab(page, 1)
    state = load_into_painter(page, sample_image(samples, 1888, 1280))
    report.check(state["width"] == 1888 and state["height"] == 1280, f"картинка загружена: {state['width']}x{state['height']}")
    report.check(bool(state["source"]), "исходник передан на сервер")

    stroke(page, [(0.3, 0.5), (0.7, 0.5)])
    report.check(painter_state(page)["strokes"] == 1, "мазок записан в историю")

    before = len(fake.requests)
    click_text(page, "Применить правку")
    page.wait_for_timeout(1500)
    status = " ".join(box.input_value() for box in page.locator("textarea").all())
    report.check(
        len(fake.requests) == before and "Опишите правку" in status,
        "без промта правка не запускается и просит описать её",
    )

    apply_edit(page)
    request = fake.wait(before + 1)
    report.check(request.source is not None and request.source.size == (1888, 1280), "сервер получил исходник целиком")
    # Результат — крупным просмотром, а не сеткой: в сетке единственная
    # картинка становилась квадратом выше окна, и видна была полоса кадра.
    page.wait_for_function("() => document.querySelector('.qs-slot-result .preview img')", timeout=20000)
    report.check(True, "результат открылся крупным просмотром")
    report.check(request.mask is not None, "сервер получил маску")
    if request.mask is not None:
        centre, corner = mask_at(request.mask, 0.5, 0.5), mask_at(request.mask, 0.05, 0.05)
        report.check(centre == 255 and corner == 0, f"маска там, где рисовали: центр {centre}, угол {corner}")
    report.check(request.mask_mode == "mask", f"режим маски: {request.mask_mode}")

    # Отмена: без пометок правится весь кадр.
    page.locator(f"#{PAINTER}").hover()
    page.keyboard.press("Control+z")
    page.wait_for_timeout(300)
    report.check(painter_state(page)["strokes"] == 0, "Ctrl+Z отменяет мазок")
    before = len(fake.requests)
    apply_edit(page)
    request = fake.wait(before + 1)
    report.check(request.mask is None and request.mask_mode == "none", f"без пометок — весь кадр: {request.mask_mode}")

    # Ластик правой кнопкой: мазок кистью и поверх — стирание.
    stroke(page, [(0.2, 0.3), (0.8, 0.3)])
    stroke(page, [(0.2, 0.3), (0.8, 0.3)], button="right")
    before = len(fake.requests)
    apply_edit(page)
    request = fake.wait(before + 1)
    report.check(request.mask is None, "правая кнопка стирает")

    report.check(not errors, "без ошибок страницы" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_annotation(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    print("аннотация:")
    page, errors = fresh_page(browser, url)
    open_tab(page, 1)
    load_into_painter(page, sample_image(samples, 1024, 768))
    page.get_by_label("Аннотация", exact=True).check()
    page.wait_for_timeout(800)
    report.check(painter_state(page)["region"] == "annotation", "кисть узнала режим аннотации")
    page.evaluate(f"""() => {{
        const host = [...document.querySelectorAll('#{PAINTER} *')].find(n => n.shadowRoot);
        host.shadowRoot.querySelector('.qp-swatch[data-color="#0000ff"]').click();
    }}""")
    stroke(page, [(0.4, 0.4), (0.6, 0.6)])
    before = len(fake.requests)
    apply_edit(page)
    request = fake.wait(before + 1)
    pixel = np.asarray(request.source.convert("RGB"))[int(0.5 * 768), int(0.5 * 1024)]
    report.check(request.mask is None and request.mask_mode == "annotation", f"режим аннотации: {request.mask_mode}")
    report.check(pixel[2] > 200 and pixel[0] < 60, f"синяя пометка сведена в исходник: {tuple(int(v) for v in pixel)}")
    report.check(not errors, "без ошибок страницы" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_outpaint(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    print("расширение холста и возврат результата:")
    page, errors = fresh_page(browser, url)
    open_tab(page, 1)
    load_into_painter(page, sample_image(samples, 1024, 768))
    page.get_by_text("Расширить холст", exact=True).first.click()
    page.wait_for_timeout(500)
    page.get_by_label("→", exact=True).check()
    page.get_by_role("button", name="Расширить холст", exact=True).last.click()
    page.wait_for_function(f"() => {API}.state().width > 1024", timeout=20000)
    page.wait_for_timeout(800)
    state = painter_state(page)
    report.check(state["width"] > 1024 and state["height"] == 768, f"холст расширен: {state['width']}x{state['height']}")

    before = len(fake.requests)
    apply_edit(page)
    request = fake.wait(before + 1)
    ok = request.mask is not None and mask_at(request.mask, 0.97, 0.5) == 255 and mask_at(request.mask, 0.1, 0.5) == 0
    report.check(ok, "новая площадь помечена, старая — нет")

    page.wait_for_timeout(1500)
    old_source = painter_state(page)["source"]
    click_text(page, "Отправить в редактор")
    page.wait_for_function(f"() => {API}.state().source !== {old_source!r}".replace("'", '"'), timeout=20000)
    report.check(painter_state(page)["strokes"] == 0, "результат вернулся в кисть без старых пометок")
    report.check(not errors, "без ошибок страницы" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_latency(browser, url, report: Report, samples: Path) -> None:
    """Длинный мазок на кадре 4K при плотности экрана 2x: стоимость движения не растёт.

    Тот самый замер, которым было поймано запаздывание gr.ImageEditor: у него
    на этом кадре движение дорожало по ходу мазка с 16.7 до 44 мс.
    """
    print("запаздывание кисти:")
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
    report.check(growth < 1.3, f"движение по четвертям мазка, мс: {text} (рост {growth:.2f}x)")
    report.check(statistics.mean(durations) < 20, f"среднее движение {statistics.mean(durations):.1f} мс (бюджет кадра 16.7)")
    report.check(not errors, "без ошибок страницы" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_paste(browser, url, report: Report) -> None:
    """Ctrl+V над кистью открывает картинку из буфера обмена.

    Пустой холст обещает это подсказкой, так что обещание проверяется.
    Событие создаётся из скрипта: доверенность ему не нужна, в отличие от
    движений мыши, — обработчик читает только ``clipboardData``.
    """
    print("вставка из буфера:")
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
    report.check(state["height"] == 200 and bool(state["source"]), f"картинка вставлена и передана: {state['width']}x{state['height']}")
    report.check(not errors, "без ошибок страницы" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_gallery(browser, url, report: Report, fake: FakeGenerator, samples: Path) -> None:
    """Карточка вместо JSON и действия с переходом на нужную вкладку."""
    print("галерея:")
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
    report.check(thumbs.count() >= 1, f"галерея обновилась при открытии: {thumbs.count()} картинок")
    # Тег на месте — ещё не картинка: сервер может отказать в файле. Так и
    # было — обновление галереи без очереди получало ссылки вида
    # /gradio_api/run/predict/gradio_api/file=… с ответом 404, и галерея
    # показывала битые значки при зелёной проверке.
    page.wait_for_timeout(1500)
    loaded = page.evaluate(
        "() => [...document.querySelectorAll('.qs-browse img')].filter(i => i.complete && i.naturalWidth > 0).length"
    )
    report.check(loaded == thumbs.count(), f"картинки галереи загрузились: {loaded} из {thumbs.count()}")
    thumbs.first.click()
    page.wait_for_timeout(1200)
    card = page.locator(".block.qs-card").inner_text()
    report.check("a lighthouse at dawn" in card and "777" in card and "{" not in card, "карточка — читаемый текст")

    click_text(page, "Открыть в редакторе")
    page.wait_for_function(f"() => {API}.state().width === 640", timeout=20000)
    active = page.locator('button[role="tab"][aria-selected="true"]').inner_text()
    report.check(active == "Редактирование", f"переход на правку: {active}")

    # Выбор переживает уход с вкладки: карточка на месте, кликать снова не
    # нужно. И нельзя: галерея открыта в режиме просмотра, где клик по
    # крупному кадру листает на следующий, — сценарий выбирал бы чужой файл.
    open_tab(page, 2)
    card = page.locator(".block.qs-card").inner_text()
    report.check("a lighthouse at dawn" in card, "выбор сохранился после ухода с вкладки")
    click_text(page, "Повторить параметры")
    page.wait_for_timeout(1500)
    active = page.locator('button[role="tab"][aria-selected="true"]').inner_text()
    prompt = page.locator("textarea").first.input_value()
    report.check(active == "Генерация" and prompt == "a lighthouse at dawn", f"повтор: вкладка {active}, промт {prompt!r}")
    report.check(not errors, "без ошибок страницы" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_send(browser, url, report: Report, fake: FakeGenerator) -> None:
    """Генерация → «Отправить в редактор»: картинка в кисти, вкладка правки открыта."""
    print("отправка результата генерации в редактор:")
    page, errors = fresh_page(browser, url)
    for box in page.locator("textarea").all():
        if box.is_visible():
            box.fill("a gray square")
            break
    before = len(fake.requests)
    click_text(page, "Сгенерировать")
    fake.wait(before + 1)
    page.wait_for_function("() => document.querySelector('.preview img')", timeout=20000)
    # Подпись кнопки локализуется под язык браузера: проверяются обе. Что
    # селектор её вообще находит, проверено на голой галерее Gradio с кнопками
    # по умолчанию — там «Поделиться» есть.
    share = 'button[aria-label="Share"], button[aria-label="Поделиться"]'
    report.check(page.locator(share).count() == 0, "у галерей нет кнопки «поделиться»")
    click_text(page, "Отправить в редактор")
    page.wait_for_function(f"() => {API}.state().width === 256", timeout=20000)
    active = page.locator('button[role="tab"][aria-selected="true"]').inner_text()
    report.check(active == "Редактирование", f"открыта вкладка правки: {active}")
    report.check(painter_state(page)["height"] == 256, "картинка генерации — в кисти")
    report.check(not errors, "без ошибок страницы" + (f": {errors[:2]}" if errors else ""))
    page.close()


# По ячейке: загружена ли картинка, что написано в строке тега и влезает ли
# тег в свою строку. Последнее проверяется шириной, а не текстом: подпись
# «<image1>», обрезанная до «<im», в innerText остаётся целой — так первая
# редакция сетки и прошла текстовую проверку с нечитаемыми тегами.
SLOTS_JS = """() => [...document.querySelectorAll('.qs-refcell')].filter(cell => cell.getBoundingClientRect().width > 0).map(cell => {
    const img = cell.querySelector('.qs-refslot img');
    const tag = cell.querySelector('.qs-reftag');
    const inner = tag.querySelector('code') || tag;
    return {
        loaded: !!(img && img.complete && img.naturalWidth > 0),
        text: tag.innerText.trim(),
        clipped: inner.getBoundingClientRect().right > tag.getBoundingClientRect().right + 1
            || tag.scrollWidth > tag.clientWidth + 1,
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
        raise AssertionError(f"ждали слоты {expected}, загружены {actual}") from error


def wait_label(page, index: int, fragment: str, timeout: int = 20000) -> str:
    """Ждёт подпись слота: она приходит ответом обработчика, позже картинки."""
    try:
        page.wait_for_function(
            f"() => ({SLOTS_JS})()[{index}].text.includes({fragment!r})", timeout=timeout
        )
    except Exception as error:
        raise AssertionError(
            f"в слоте {index} нет «{fragment}»: «{reference_slots(page)[index]['text']}»"
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
    print("сетка референсов:")
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
        report.check(geometry["count"] == 10, f"{size}: ячеек десять: {geometry['count']}")
        report.check(geometry["rows"] == 5 and geometry["columns"] == 2,
                     f"{size}: два столбца по пять: {geometry['columns']}×{geometry['rows']}")
        report.check(abs(refs_top - board_top) <= 2 and abs(refs_bottom - board_bottom) <= 2,
                     f"{size}: сетка в рост поля результата: {refs_top}–{refs_bottom} и {board_top}–{board_bottom}")
        report.check(geometry["bottomCell"] <= refs_bottom + 1,
                     f"{size}: ячейки не вылезают за сетку: {geometry['bottomCell']} ≤ {refs_bottom}")
        report.check(refs_right <= board_left + 1, f"{size}: сетка левее результата: {refs_right} ≤ {board_left}")
        report.check(geometry["ratio"] <= 1.5,
                     f"{size}: ячейки близки к квадрату: {geometry['cell']} px, вытянутость {geometry['ratio']:.2f}")
        if geometry["ratio"] > 1.5:
            print("    цепочка:")
            for link in geometry["chain"]:
                print(f"      {link}")
    page.set_viewport_size({"width": 1920, "height": 1080})
    page.wait_for_timeout(300)
    report.check(not any(slot["loaded"] for slot in reference_slots(page)), "по умолчанию все пусты")

    # Картинка кладётся прямо в третью ячейку — как это сделал бы человек,
    # перетащив файл или выбрав его кликом (оба пути ведут в тот же input).
    page.locator(".qs-refslot").nth(2).locator('input[type="file"]').set_input_files(
        str(sample_image(samples, 300, 200))
    )
    wait_loaded(page, [2])
    # Не «тега нет», а «подпись о том, что тег не нужен, есть»: отсутствие
    # тега выполняется и тогда, когда подписи нет вовсе.
    label = wait_label(page, 2, "без тега")
    report.check("<image" not in label, f"один референс — подписан без тега: «{label}»")

    # Результат генерации — в первый свободный слот, то есть в первый.
    for box in page.locator("textarea").all():
        if box.is_visible():
            box.fill("a gray square")
            break
    before = len(fake.requests)
    click_text(page, "Сгенерировать")
    fake.wait(before + 1)
    page.wait_for_function("() => document.querySelector('.preview img')", timeout=20000)
    click_text(page, "Отправить в референсы")
    wait_loaded(page, [0, 2])
    first, third = wait_label(page, 0, "<image1>"), wait_label(page, 2, "<image2>")
    report.check(True, f"теги по порядку заполненных: «{first}», «{third}»")

    # С правки — снова в первый свободный, то есть во второй, и переход сюда.
    open_tab(page, 1)
    load_into_painter(page, sample_image(samples, 1024, 768))
    apply_edit(page)
    fake.wait(before + 2)
    page.wait_for_function("() => document.querySelector('.qs-slot-result .preview img')", timeout=20000)
    click_text(page, "Отправить в референсы")
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
        raise AssertionError(f"{error}; вкладка «{tab}»; строки: {statuses[-3:]}; слот 2: {slot}") from error
    active = page.locator('button[role="tab"][aria-selected="true"]').inner_text()
    report.check(active == "Генерация", f"с правки — переход на генерацию: {active}")
    tags = [wait_label(page, index, f"<image{index + 1}>") for index in (0, 1, 2)]
    report.check(True, f"теги трёх слотов сдвинулись по порядку: {tags}")
    clipped = [slot["text"] for slot in reference_slots(page) if slot["clipped"]]
    report.check(not clipped, "теги видны целиком, не обрезаны" + (f": {clipped}" if clipped else ""))
    # Снимок заполненной сетки — глазам: текстом проверено, что подписи
    # есть, но не то, как подпись ложится поверх картинки в ячейке.
    shot = samples.parent / "references-grid.png"
    page.locator(".qs-refs").first.screenshot(path=str(shot))
    print(f"  снимок сетки: {shot}")

    # Модель получает ровно заполненные слоты и ровно в порядке тегов.
    click_text(page, "Сгенерировать")
    request = fake.wait(before + 3)
    sizes = [image.size for image in request.references]
    report.check(sizes == [(256, 256), (1024, 768), (300, 200)],
                 f"в запрос ушли слоты 1, 2, 3 по порядку: {sizes}")

    click_text(page, "Очистить референсы")
    wait_loaded(page, [])
    report.check(True, "«Очистить референсы» опустошает сетку")
    report.check(not errors, "без ошибок страницы" + (f": {errors[:2]}" if errors else ""))
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

    print("поза и эскиз в ячейке:")
    page, errors = fresh_page(browser, url)

    icons = visible_icons(page)
    flat = [icon for cell in icons for icon in cell]
    report.check(len(icons) == 10 and all(len(cell) == 2 for cell in icons),
                 f"на каждой из десяти ячеек два значка: {[len(cell) for cell in icons]}")
    report.check(all(icon["inside"] for icon in flat), "значки лежат поверх картинки ячейки")
    report.check(all(icon["title"] for icon in flat),
                 f"у значков есть подсказки: «{flat[0]['title']}», «{flat[1]['title']}»")

    # --- поза из каталога — в третью ячейку ---
    page.locator(".qs-refpose:visible").nth(2).click()
    wait_modal(page, True)
    catalog = len(library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir()))
    wait_pose_tiles(page, catalog + 1)
    report.check(pose_tiles_loaded(page) == catalog + 1,
                 f"в окне {catalog} поз и плитка «Добавить позу»: {pose_tiles_loaded(page)}")
    box = page.evaluate(f"() => {{ const r = ({VISIBLE_MODAL_JS})().getBoundingClientRect();"
                        " return [r.left, r.top, r.width, r.height].map(Math.round); }")
    report.check(box[0] == 0 and box[1] == 0 and box[2] >= 1900 and box[3] >= 1070,
                 f"окно накрывает страницу целиком: {box}")
    shot = samples.parent / "pose-window.png"
    page.screenshot(path=str(shot))
    print(f"  снимок окна поз: {shot}")
    page.locator(".qs-posegrid:visible img").first.click()
    wait_loaded(page, [2])
    wait_modal(page, False)
    report.check(True, "плитка каталога: скелет в ячейке 3, окно закрылось")

    # --- «Добавить позу» по фото — в первую ячейку ---
    page.locator(".qs-refpose:visible").nth(0).click()
    wait_modal(page, True)
    wait_pose_tiles(page, catalog + 1)
    page.locator(".qs-posegrid:visible img").last.click()
    page.wait_for_selector(".qs-posephoto:visible input[type=file]", state="attached", timeout=20000)
    report.check("распозна" in modal_message(page), f"подсказка к фото: «{modal_message(page)[:60]}…»")
    photo = samples / "pose_photo.jpg"
    shutil.copy(config.POSE_LIBRARY_DIR / "dance_02.jpg", photo)
    before = len(fake.requests)
    page.locator(".qs-posephoto:visible input[type=file]").set_input_files(str(photo))
    wait_loaded(page, [0, 2], timeout=60000)
    request = fake.wait(before + 1, timeout=60)
    report.check(request.prompt == tile.PROMPT and len(request.references) == 1,
                 f"плитка новой позы заказана модели: скелет-референс {request.references[0].size}")
    page.wait_for_function(
        f"() => (({VISIBLE_MODAL_JS})()?.innerText || '').includes('Плитка готова')", timeout=60000,
    )
    custom = library.list_poses(config.POSE_LIBRARY_DIR, config.user_pose_dir())[catalog:]
    report.check(len(custom) == 1 and custom[0].tile.exists() and custom[0].thumb.exists(),
                 f"своя поза сохранена с плиткой: {[entry.name for entry in custom]}")
    wait_pose_tiles(page, catalog + 2, timeout=20000)
    report.check(True, "новая поза встала в окно перед «Добавить позу»")
    click_text(page, "Закрыть")
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
    print(f"  снимок окна эскиза: {shot}")
    click_text(page, "Принять")
    wait_loaded(page, [0, 1, 2])
    wait_modal(page, False)
    report.check(True, "«Принять»: эскиз в ячейке 2, окно закрылось")

    page.locator(".qs-refsketch:visible").nth(4).click()
    wait_modal(page, True)
    click_text(page, "Отмена")
    wait_modal(page, False)
    report.check(not reference_slots(page)[4]["loaded"], "«Отмена» закрывает окно, ячейка 5 пуста")

    # --- модель получает ровно это и в этом порядке ---
    for box in page.locator("textarea").all():
        if box.is_visible():
            box.fill("a dancer <image1>")
            break
    count = len(fake.requests)
    click_text(page, "Сгенерировать")
    request = fake.wait(count + 1)
    sizes = [image.size for image in request.references]
    report.check(sizes == [(768, 768), (1024, 1024), (768, 768)],
                 f"в запрос ушли поза с фото, эскиз, поза каталога: {sizes}")
    pose_img, sketch_img = np.asarray(request.references[0]), np.asarray(request.references[1].convert("L"))
    report.check(pose_img.mean() < 40, f"поза — скелет на чёрном: средняя яркость {pose_img.mean():.0f}")
    centre, corner = sketch_img[500:524, 500:524].mean(), sketch_img[40:80, 900:980].mean()
    report.check(centre < 128 < corner,
                 f"эскиз: мазок в центре тёмный ({centre:.0f}), холст белый ({corner:.0f})")

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
                 f"правка: сетка в рост кисти: {refs_top}–{refs_bottom} и {painter_top}–{painter_bottom}")
    report.check(refs_right <= painter_left + 1, f"правка: сетка левее кисти: {refs_right} ≤ {painter_left}")
    icons = visible_icons(page)
    report.check(len(icons) == 10 and all(icon["title"] for cell in icons for icon in cell),
                 "правка: десять ячеек, у значков есть подсказки (вкладка отрисована после загрузки)")
    report.check(not any(slot["loaded"] for slot in reference_slots(page)),
                 "правка: своя сетка, референсы генерации сюда не попали")

    page.locator(".qs-refpose:visible").nth(0).click()
    wait_modal(page, True)
    wait_pose_tiles(page, catalog + 2)
    page.locator(".qs-posegrid:visible img").first.click()
    wait_loaded(page, [0])
    wait_modal(page, False)
    tag = wait_label(page, 0, "<image3>")
    report.check(True, f"правка, режим «Маска»: исходник и маска впереди, референс — «{tag}»")
    page.get_by_label("Без области — править весь кадр", exact=True).check()
    tag = wait_label(page, 0, "<image2>")
    report.check(True, f"правка, «Без области»: маски нет, референс — «{tag}»")
    page.get_by_label("Маска", exact=True).check()
    wait_label(page, 0, "<image3>")

    count = len(fake.requests)
    apply_edit(page, "put the dancer from <image3> into the marked area")
    request = fake.wait(count + 1)
    sizes = [image.size for image in request.references]
    report.check(sizes == [(768, 768)] and request.mask is not None,
                 f"правка: в запрос ушли маска и референс-поза: {sizes}")
    report.check(not errors, "без ошибок страницы" + (f": {errors[:2]}" if errors else ""))


def scenario_performance(browser, url, report: Report, fake: FakeGenerator) -> None:
    """Секция «Производительность»: состояние, точность, SageAttention на лету."""
    print("производительность:")
    from fooocus_qwen import settings
    from fooocus_qwen.engine import attention

    page, errors = fresh_page(browser, url)
    open_tab(page, 3)
    page.wait_for_timeout(800)
    status = " ".join(box.input_value() for box in page.locator("textarea").all())
    report.check("Точность: BF16" in status, "состояние показывает точность и Turbo")
    report.check(page.get_by_label("bf16 — исходная точность, 13.3 ГиБ видеопамяти").is_checked(),
                 "выбрана текущая точность")

    sage = page.get_by_label("SageAttention — быстрое внимание")
    if not attention.sage_available():
        report.check(sage.is_disabled(), "без пакета галочка SageAttention недоступна")
    else:
        sage.check()
        # Первое включение подгружает модуль трансформера diffusers — секунды.
        try:
            page.wait_for_function(
                "() => [...document.querySelectorAll('textarea')].some(t => t.value.includes('Внимание: SageAttention'))",
                timeout=30000,
            )
        except Exception:  # noqa: BLE001 — проверка ниже скажет, что не так
            pass
        status = " ".join(box.input_value() for box in page.locator("textarea").all())
        report.check(settings.load().sage_attention and "Внимание: SageAttention" in status,
                     f"SageAttention включился и записан: {getattr(fake, 'attention', None)}; {status[-120:]!r}")
        sage.uncheck()
        try:
            page.wait_for_function(
                "() => [...document.querySelectorAll('textarea')].some(t => t.value.includes('Внимание: штатное'))",
                timeout=30000,
            )
        except Exception:  # noqa: BLE001
            pass
        report.check(not settings.load().sage_attention and fake.attention == "native",
                     "и выключился обратно")
    report.check(not errors, "без ошибок страницы" + (f": {errors[:2]}" if errors else ""))
    page.close()


def scenario_secret(browser, url, report: Report) -> None:
    """Токен языковой модели не попадает на страницу ни в каком виде."""
    print("секрет:")
    config.ENDPOINT_FILE.write_text(
        "\n".join(["llama.cpp", "192.0.2.10:8000", "token=ui-check-secret", ""]), encoding="utf-8"
    )
    page, errors = fresh_page(browser, url)
    open_tab(page, 3)
    page.wait_for_timeout(800)
    html = page.content()
    fields = " ".join(t.input_value() for t in page.locator("textarea, input").all() if t.is_visible())
    report.check("ui-check-secret" not in html and "ui-check-secret" not in fields, "токена нет ни в разметке, ни в полях")
    report.check("192.0.2.10" in fields, "адрес при этом виден")
    report.check(not errors, "без ошибок страницы" + (f": {errors[:2]}" if errors else ""))
    page.close()


# --- запуск --------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка интерфейса в браузере")
    parser.add_argument("--port", type=int, default=7899)
    parser.add_argument("--only", nargs="*", default=None,
                        help="layout language painter annotation outpaint latency paste gallery send references tools performance secret")
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
    cfg = config.AppConfig(host="127.0.0.1", port=args.port, preload=False)
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
        "performance": lambda b, r: scenario_performance(b, url, r, fake),
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
                report.check(False, f"{name}: сценарий оборвался: {type(error).__name__}: {error}")
        browser.close()

    print(f"\nвсего {len(report.passed) + len(report.failed)}, провалов {len(report.failed)}")
    for item in report.failed:
        print(f"  ✗ {item}")
    return len(report.failed)


if __name__ == "__main__":
    raise SystemExit(main())

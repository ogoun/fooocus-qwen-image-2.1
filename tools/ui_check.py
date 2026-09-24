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


def image_rect(page) -> dict:
    """Где на экране лежит картинка внутри холста кисти."""
    return page.evaluate(f"""() => {{
        const host = [...document.querySelectorAll('#{PAINTER} *')].find(n => n.shadowRoot);
        const r = host.shadowRoot.querySelector('.qp-image').getBoundingClientRect();
        return {{x: r.left, y: r.top, width: r.width, height: r.height}};
    }}""")


def stroke(page, points: list[tuple[float, float]], steps: int = 25, button: str = "left") -> None:
    """Мазок по долям картинки: (0,0) — левый верхний угол, (1,1) — правый нижний."""
    rect = image_rect(page)

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
                document.querySelectorAll('.qs-board, .qs-browse, .qs-side, .qs-canvas').forEach(el => {
                    const r = el.getBoundingClientRect();
                    if (!r.width) return;
                    const name = (el.className.toString().match(/qs-[a-z-]+/) || ['?'])[0];
                    if (r.right > doc.clientWidth + 1) out.push(`${name} за краем на ${Math.round(r.right - doc.clientWidth)}px`);
                    if ((name === 'qs-board' || name === 'qs-browse') && r.height > innerHeight) out.push(`${name} выше окна`);
                });
                return out;
            }""")
            if tab == 1:
                painter_width = page.evaluate(f"() => document.getElementById('{PAINTER}').getBoundingClientRect().width")
                column_width = page.evaluate(f"() => document.getElementById('{PAINTER}').parentElement.getBoundingClientRect().width")
                if painter_width < column_width * 0.95:
                    problems.append(f"кисть уже колонки: {painter_width:.0f} из {column_width:.0f}px")
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
                        help="layout language painter annotation outpaint latency paste gallery send performance secret")
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

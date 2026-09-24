r"""Скриншоты интерфейса и примеры работы для README.

Интерфейс поднимается той же дорогой, что боевой запуск (``app.start``), на
английском и с настоящей моделью; сценарии проходятся в браузере так, как их
прошёл бы человек. Картинки пересоздаются одной командой после любой правки
интерфейса — руками снятые скриншоты устаревали бы молча.

Что попадает в репозиторий: только ``docs/images/``. Всё рабочее — результаты
генерации, промты, адрес языковой модели для снимка настроек — лежит в
``tmp/docs_screenshots/`` (под .gitignore). Галерея на снимке показывает
только картинки, созданные этим скриптом: пользовательские результаты из
``user/outputs`` в документацию не попадают. Снимок настроек делается с
примерным адресом и без токена — настоящий ``llm_endpoint.txt`` на странице
не появляется (AI буст при этом работает по настоящему адресу: он нужен
снимку генерации).

Две фазы, чтобы между ними можно было посмотреть на витрину и выбрать
область для маски:

    .venv\Scripts\python tools\docs_screenshots.py --phase showcase
    .venv\Scripts\python tools\docs_screenshots.py --phase ui

Нужны видеокарта, веса модели и Chromium для Playwright (requirements-dev.txt).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from PIL import Image  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import presets  # noqa: E402
from fooocus_qwen.engine.generator import GenerationRequest  # noqa: E402
from fooocus_qwen.imaging import metadata  # noqa: E402
from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402
from fooocus_qwen.storage import gallery  # noqa: E402

IMAGES = ROOT / "docs" / "images"
WORK = ROOT / "tmp" / "docs_screenshots"
PORT = 7894
# Ширина картинок в README: GitHub показывает содержимое примерно в 1000 px,
# 1600 — запас на экраны высокой плотности без лишнего веса файла.
DOC_WIDTH = 1600

SHOWCASE = [
    (
        "showcase-storefront",
        'a small coffee shop storefront at dusk, a glowing neon sign that reads "QWEN CAFE", '
        "rain-soaked street with reflections, warm light from the windows, photorealistic",
        "3:2",
        21,
    ),
    (
        "showcase-fox",
        "a watercolor illustration of a red fox reading a book under a tree hung with paper "
        "lanterns, soft washes of colour, visible paper texture",
        "1:1",
        7,
    ),
    (
        "showcase-mug",
        "product photograph of a handmade ceramic mug on a wooden table, morning window light, "
        "shallow depth of field, plenty of empty table space to the right",
        "3:2",
        3,
    ),
]


def configure() -> None:
    config.OUTPUT_DIR = WORK / "outputs"
    config.PROMPT_DIR = WORK / "prompts"
    config.ensure_directories()
    IMAGES.mkdir(parents=True, exist_ok=True)


def save_doc_image(image: Image.Image, name: str) -> Path:
    """Картинка для README: ширина DOC_WIDTH, формат WebP.

    WebP, а не PNG: снимок интерфейса в PNG весил до мегабайта, и каждая
    пересъёмка оседала бы в истории git. При качестве 90 текст интерфейса
    остаётся чётким, а вес падает в несколько раз; GitHub WebP показывает.
    """
    if image.width > DOC_WIDTH:
        image = image.resize((DOC_WIDTH, round(image.height * DOC_WIDTH / image.width)), Image.LANCZOS)
    path = IMAGES / f"{name}.webp"
    image.convert("RGB").save(path, "WEBP", quality=90, method=6)
    print(f"  -> {path.relative_to(ROOT)} ({path.stat().st_size // 1024} КБ)")
    return path


def side_by_side(images: list[Image.Image], height: int = 640, gap: int = 12) -> Image.Image:
    scaled = [im.convert("RGB").resize((round(im.width * height / im.height), height), Image.LANCZOS) for im in images]
    canvas = Image.new("RGB", (sum(im.width for im in scaled) + gap * (len(scaled) - 1), height), "white")
    x = 0
    for im in scaled:
        canvas.paste(im, (x, 0))
        x += im.width + gap
    return canvas


# --- фаза 1: витрина -----------------------------------------------------------

def phase_showcase() -> None:
    from fooocus_qwen.ui.state import Studio

    shutil.rmtree(config.OUTPUT_DIR, ignore_errors=True)
    config.ensure_directories()
    studio = Studio(config.AppConfig(lang="en"))
    engine = studio.generator
    made, kept = [], []
    for name, prompt, ratio, seed in SHOWCASE:
        started = time.time()
        result = engine.generate(GenerationRequest(
            prompt=prompt, prompt_original=prompt, preset=presets.get("MiddleQuality"), aspect=ratio, seed=seed,
        ))[0]
        path = gallery.next_path(config.OUTPUT_DIR)
        metadata.save_png(result.image, path, result.parameters)
        shutil.copy(path, WORK / f"{name}.png")
        kept.append(path.relative_to(config.OUTPUT_DIR).as_posix())
        made.append(result.image)
        print(f"{name}: {time.time() - started:.0f} с -> {path.name}")
        time.sleep(1.1)  # имена файлов — с точностью до секунды
    (WORK / "showcase.txt").write_text("\n".join(kept), encoding="utf-8")
    save_doc_image(side_by_side(made, height=560), "showcase")


# --- фаза 2: интерфейс -----------------------------------------------------------

def phase_ui(mask_box: tuple[float, float, float, float]) -> None:
    import ui_check as u
    from playwright.sync_api import sync_playwright

    from fooocus_qwen.ui import app

    # Галерея снимка начинается с одной витрины: результаты прошлых прогонов
    # этой фазы убираются, чтобы снимки не зависели от истории запусков.
    kept = set((WORK / "showcase.txt").read_text(encoding="utf-8").split())
    for path in config.OUTPUT_DIR.rglob("*.png"):
        if path.relative_to(config.OUTPUT_DIR).as_posix() not in kept:
            path.unlink()

    demo, studio = app.start(config.AppConfig(host="127.0.0.1", port=PORT, lang="en", preload=False))
    studio.generator  # noqa: B018 — модель грузится до первого снимка
    url = f"http://127.0.0.1:{PORT}/"

    def outputs() -> int:
        return len(list(config.OUTPUT_DIR.rglob("*.png")))

    def wait_for_output(before: int, page, timeout: float = 900) -> Path:
        deadline = time.time() + timeout
        while time.time() < deadline and outputs() <= before:
            page.wait_for_timeout(1000)
        if outputs() <= before:
            raise TimeoutError("результат не появился")
        page.wait_for_timeout(2500)  # результат доезжает до страницы
        return max(config.OUTPUT_DIR.rglob("*.png"), key=lambda p: p.stat().st_mtime)

    def visible(locator):
        for item in locator.all():
            if item.is_visible():
                return item
        raise LookupError("нет видимого элемента")

    def shot(page, name: str) -> None:
        # Снимок — только когда все видимые картинки догрузились: иначе в кадр
        # попадает полоса недорисованного изображения.
        page.wait_for_function(
            """() => [...document.querySelectorAll('img')]
                .filter(i => i.offsetParent !== null)
                .every(i => i.complete && i.naturalWidth > 0)""",
            timeout=60000,
        )
        page.wait_for_timeout(500)
        path = WORK / f"{name}.raw.png"
        page.screenshot(path=str(path))
        save_doc_image(Image.open(path), name)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)

        # Генерация с AI бустом.
        print("генерация:")
        # Окно выше обычного: в кадр должно войти поле переписанного промта.
        page, _ = u.fresh_page(browser, url, 1920, 1240)
        visible(page.get_by_label("LowQuality", exact=True)).check()
        visible(page.locator("textarea")).fill("a red fox in a snowy birch forest at golden hour")
        # «Rewrite now» показывает переписанный промт до генерации — ради него
        # снимок и нужен; галочку AI boost кнопка включает сама.
        u.click_text(page, "Rewrite now")
        page.wait_for_function(
            "() => [...document.querySelectorAll('textarea')].some(t => t.offsetParent && t.value.length > 80)",
            timeout=120000,
        )
        before = outputs()
        u.click_text(page, "Generate")
        wait_for_output(before, page)
        shot(page, "generate")
        page.close()

        # Правка по маске: кисть и результат рядом — широкое окно.
        print("правка по маске:")
        page, _ = u.fresh_page(browser, url, 2400, 1300)
        u.open_tab(page, 1)
        source = WORK / "showcase-mug.png"
        u.load_into_painter(page, source)
        visible(page.get_by_label("LowQuality", exact=True)).check()
        x0, y0, x1, y1 = mask_box
        steps = 7
        for row in range(steps):
            y = y0 + (y1 - y0) * row / (steps - 1)
            u.stroke(page, [(x0, y), (x1, y)], steps=20)
        visible(page.locator("textarea")).fill("a small potted succulent in a terracotta pot")
        before = outputs()
        u.click_text(page, "Apply edit")
        edited = wait_for_output(before, page)
        page.locator(f"#{u.PAINTER}").hover()
        shot(page, "edit-mask")
        save_doc_image(side_by_side([Image.open(source), Image.open(edited)]), "edit-mask-result")
        page.close()

        # Аннотация: палитра и пометки двумя цветами, без генерации.
        print("аннотация:")
        page, _ = u.fresh_page(browser, url, 1920, 1080)
        u.open_tab(page, 1)
        source = WORK / "showcase-fox.png"
        u.load_into_painter(page, source)
        visible(page.get_by_label("LowQuality", exact=True)).check()
        page.get_by_label("Annotation", exact=True).check()
        page.wait_for_timeout(500)
        u.stroke(page, [(0.33, 0.12), (0.8, 0.09), (0.83, 0.55), (0.36, 0.52), (0.33, 0.12)], steps=20)
        page.evaluate(f"""() => {{
            const host = [...document.querySelectorAll('#{u.PAINTER} *')].find(n => n.shadowRoot);
            host.shadowRoot.querySelector('.qp-swatch[data-color="#0000ff"]').click();
        }}""")
        u.stroke(page, [(0.07, 0.72), (0.26, 0.7), (0.27, 0.88), (0.08, 0.9), (0.07, 0.72)], steps=15)
        visible(page.locator("textarea")).fill(
            "make the lanterns in the red outline glow soft blue, and place a small teacup in the blue outline"
        )
        page.locator(f"#{u.PAINTER}").hover()
        shot(page, "edit-annotation")
        before = outputs()
        u.click_text(page, "Apply edit")
        annotated = wait_for_output(before, page)
        save_doc_image(side_by_side([Image.open(source), Image.open(annotated)]), "edit-annotation-result")
        page.close()

        # Расширение холста: исходник и результат.
        print("расширение холста:")
        page, _ = u.fresh_page(browser, url, 1920, 1080)
        u.open_tab(page, 1)
        source = WORK / "showcase-storefront.png"
        width = u.load_into_painter(page, source)["width"]
        visible(page.get_by_label("LowQuality", exact=True)).check()
        page.get_by_text("Outpaint", exact=True).first.click()
        page.wait_for_timeout(500)
        page.get_by_label("←", exact=True).check()
        page.get_by_label("→", exact=True).check()
        page.get_by_role("button", name="Outpaint", exact=True).last.click()
        page.wait_for_function(f"() => {u.API}.state().width > {width}", timeout=30000)
        page.wait_for_timeout(1000)
        # Описание всей сцены, а не операции: так и советует строка состояния.
        visible(page.locator("textarea")).fill(
            'a quiet city street corner at dusk in the rain, a small coffee shop with a glowing neon '
            'sign that reads "QWEN CAFE", old brick buildings and bare trees along the wet street, '
            "warm window light reflected on the pavement, photorealistic"
        )
        before = outputs()
        u.click_text(page, "Apply edit")
        widened = wait_for_output(before, page)
        assert Image.open(widened).width / Image.open(widened).height > 1.6, "холст не расширился"
        save_doc_image(side_by_side([Image.open(source), Image.open(widened)], height=520), "outpaint-result")
        page.close()

        # Галерея с карточкой параметров.
        print("галерея:")
        page, _ = u.fresh_page(browser, url, 1920, 1080)
        u.open_tab(page, 2)
        page.wait_for_timeout(1000)
        page.locator(".qs-browse img").first.click()
        page.wait_for_timeout(1500)
        shot(page, "gallery")
        page.close()

        # Настройки: примерный адрес вместо настоящего, токена нет.
        print("настройки:")
        example = WORK / "llm_endpoint.txt"
        example.write_text("llama.cpp\n192.168.1.10:8080\n", encoding="utf-8")
        config.ENDPOINT_FILE = example
        page, _ = u.fresh_page(browser, url, 1920, 1080)
        u.open_tab(page, 3)
        page.wait_for_timeout(1000)
        shot(page, "settings")
        page.close()

        browser.close()
    print(f"готово; рабочие файлы: {WORK.relative_to(ROOT)}")


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(description="Скриншоты и примеры для README")
    parser.add_argument("--phase", choices=["showcase", "ui"], required=True)
    parser.add_argument("--mask", type=float, nargs=4, default=(0.68, 0.64, 0.92, 0.86),
                        metavar=("X0", "Y0", "X1", "Y1"), help="область маски на кадре с кружкой, доли")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    configure()
    if args.phase == "showcase":
        phase_showcase()
    else:
        phase_ui(tuple(args.mask))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

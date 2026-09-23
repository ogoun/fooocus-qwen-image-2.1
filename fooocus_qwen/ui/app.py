"""Сборка интерфейса и запуск сервера."""

from __future__ import annotations

import logging
import webbrowser
from pathlib import Path

import gradio as gr

from .. import config
from . import layout, tab_edit, tab_gallery, tab_generate, tab_settings
from .i18n import LANGUAGES, Localizer, pick
from .state import Studio

LOGGER = logging.getLogger(__name__)

# Обход вкладок при загрузке страницы: см. пояснение у ``demo.load`` ниже.
# Шаг в 60 мс — не украшение: Svelte монтирует содержимое вкладки не в
# обработчике клика, а в следующем цикле отрисовки, и подряд идущие клики
# смонтировали бы только последнюю.
WARM_UP_TABS = """
() => {
    const tabs = Array.from(document.querySelectorAll('button[role="tab"]'));
    if (tabs.length < 2) { return; }
    let index = 0;
    const step = () => {
        if (index < tabs.length) {
            tabs[index].click();
            index += 1;
            setTimeout(step, 60);
        } else {
            tabs[0].click();
        }
    };
    step();
}
"""


def build(cfg: config.AppConfig, return_studio: bool = False):
    """Собирает интерфейс. С ``return_studio`` отдаёт ещё и состояние.

    Состояние нужно запуску, чтобы начать фоновую загрузку модели, и
    только ему: тесты собирают интерфейс без него и ничего лишнего не
    получают.
    """
    studio = Studio(cfg)
    localizer = Localizer(cfg.lang)

    # ``fill_width`` и ``fill_height`` — штатный способ Gradio отдать
    # приложению всё окно. Без первого Blocks центрирует себя и упирается во
    # внутренний потолок ширины: на мониторе 4K под интерфейс уходило 28 %
    # экрана, остальное — поля. Без второго холсты не тянутся по высоте.
    with gr.Blocks(
        title=pick("app_title", cfg.lang),
        analytics_enabled=False,
        fill_width=True,
        fill_height=True,
        # Уборка временных файлов: раз в час удаляется всё, что старше суток.
        # Кисть маски кладёт исходники и подготовленные сервером слои в
        # каталог загрузок Gradio, и без уборки он рос бы с каждой правкой.
        delete_cache=(3600, 86400),
    ) as demo:
        # Заголовка на странице нет намеренно. Название приложения стоит в
        # заголовке окна браузера (`title=` у Blocks), и повторять его строкой
        # в сто пикселей высотой, которую пользователь читает один раз в
        # жизни, — расточительство: эту высоту отнимали у холста каждый раз.
        #
        # Язык живёт в состоянии, а не в виджете: значение нужно каждому
        # обработчику как последний вход, а выпадающий список с подписью
        # «RU / EN» занимал отдельную строку ради двух букв. Видимое
        # управление — одна кнопка в правом верхнем углу, поверх полосы
        # вкладок и без собственной высоты (позиционирование в style.css).
        language = gr.State(cfg.lang)
        language_button = gr.Button(
            cfg.lang.upper(), size="sm", elem_classes=[layout.LANG], scale=0, min_width=0
        )

        def switch_language(current: str) -> list:
            """Переключает язык, надпись на кнопке и все подписи разом.

            По кругу, а не «включить английский»: языков два, и кнопка
            показывает тот, что сейчас выбран, — так же, как это делают
            переключатели языка в браузерах и почтовых клиентах.

            Подписи перерисовывает этот же обработчик, а не событие
            ``change`` у состояния, как было раньше. В Gradio 6.5.1 это
            событие не наступает, когда состояние меняется выходом другого
            обработчика: значение ``gr.State`` живёт на сервере, клиенту не
            передаётся, и обнаружить его изменение он не может. Проверено на
            чистом Gradio без нашего кода — после клика уходит один запрос
            вместо двух, кнопка меняет надпись, а подписи остаются на
            прежнем языке.

            Один обработчик вместо цепочки ещё и атомарен: состояние,
            кнопка и подписи приезжают одним ответом, и промежуточного
            состояния, где язык уже сменился, а интерфейс ещё нет, не
            существует.
            """
            order = list(LANGUAGES)
            following = order[(order.index(current) + 1) % len(order)] if current in order else order[0]
            return [following, gr.update(value=following.upper()), *localizer.updates(following)]


        with gr.Tabs():
            # Заголовок вкладки — такая же переводимая подпись, как и всё
            # остальное: без регистрации в localizer он застыл бы на языке
            # запуска и не откликался бы на переключатель RU/EN.
            generate_tab = localizer.bind(
                gr.Tab(pick("tab_generate", cfg.lang)),
                label=("Генерация", "Generate"),
            )
            with generate_tab:
                generate_components = tab_generate.build(studio, localizer, language)

            edit_tab = localizer.bind(
                gr.Tab(pick("tab_edit", cfg.lang)),
                label=("Редактирование", "Edit"),
            )
            with edit_tab:
                tab_edit.build(studio, localizer, language)

            gallery_tab = localizer.bind(
                gr.Tab(pick("tab_gallery", cfg.lang)),
                label=("Галерея", "Gallery"),
            )
            with gallery_tab:
                tab_gallery.build(studio, localizer, generate_components, language)

            settings_tab = localizer.bind(
                gr.Tab(pick("tab_settings", cfg.lang)),
                label=("Настройки", "Settings"),
            )
            with settings_tab:
                tab_settings.build(studio, localizer, language)

        # Переключатель языка делает две вещи. Явно — перерисовывает подписи
        # уже собранных компонентов. Неявно, но не менее важно — его значение
        # ходит последним входом в каждый обработчик, который что-то сообщает
        # пользователю: сообщения собираются в момент ответа, а не при сборке
        # интерфейса, поэтому строка состояния говорит на текущем языке, а не
        # на языке запуска.
        # Подпись вкладки Gradio обновляет только у той, чьё содержимое уже
        # смонтировано: у неоткрытых кнопка в полосе вкладок остаётся на
        # прежнем языке до первого захода внутрь. Проверено на чистом Gradio
        # 6.5.1 без нашего кода — обновление доходит до всех, применяется к
        # одной. Обновление контейнера ``gr.Tabs`` не помогает, а вот один
        # заход в каждую вкладку — да: после него переключатель переписывает
        # всю полосу разом.
        #
        # Поэтому при загрузке страницы вкладки открываются по очереди и
        # управление возвращается на первую. Это стоит четверть секунды на
        # старте и избавляет от наполовину переведённой полосы вкладок.
        demo.load(None, None, None, js=WARM_UP_TABS)

        # Регистрация именно здесь, после сборки вкладок: до неё
        # ``localizer.components`` ещё пуст, и клик обновлял бы одну кнопку.
        language_button.click(
            switch_language,
            language,
            [language, language_button, *localizer.components],
            queue=False,
        )

    return (demo, studio) if return_studio else demo


def stylesheet() -> str:
    """Читает файл стиля.

    Вся геометрия живёт в нём: подстановка значений из Python больше не
    нужна — размеры перестали дублироваться между модулем и стилем.
    """
    return (Path(__file__).parent / "style.css").read_text(encoding="utf-8")


def browser_url(cfg: config.AppConfig) -> str:
    """Адрес, по которому открывать интерфейс на этой же машине.

    ``0.0.0.0`` и ``::`` — это «слушать на всех адресах», а не адрес, по
    которому можно постучаться; для браузера они превращаются в петлю.

    Петля записана числом, а не именем: ``localhost`` на машине заказчика
    разрешается сначала в ``::1``, где сервер не слушает (он на IPv4), и
    каждое обращение платит 4.4 секунды до отката на IPv4 — на старте это
    заметно (docs/research/2026-09-22-zapusk-padal-na-proverke-localhost.md).
    """
    host = cfg.host if cfg.host not in ("0.0.0.0", "::", "") else "127.0.0.1"
    return f"http://{host}:{cfg.port}"


def open_in_browser(url: str) -> bool:
    """Открывает страницу. Неудача — не повод ронять запуск.

    Оболочку запускают и по SSH, и из планировщика, где открывать нечем:
    уронить работающий сервер из-за ненайденного браузера значило бы
    поменять местами главное и второстепенное.
    """
    LOGGER.info("Открываю интерфейс в браузере: %s", url)
    try:
        opened = bool(webbrowser.open(url, new=2))
    except Exception as error:  # noqa: BLE001 — причина важна, тип нет
        LOGGER.warning("Не удалось открыть браузер (%s). Откройте вручную: %s", error, url)
        return False
    if not opened:
        LOGGER.warning("Браузер не открылся. Откройте вручную: %s", url)
    return opened


def start(cfg: config.AppConfig, prepare=None) -> tuple[gr.Blocks, Studio]:
    """Собирает интерфейс и поднимает сервер, не блокируя поток.

    Отдельно от ``launch`` потому, что поднимать интерфейс нужно не только
    для работы: ``tools/ui_check.py`` проверяет его в настоящем браузере с
    подставным генератором — и обязан пройти ровно той же дорогой, что и
    боевой запуск, иначе он проверял бы не то приложение. ``prepare``
    получает состояние до подъёма сервера — там и подменяется генератор.
    """
    demo, studio = build(cfg, return_studio=True)
    if prepare is not None:
        prepare(studio)
    demo.queue(default_concurrency_limit=1)
    # В Gradio 6 параметр css переехал из конструктора Blocks в launch() — передача
    # его в конструктор всё ещё работает, но с предупреждением об устаревании.
    css = stylesheet()
    LOGGER.info("Интерфейс на http://%s:%s", cfg.host, cfg.port)
    # В Gradio 6.5.1 параметра show_api больше нет ни в Blocks, ни в launch() —
    # он был убран выше по течению. Ссылку «Use via API» и так прячет footer
    # в style.css, так что отдельно отключать нечего.
    # ``prevent_thread_lock`` — не косметика, а условие корректного старта.
    # Gradio заканчивает запуск контрольным ``HEAD`` на собственный адрес с
    # таймаутом в три секунды (``gradio/networking.py``, ``url_ok``); не
    # дождавшись ответа, он объявляет localhost недоступным и валит запуск
    # требованием ``share=True``. Пока прогрев шёл до ``launch``, греющий поток
    # читал тридцать три гигабайта весов ровно в эти секунды и отнимал у
    # собственного сервера и диск, и GIL: на прогретом файловом кэше проверка
    # успевала, на холодном — нет, и запуск падал через раз. Лечит это не
    # таймаут, а порядок — сначала сервер отвечает на проверку, потом греем.
    demo.launch(
        server_name=cfg.host,
        server_port=cfg.port,
        inbrowser=False,
        css=css,
        prevent_thread_lock=True,
    )
    return demo, studio


def launch(cfg: config.AppConfig) -> None:
    demo, studio = start(cfg)
    if cfg.open_browser:
        # Именно здесь, а не в run.ps1: скрипт оболочки не знает, когда
        # сервер поднялся, а Python с torch и diffusers стартует секунды.
        # К этой строке Gradio уже достучался до собственного сервера, то
        # есть страницу точно отдадут — окно не встретит «не удаётся
        # подключиться». И до прогрева весов: интерфейс открывается сразу,
        # модель догружается, пока человек набирает первый промт.
        open_in_browser(browser_url(cfg))

    if cfg.preload:
        # Пока пользователь открывает браузер и набирает промт, модель
        # успевает загрузиться: эти секунды больше не его.
        studio.preload_in_background()
    # Поток держим сами — тем же ``block_thread``, который позвал бы Gradio,
    # не попроси мы управление обратно. Он же ловит Ctrl+C и закрывает сервер.
    demo.block_thread()

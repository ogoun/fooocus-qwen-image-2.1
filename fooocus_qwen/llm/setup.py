"""Опрос об адресе и токене языковой модели при установке.

Файл ``llm_endpoint.txt`` в репозиторий не попадает — он единственное место,
где лежит секрет. Значит у того, кто забрал проект с GitHub, его нет, и
узнавать о нём из документации он не обязан: о нём спрашивает установка.

Отказ отвечать — полноправный ответ. AI-буст промтов необязателен, без
внешней модели работает всё остальное, поэтому пустой ввод означает
«пропустить», а не «записать пустоту».
"""

from __future__ import annotations

import getpass
import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from .endpoint import LlmEndpoint, load_endpoint, parse_endpoint_file

LOGGER = logging.getLogger(__name__)

HEADER = """# Адрес внешней языковой модели для AI-буста промтов.
# Формат свободный: имя бэкенда, адрес, токен. Порядок строк не важен.
# Файл можно править руками или во вкладке «Настройки».
"""


def render(host: str, token: str | None = None, backend: str = "") -> str:
    """Собирает содержимое файла. Пустой токен строкой не пишется вовсе."""
    lines = [HEADER.rstrip("\n")]
    if backend.strip():
        lines.append(backend.strip())
    lines.append(host.strip())
    if token:
        lines.append(f"token={token.strip()}")
    return "\n".join(lines) + "\n"


def probe_server(endpoint: LlmEndpoint) -> Sequence[str]:
    """Спрашивает у сервера список моделей. Отдельно — чтобы тесты не ходили в сеть."""
    from .client import LlmClient

    return LlmClient(endpoint, timeout=10.0).ping()


def configure(
    path: Path,
    ask: Callable[[str], str] = input,
    ask_secret: Callable[[str], str] = getpass.getpass,
    out: Callable[..., None] = print,
    probe: Callable[[LlmEndpoint], Sequence[str]] | None = probe_server,
) -> bool:
    """Спрашивает адрес и токен и пишет файл. ``True``, если файл записан.

    Существующая настройка сохраняется при пустом ответе: потерять её по
    недосмотру человека, нажавшего Enter, нельзя — второй раз токен ему никто
    не покажет. Токен не печатается ни при каких обстоятельствах, даже свой
    собственный: на экран смотрит не только тот, кто его вводит.
    """
    path = Path(path)
    ask, ask_secret = _forgiving(ask), _forgiving(ask_secret)
    current = _describe_current(path)
    if current:
        out(f"  сейчас настроено: {current}")
        question = "  Новый адрес языковой модели (Enter — оставить как есть): "
    else:
        out("  AI-буст промтов работает через внешний OpenAI-совместимый сервер")
        out("  (llama.cpp, vLLM, LM Studio). Без него работает всё остальное.")
        question = "  Адрес языковой модели, например 192.0.2.10:8000 (Enter — пропустить): "

    host = ask(question).strip()
    if not host:
        out("  Пропускаю: " + ("настройка осталась прежней" if current else "AI-буст будет отключён"))
        return False

    token = ask_secret("  Токен (Enter — без токена, ввод не отображается): ").strip()

    text = render(host, token or None)
    try:
        endpoint = parse_endpoint_file(text)
    except ValueError as error:
        out(f"  Не понял адрес «{host}»: {error}")
        out("  Файл не тронут, адрес можно задать позже во вкладке «Настройки».")
        return False

    path.write_text(text, encoding="utf-8")
    out(f"  Записано в {path.name}: {endpoint.base_url}" + (", токен задан" if token else ", без токена"))

    if probe is not None:
        _report_probe(endpoint, probe, out)
    return True


def _forgiving(ask: Callable[[str], str]) -> Callable[[str], str]:
    """Оборачивает вопрос так, что конец ввода и Ctrl+C означают «пропустить».

    Проверять ``sys.stdin.isatty()`` заранее недостаточно, и это не
    предположение: в Git Bash под Windows перенаправление из ``/dev/null``
    даёт ``isatty() == True`` — MSYS эмулирует его символьным устройством,
    которое Windows считает консольным. Установка, запущенная сценарием,
    падала на этом месте с ``EOFError`` уже после того, как зависимости
    поставлены. Поэтому отказ обрабатывается там, где он случается, а не
    предсказывается заранее.
    """

    def guarded(question: str) -> str:
        try:
            return ask(question)
        except (EOFError, KeyboardInterrupt):
            return ""

    return guarded


def _describe_current(path: Path) -> str:
    """Описывает уже настроенный сервер, не показывая токен."""
    try:
        endpoint = load_endpoint(path)
    except (OSError, ValueError):
        return ""
    return endpoint.base_url + (", токен задан" if endpoint.token else ", без токена")


def _report_probe(
    endpoint: LlmEndpoint,
    probe: Callable[[LlmEndpoint], Sequence[str]],
    out: Callable[..., None],
) -> None:
    """Проверяет связь. Неудача ничего не отменяет: сервер бывает выключен."""
    try:
        models = list(probe(endpoint))
    except Exception as error:  # noqa: BLE001 — причина важна человеку, а не типу
        out(f"  Сервер {endpoint.base_url} не откликнулся ({error}).")
        out("  Настройка сохранена — проверить связь можно позже во вкладке «Настройки».")
        return
    if models:
        out(f"  Сервер ответил, моделей доступно {len(models)}: {', '.join(models[:3])}")
    else:
        out("  Сервер ответил, но список моделей пуст — проверьте, что модель загружена.")

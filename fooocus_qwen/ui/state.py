"""Состояние приложения: модель, каталог стилей, связь с языковой моделью.

Модель грузится лениво и один раз. Держать тридцать три гигабайта весов ради
того, чтобы пользователь открыл вкладку настроек, незачем, а первый запрос всё
равно подождёт загрузки.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from PIL import Image

from .. import config
from ..llm import LlmClient, LlmError, load_endpoint
from ..prompting import boost
from ..prompting.styles import Style, load_styles
from .i18n import say

LOGGER = logging.getLogger(__name__)

# Общая группа очереди Gradio для всех событий, которые доходят до видеокарты.
#
# ``demo.queue(default_concurrency_limit=1)`` сам по себе приложение НЕ
# сериализует: Gradio заводит группу на идентичность функции
# (``BlockFunction.concurrency_id = concurrency_id or str(id(fn))``), поэтому
# обработчики «Сгенерировать» и «Применить правку» — разные объекты и попадают
# в разные группы, каждая со своим пределом в единицу. Нажатие обеих кнопок
# запускает обе. Явный общий идентификатор кладёт их в одну очередь, и вторая
# работа не покидает очередь, пока первая не закончилась.
#
# Это вторая линия обороны: настоящий инвариант «одна генерация за раз»
# держит замок внутри ``engine.generator.Generator`` (см. его докстринг).
# Здесь — то, что не даёт очереди выпустить работу, которая всё равно
# немедленно упрётся в этот замок.
GPU_CONCURRENCY_ID = "qwen-image-gpu"

# Насколько длинную цитату из исключения показывать в строке состояния.
# Сообщение torch о нехватке видеопамяти занимает несколько строк со сводкой
# аллокатора — в однострочном поле статуса от него нужна только первая фраза,
# полный текст уходит в журнал вместе со стеком.
_MESSAGE_LIMIT = 220


def _quote(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return text[: _MESSAGE_LIMIT - 1] + "…" if len(text) > _MESSAGE_LIMIT else text


def _is_out_of_memory(error: BaseException) -> bool:
    """Нехватка видеопамяти — по типу исключения, а не по тексту сообщения.

    ``torch`` импортируется лениво и только здесь: ``ui/`` не тянет его при
    сборке интерфейса, иначе открыть вкладку настроек стоило бы загрузки
    библиотеки (см. ``test_studio_lazy.py``). К моменту, когда сюда попадает
    сбой генерации, ``torch`` давно импортирован движком, и это обращение
    ничего не стоит.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover — движок без torch не запустится
        return False
    out_of_memory = getattr(torch, "OutOfMemoryError", None)
    return out_of_memory is not None and isinstance(error, out_of_memory)


def describe_failure(error: BaseException, lang: str) -> str:
    """Читаемая строка состояния вместо сырого traceback в тосте.

    Отсутствующие веса, испорченный ``model_index.json`` и нехватка
    видеопамяти в цикле денойзинга — это то, с чем пользователь способен
    что-то сделать, если ему сказать, что случилось. Стек вызовов в тосте не
    говорит ничего и выглядит как падение приложения. Установщик считает, что
    «установилось» значит «запустится»; работающему приложению разумно
    держаться того же стандарта.

    Сам текст исключения не переводится: он приходит из torch, diffusers или
    операционной системы и всегда английский. Переводится обрамление — то,
    что объясняет пользователю, что делать.
    """
    if _is_out_of_memory(error):
        return say("failure_out_of_memory", lang, error=_quote(error))
    if isinstance(error, FileNotFoundError):
        return say("failure_file_not_found", lang, error=_quote(error))
    if isinstance(error, OSError):
        return say("failure_io", lang, error=_quote(error))
    return say("failure_other", lang, kind=type(error).__name__, error=_quote(error))


class Studio:
    """Единственный владелец тяжёлых ресурсов."""

    def __init__(self, cfg: config.AppConfig) -> None:
        self.config = cfg
        self.catalogue: dict[str, Style] = load_styles(config.STYLES_DIR)
        self._lock = threading.Lock()
        self._generator: Any = None
        self._residency: Any = None
        self._cache: Any = None

    @property
    def generator(self):
        """Возвращает генератор, загрузив модель при первом обращении."""
        if self._generator is None:
            with self._lock:
                if self._generator is None:
                    from ..engine import loader
                    from ..engine.generator import Generator

                    pipe, residency, cache = loader.load(
                        self.config.model_dir, pin_memory=self.config.pin_memory
                    )
                    self._generator = Generator(pipe, residency, cache, self.catalogue)
                    self._residency = residency
                    self._cache = cache
        return self._generator

    @property
    def model_loaded(self) -> bool:
        return self._generator is not None

    def memory_report(self, lang: str) -> str:
        if self._residency is None:
            return say("model_not_loaded", lang)
        stats = self._residency.stats()
        return say(
            "memory_report",
            lang,
            allocated=stats["allocated_gib"],
            reserved=stats["reserved_gib"],
            swaps=int(stats["swaps"]),
            hits=self._cache.hits,
            misses=self._cache.misses,
        )

    def run_generation(
        self, request: Any, lang: str, progress: Any = None
    ) -> tuple[list[Any], str | None]:
        """Генерация, у которой сбой — это строка состояния, а не traceback.

        Возвращает пару (результаты, сообщение об ошибке или ``None``). Второй
        элемент непуст ровно тогда, когда генерация не состоялась, поэтому
        вызывающая вкладка не обязана отличать «прервали» от «упало» по
        пустому списку.

        После любого сбоя размещение весов приводится в порядок: сбой мог
        оборвать перестановку моделей где угодно, в том числе внутри цикла
        денойзинга по нехватке памяти.
        """
        try:
            return self.generator.generate(request, progress=progress), None
        except Exception as error:  # noqa: BLE001 — тост с traceback хуже строки статуса
            LOGGER.exception("Генерация не выполнена")
            self.recover_residency()
            return [], describe_failure(error, lang)

    def recover_residency(self) -> None:
        """Возвращает веса на штатные места после сбоя.

        Без этого трансформер может остаться на хосте, а следующий запрос с
        тем же промтом попадёт в кэш эмбеддингов, не зайдёт в
        ``text_encoder_resident()`` — и цикл денойзинга пойдёт по весам,
        которых на видеокарте нет. Только перезапуск помогал бы.
        """
        if self._residency is None:
            return
        try:
            self._residency.restore()
        except Exception:  # noqa: BLE001
            LOGGER.exception("Не удалось вернуть веса на штатные места")

    def llm_client(self) -> LlmClient:
        """Создаёт клиента заново: файл адреса правится без перезапуска."""
        endpoint = load_endpoint(config.ENDPOINT_FILE)
        client = LlmClient(endpoint)
        models = client.ping()
        return LlmClient(endpoint, model=models[0] if models else "")

    def boost_prompt(
        self,
        prompt: str,
        mode: str,
        lang: str,
        references: list[Image.Image] | None = None,
    ) -> tuple[str, str | None, str]:
        """Возвращает переписанный промт, соотношение сторон и сообщение о результате."""
        try:
            client = self.llm_client()
            result = boost.boost(
                client, prompt, mode=mode, prompt_dir=config.SYSTEM_PROMPT_DIR, references=references
            )
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            LOGGER.warning("AI-буст не выполнен: %s", error)
            return prompt, None, say("boost_failed", lang, error=error)
        return result.prompt, result.wh_ratio, say("boost_done", lang)

    def describe_image(self, image: Image.Image, lang: str) -> tuple[str, str]:
        try:
            text = boost.describe(self.llm_client(), image, config.SYSTEM_PROMPT_DIR)
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            LOGGER.warning("Описание не выполнено: %s", error)
            return "", say("describe_failed", lang, error=error)
        return text, say("describe_done", lang)

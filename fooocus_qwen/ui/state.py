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

    def memory_report(self) -> str:
        if self._residency is None:
            return "модель ещё не загружена"
        stats = self._residency.stats()
        return (
            f"видеопамять: {stats['allocated_gib']:.1f} ГиБ занято, "
            f"{stats['reserved_gib']:.1f} ГиБ зарезервировано; "
            f"перестановок энкодера: {int(stats['swaps'])}; "
            f"кэш промтов: {self._cache.hits} попаданий / {self._cache.misses} промахов"
        )

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
            return prompt, None, f"AI буст не выполнен: {error}"
        return result.prompt, result.wh_ratio, "AI буст выполнен"

    def describe_image(self, image: Image.Image) -> tuple[str, str]:
        try:
            return boost.describe(self.llm_client(), image, config.SYSTEM_PROMPT_DIR), "Описание готово"
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            LOGGER.warning("Описание не выполнено: %s", error)
            return "", f"Описание не выполнено: {error}"

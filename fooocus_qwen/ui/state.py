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
from .. import settings as settings_module
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


def seeds_phrase(seeds: list[int], lang: str) -> str:
    """«Сид: 5» для одной картинки, «Сиды: 5, 6» для нескольких.

    Сид в строке состояния — то, чем результат повторяют: его показывают и
    генерация, и правка (правка раньше молчала, и повторить удачную правку
    было нечем, кроме как лезть в метаданные файла).
    """
    joined = ", ".join(str(seed) for seed in seeds)
    return say("seed_one" if len(seeds) == 1 else "seed_many", lang, seeds=joined)


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
        # Модели, не принявшие изображения: (адрес, имя модели). Следующий буст
        # к ним сразу идёт одним текстом, без заведомо неудачного запроса с
        # картинками. Живёт до перезапуска: сменил модель на сервере под тем
        # же именем — перезапуск оболочки вернёт попытку.
        self._text_only_models: set[tuple[str, str]] = set()
        self._pose_detector: Any = None

    @property
    def generator(self):
        """Возвращает генератор, загрузив модель при первом обращении.

        Точность трансформера и механизм внимания берутся из настроек в
        момент загрузки (``user/settings.json``): смена точности — это
        перезагрузка модели (``switch_precision``), а механизм внимания
        меняется на лету (``set_sage_attention``).
        """
        if self._generator is None:
            with self._lock:
                if self._generator is None:
                    from ..engine import fetch, loader
                    from ..engine.generator import Generator
                    from ..engine.turbo import TurboAdapter

                    chosen = settings_module.load()
                    int8_file = None
                    if chosen.precision == settings_module.PRECISION_INT8:
                        int8_file = config.INT8_DIR / fetch.INT8_FILE
                    pipe, residency, cache = loader.load(
                        self.config.model_dir,
                        pin_memory=self.config.pin_memory,
                        int8_file=int8_file,
                        sage_attention=chosen.sage_attention,
                    )
                    turbo = TurboAdapter(pipe, residency, config.TURBO_DIR)
                    # Полосы шагов diffusers в интерфейсе не нужны: ход
                    # генерации рисует свой обратный вызов, а
                    # ``gr.Progress(track_tqdm=True)`` (ради полосы скачивания
                    # весов) подхватил бы и их, перебивая подпись.
                    pipe.set_progress_bar_config(disable=True)
                    self._generator = Generator(pipe, residency, cache, self.catalogue, turbo=turbo)
                    self._residency = residency
                    self._cache = cache
        return self._generator

    # --- производительность: точность, внимание, turbo -------------------------

    def weights_for(self, preset, lang: str, progress: Any = None) -> str | None:
        """Докачивает веса, которых требует пресет. Сообщение об ошибке или None.

        Пресету Turbo нужен адаптер (1.3 ГБ); при первом выборе он качается
        прямо перед генерацией, и строка прогресса говорит об этом —
        полосу скачивания рисует ``gr.Progress(track_tqdm=True)``.
        """
        if not getattr(preset, "turbo", False) or self.turbo_weights_present():
            return None
        if progress is not None:
            progress(0, desc=say("turbo_downloading", lang))
        try:
            self.ensure_turbo_weights()
        except Exception as error:  # noqa: BLE001 — сеть, диск, Hugging Face: всё это строка состояния
            LOGGER.exception("Веса turbo не скачались")
            return say("turbo_download_failed", lang, error=_quote(error))
        return None

    def ensure_turbo_weights(self) -> bool:
        """Докачивает веса turbo, если их нет. ``True`` — что-то качалось."""
        from ..engine import fetch

        return fetch.ensure_turbo(config.TURBO_DIR)

    def pose_detector(self):
        """Распознавание позы на фото — одно на процесс: сессии ONNX дороги в создании."""
        if self._pose_detector is None:
            from ..poses.detect import PoseDetector

            self._pose_detector = PoseDetector(config.DWPOSE_DIR)
        return self._pose_detector

    def turbo_weights_present(self) -> bool:
        from ..engine import fetch

        return not fetch.missing_extra(config.TURBO_DIR, fetch.TURBO_FILES)

    def ensure_precision_weights(self, precision: str) -> bool:
        """Докачивает веса выбранной точности. ``True`` — что-то качалось.

        Для INT8 это файл Unsloth; для bf16 — шарды bf16-трансформера,
        которых нет у того, кто ставил приложение сразу в INT8.
        """
        from ..engine import fetch

        if precision == settings_module.PRECISION_INT8:
            return fetch.ensure_int8(config.INT8_DIR)
        return fetch.ensure_model(self.config.model_dir, include_transformer=True)

    def switch_precision(self, precision: str) -> None:
        """Сохраняет выбор точности и выгружает модель: следующая загрузка — в новой.

        Веса выбранной точности обязаны быть на месте заранее
        (``ensure_precision_weights``): иначе модель выгрузилась бы, а
        загрузиться обратно не смогла бы.
        """
        settings_module.update(precision=precision)
        self.unload()

    def unload(self) -> None:
        """Выгружает модель, дождавшись конца текущей генерации."""
        generator = self._generator
        if generator is None:
            return
        with self._lock, generator.exclusive():
            self._generator = None
            self._residency = None
            self._cache = None
        del generator
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass
        LOGGER.info("Модель выгружена")

    def set_sage_attention(self, enabled: bool) -> bool:
        """Сохраняет выбор и, если модель загружена, применяет его сразу.

        Возвращает, работает ли SageAttention на самом деле: выбор без
        установленного пакета сохраняется, но внимание остаётся штатным.
        """
        from ..engine import attention

        settings_module.update(sage_attention=enabled)
        if self._generator is None:
            return enabled and attention.sage_available()
        with self._generator.exclusive():
            return attention.apply(self._generator.pipe.transformer, enabled)

    def preload_in_background(self) -> None:
        """Начинает загрузку модели, не дожидаясь первого запроса.

        Загрузка занимает от двадцати до пятидесяти секунд: чтение весов с
        диска и подготовка закреплённых копий на хосте (последняя измеренно
        стоит около десяти секунд и окупается уже на двух промахах кэша —
        см. ``tools/experiments/pinning_cost.py``). До этой правки пользователь
        платил их при первой же генерации, ничего при этом не видя.

        Поток демонский и ошибку наружу не выносит: если веса не читаются,
        сказать об этом некому — интерфейс ещё никто не трогал. Настоящий
        запрос упрётся в ту же ошибку и покажет её строкой состояния, как и
        раньше. Повторной загрузки при этом не будет: ``generator`` защищён
        тем же замком и двойной проверкой.
        """
        if self._generator is not None:
            return

        def warm() -> None:
            try:
                self.generator  # noqa: B018 — обращение и есть загрузка
                LOGGER.info("Модель загружена заранее, первая генерация начнётся сразу")
            except Exception:  # noqa: BLE001 — показывать некому, запрос повторит
                LOGGER.exception("Фоновая загрузка модели не удалась")

        threading.Thread(target=warm, name="preload", daemon=True).start()

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

        Маршрутизировано через ``Generator.recover()``, а не прямое обращение
        к ``ResidencyManager.restore()``: восстановление обязано брать тот же
        замок, что и ``generate()``, иначе оно могло бы вклиниться в чужую
        перестановку моделей прямо посреди ``text_encoder_resident()``. Раньше
        это было безопасно только потому, что общая группа очереди Gradio не
        выпускает второй обработчик, пока первый не вернётся, — а полагаться
        на слой очереди в инварианте «одна операция с видеокартой за раз»
        запрещено тем же рассуждением, что и в ``engine/generator.py``.
        """
        if self._generator is None:
            return
        self._generator.recover()

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
        tags: list[str] | None = None,
    ) -> tuple[str, str | None, str]:
        """Возвращает переписанный промт, соотношение сторон и сообщение о результате.

        ``tags`` — теги изображений, если они идут не подряд (см.
        ``boost.build_user_message``).
        """
        try:
            client = self.llm_client()
            key = (client.base_url, client.model)
            result = boost.boost(
                client, prompt, mode=mode, prompt_dir=config.SYSTEM_PROMPT_DIR,
                references=references, send_images=key not in self._text_only_models, tags=tags,
            )
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            LOGGER.warning("AI-буст не выполнен: %s", error)
            return prompt, None, say("boost_failed", lang, error=error)
        if result.images_skipped:
            self._text_only_models.add(key)
            return result.prompt, result.wh_ratio, say("boost_done_text_only", lang)
        return result.prompt, result.wh_ratio, say("boost_done", lang)

    def describe_image(self, image: Image.Image, lang: str) -> tuple[str, str]:
        try:
            text = boost.describe(self.llm_client(), image, config.SYSTEM_PROMPT_DIR)
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            LOGGER.warning("Описание не выполнено: %s", error)
            return "", say("describe_failed", lang, error=error)
        return text, say("describe_done", lang)

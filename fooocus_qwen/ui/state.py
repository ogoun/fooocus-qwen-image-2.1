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

        Точность трансформера, профиль памяти и механизм внимания берутся из
        настроек в момент загрузки (``user/settings.json``): смена точности
        или профиля — это перезагрузка модели (``switch_precision``,
        ``switch_memory_profile``), а механизм внимания меняется на лету
        (``set_sage_attention``).

        Профиль «low» (карты на 6–16 ГБ) — трансформер не покидает карту, а
        энкодер в INT8 подаётся по блоку. INT8-копия энкодера собирается
        здесь же, если её ещё нет: около 20 секунд на видеокарте, один раз.
        """
        if self._generator is None:
            with self._lock:
                if self._generator is None:
                    from ..engine import fetch, loader
                    from ..engine.generator import Generator
                    from ..engine.turbo import Turbo4Transformer, TurboAdapter

                    chosen = settings_module.load()
                    plan = self.weights_plan()
                    # Окружение могли поставить до того, как выбор понадобился.
                    plan.ensure_packages()
                    plan.ensure_text_encoder()
                    pipe, residency, cache = loader.load(
                        self.config.model_dir,
                        pin_memory=self.config.pin_memory,
                        sage_attention=chosen.sage_attention,
                        **plan.loader_arguments(),
                    )
                    turbo = TurboAdapter(pipe, residency, config.TURBO_DIR, light=plan.low)
                    turbo4 = Turbo4Transformer(
                        pipe, residency, config.GGUF_DIR / fetch.TURBO4_FILE, config.TURBO_DIR,
                        self.config.model_dir,
                        base_loader=loader.base_transformer_loader(
                            self.config.model_dir, int8_file=plan.int8_file, gguf_file=plan.gguf_file
                        ),
                        # Механизм внимания — по настройке на момент подмены:
                        # его меняют на лету, и подменённый должен следовать.
                        prepare=lambda module: loader.prepare_transformer(
                            module, settings_module.load().sage_attention, plan.policy, residency.device
                        ),
                        turbo=turbo,
                    )
                    # Полосы шагов diffusers в интерфейсе не нужны: ход
                    # генерации рисует свой обратный вызов, а
                    # ``gr.Progress(track_tqdm=True)`` (ради полосы скачивания
                    # весов) подхватил бы и их, перебивая подпись.
                    pipe.set_progress_bar_config(disable=True)
                    self._generator = Generator(pipe, residency, cache, self.catalogue, turbo=turbo, turbo4=turbo4)
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
        from ..engine import presets

        if getattr(preset, "transformer", "") == presets.TURBO4:
            if self.turbo4_weights_present():
                return None
            message, fetch_weights = "turbo4_downloading", self.ensure_turbo4_weights
        elif getattr(preset, "turbo", False) and not self.turbo_weights_present():
            message, fetch_weights = "turbo_downloading", self.ensure_turbo_weights
        else:
            return None
        if progress is not None:
            progress(0, desc=say(message, lang))
        try:
            fetch_weights()
        except Exception as error:  # noqa: BLE001 — сеть, диск, Hugging Face: всё это строка состояния
            LOGGER.exception("Turbo weights download failed")
            return say("turbo_download_failed", lang, error=_quote(error))
        return None

    def outpaint_lora(self, lang: str, progress: Any = None):
        """Лора outpaint, скачанная при первом расширении: ``(ResolvedLora, None)`` или ``(None, сообщение)``."""
        from ..engine import fetch, lora

        path = config.OUTPAINT_DIR / fetch.OUTPAINT_FILE
        if not path.is_file():
            if progress is not None:
                progress(0, desc=say("outpaint_downloading", lang))
            try:
                fetch.ensure_outpaint(config.OUTPAINT_DIR)
            except Exception as error:  # noqa: BLE001 — сеть, диск, Hugging Face: строка состояния
                LOGGER.exception("Outpaint LoRA download failed")
                return None, say("outpaint_download_failed", lang, error=_quote(error))
        return lora.ResolvedLora(path.stem, 1.0, path, lora.short_hash(path)), None

    def weights_plan(self, **changes):
        """Следствия выбора из настроек (``engine/plan.py``); ``changes`` — поправки к нему."""
        from ..engine import plan

        return plan.resolve(model_dir=self.config.model_dir, **changes)

    def memory_profile(self) -> str:
        """Профиль памяти, в котором работает (или будет работать) модель: «high» или «low».

        «auto» из настроек решается по объёму видеопамяти
        (``settings.resolve_profile``).
        """
        return self.weights_plan().profile

    def ensure_turbo_weights(self) -> bool:
        """Докачивает веса turbo, если их нет. ``True`` — что-то качалось."""
        from ..engine import fetch

        return fetch.ensure_turbo(config.TURBO_DIR, light=self.weights_plan().low)

    def turbo4_weights_present(self) -> bool:
        from ..engine import fetch

        return not fetch.turbo4_missing(config.GGUF_DIR, config.TURBO_DIR)

    def ensure_turbo4_weights(self) -> bool:
        """Докачивает трансформер Turbo4 (4.2 ГБ). ``True`` — что-то качалось."""
        from ..engine import fetch

        return fetch.ensure_turbo4(config.GGUF_DIR, config.TURBO_DIR)

    def pose_detector(self):
        """Распознавание позы на фото — одно на процесс: сессии ONNX дороги в создании."""
        if self._pose_detector is None:
            from ..poses.detect import PoseDetector

            self._pose_detector = PoseDetector(config.DWPOSE_DIR)
        return self._pose_detector

    def turbo_weights_present(self) -> bool:
        from ..engine import fetch

        return not fetch.missing_extra(config.TURBO_DIR, self.weights_plan().turbo_files)

    def ensure_precision_weights(self, precision: str) -> bool:
        """Докачивает веса выбранной точности. ``True`` — что-то качалось.

        Для INT8 это файл Unsloth; для GGUF — файл варианта (Unsloth); для
        bf16 — шарды bf16-трансформера, которых нет у того, кто ставил
        приложение сразу в INT8 или GGUF. bf16-шарды энкодера при этом не
        качаются, если его INT8-копия уже собрана (``WeightsPlan``).
        """
        return self.weights_plan(precision=precision).ensure_weights()

    def switch_precision(self, precision: str) -> None:
        """Сохраняет выбор точности и выгружает модель: следующая загрузка — в новой.

        Веса выбранной точности обязаны быть на месте заранее
        (``ensure_precision_weights``): иначе модель выгрузилась бы, а
        загрузиться обратно не смогла бы.
        """
        settings_module.update(precision=precision)
        self.unload()

    def switch_memory_profile(self, profile: str) -> None:
        """Сохраняет профиль памяти и выгружает модель: следующая загрузка — в новом.

        Профиль меняет раскладку целиком (кто живёт на карте, в каком виде
        энкодер), на загруженной модели его не переключить.
        """
        settings_module.update(memory_profile=profile)
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
        LOGGER.info("Model unloaded")

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
                LOGGER.info("Model preloaded; the first generation will start immediately")
            except Exception:  # noqa: BLE001 — показывать некому, запрос повторит
                LOGGER.exception("Background model loading failed")

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
            LOGGER.exception("Generation failed")
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
            LOGGER.warning("AI boost failed: %s", error)
            return prompt, None, say("boost_failed", lang, error=error)
        if result.images_skipped:
            self._text_only_models.add(key)
            return result.prompt, result.wh_ratio, say("boost_done_text_only", lang)
        return result.prompt, result.wh_ratio, say("boost_done", lang)

    def describe_for_outpaint(self, canvas: Image.Image, lang: str) -> tuple[str, str]:
        """Сцена для расширения кадра словами, как подписи, на которых учили лору outpaint.

        Языковая модель видит серый холст и описывает сцену так, будто она
        занимает весь кадр (``outpaint.DESCRIBE_INSTRUCTION``). Нет модели или
        она не ответила — пустое описание и сообщение: расширение работает и
        без него.
        """
        from ..imaging import outpaint

        try:
            text = self.llm_client().complete(
                outpaint.DESCRIBE_INSTRUCTION, "Describe this image.", images=[canvas]
            ).strip()
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            LOGGER.warning("Outpaint description failed: %s", error)
            return "", say("outpaint_describe_failed", lang, error=error)
        return text, say("outpaint_described", lang)

    def describe_image(self, image: Image.Image, lang: str) -> tuple[str, str]:
        try:
            text = boost.describe(self.llm_client(), image, config.SYSTEM_PROMPT_DIR)
        except (LlmError, FileNotFoundError, ValueError, OSError) as error:
            LOGGER.warning("Describe failed: %s", error)
            return "", say("describe_failed", lang, error=error)
        return text, say("describe_done", lang)

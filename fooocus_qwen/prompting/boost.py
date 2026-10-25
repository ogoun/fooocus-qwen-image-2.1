"""AI-буст: переписывание пользовательского промта внешней языковой моделью.

Системные промты взяты из официального репозитория Qwen и лежат файлами —
пользователь волен их править, поэтому они перечитываются при каждом запросе.

Официальные промты писались под дообученную модель Qwen-Image-2.1-PE-T2I и
требуют строгий JSON в ответе. Обычная модель на llama.cpp нередко оборачивает
его в блок кода или отвечает прозой. Терять из-за этого генерацию нельзя:
при неудаче разбора берём текст как есть.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image

from ..llm import LlmClient, LlmImagesRejected

LOGGER = logging.getLogger(__name__)

MODE_T2I = "t2i"
MODE_EDIT = "edit"

_PROMPT_FILES = {
    MODE_T2I: "system_prompt_t2i.txt",
    MODE_EDIT: "system_prompt_edit.txt",
}
_DESCRIBE_FILE = "system_prompt_describe.txt"

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# Рассуждения модели перед ответом. Официальные переписыватели Qwen-Image-2.1
# (PE-T2I, PE-I2I) думают в блоке <think> и только потом выдают JSON. Обычно
# сервер уносит рассуждения в отдельное поле, но не всякий: оставшись в тексте,
# они путали бы поиск JSON — в рассуждениях тоже бывают фигурные скобки.
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Иероглифы и текст в кавычках — для проверки языка описания.
_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”|«[^»]*»|「[^」]*」|『[^』]*』')

# Пометка к запросу правки, когда изображений переписыватель не видит.
# Системный промт правки целиком построен на чтении картинки; пометка говорит
# модели правду о том, что картинки нет. Опыт tools/experiments/
# boost_text_only.py (qwen3.8-27b, 80 ответов на вариант, перепроверен
# 2026-09-26 — в первом прогоне мера выдумок была сломана и молчала): без
# пометки переписыватель выдумывает содержимое кадра («сохранить здания,
# горы, людей») в 51 % ответов, с пометкой — в 4 %; ответ короче и быстрее,
# 5.0 с против 7.0. Пометка по-английски: так написан системный промт.
#
# Язык описания пометка не переопределяет, а напоминает правило (A). Прежняя
# редакция кончалась словами «Write the description in the language of the
# instruction» — для английских инструкций опыта это совпадало с правилом, а
# для русских противоречило ему, и пометка, стоящая последней, побеждала:
# tools/experiments/boost_language.py (qwen3.8-27b, русские инструкции) —
# 75 % ответов по-русски с прежней редакцией, 0 % без пометки и 0 % из 20
# с нынешней.
BLIND_NOTE = (
    "\n\nNote: the input image is NOT available to you — you cannot see it. "
    "Do not guess, describe or name anything the image might contain. Rewrite "
    "only the instruction itself: make the requested change concrete and say "
    "that everything else in the image stays exactly as it is, without listing "
    "what that is. Language decision (A) still applies: write the description "
    "in English unless the instruction is in Chinese."
)

# Дополнения к официальным промтам. Файлы промтов — поставка QwenLM байт в
# байт (tools/fetch_system_prompts.py), поэтому то, чего в них нет, живёт
# здесь. Обе пометки — из сравнения с промтом, который предложила сторонняя
# модель: tools/experiments/boost_prompt_variants.py (qwen3.8-27b, 2026-10-25,
# по 4 ответа на вариант), docs/research/2026-10-25-boost-prompt.md.
#
# Прозрачность. README Qwen-Image-2.1 даёт для RGBA готовый формат промта, а
# официальный промт T2I о нём молчит: на «прозрачный фон» переписыватель
# писал «a completely transparent background», однажды — «checkerboard
# pattern», и формата не выдал ни разу из 4. С пометкой — 4 из 4, ложных
# срабатываний на обычных запросах 0 из 8. Только для генерации: в промте
# правки слово «прозрачность» включает альфу даже с отрицанием
# (docs/research/2026-09-22-maska-kak-alfa.md).
RGBA_OPEN = "This is an RGBA image with transparency."
RGBA_CLOSE = "The image has alpha channel and the background is transparent."
T2I_NOTE = (
    "\n\n## Transparent images\n\n"
    "If the user asks for a transparent background, a cut-out, or an isolated asset with "
    f"no background, the description begins with exactly \"{RGBA_OPEN}\" and ends with "
    f"exactly \"{RGBA_CLOSE}\" — this is the prompt format Qwen-Image-2.1 is trained on "
    "for transparent output. Between them describe only the subject: no backdrop, floor, "
    "surface or background colour, and no white or checkerboard background. Never use "
    "these two sentences when transparency was not asked for."
)

# Режим «Аннотация»: цветные пометки нарисованы прямо на исходнике, а
# официальный промт правки о них ничего не знает. Без пометки 2 из 4 ответов
# вышли с изъянами (один по-русски, в одном рассуждения вместо JSON), в
# среднем 30 с; с пометкой 4 из 4 называют цвет и убирают метку, 8 с.
ANNOTATION_NOTE = (
    "\n\nNote: the coloured strokes, circles and arrows drawn over the input image are the "
    "user's annotation marks. They only show where each edit applies. Refer to each marked "
    "area by its colour and by the object under it, and state that the marks themselves "
    "are gone from the result."
)


@dataclass(frozen=True)
class BoostResult:
    """Результат переписывания.

    ``wh_ratio`` и ``ratio_follow`` взаимоисключающие: первый задаёт соотношение
    явно, второй велит наследовать его от указанного референса.
    """

    prompt: str
    wh_ratio: str | None = None
    ratio_follow: str | None = None
    raw: str = ""
    # Изображения были, но переписыватель их не видел: модель их не читает
    # (или о ней это уже известно). Промт переписан по одному тексту.
    images_skipped: bool = False


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def parse_response(text: str) -> BoostResult:
    raw = _THINK.sub("", text).strip()
    if not raw:
        return BoostResult(prompt="", raw=text)

    match = _JSON_OBJECT.search(raw)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            # Ответ по форме, но без текста — это пустой ответ, а не проза.
            # Раньше он уходил «как есть», и промтом становился сам JSON
            # (`{"rewritten_prompt": "", …}` — увидено в опыте boost_language).
            # Пустой промт `_ask` переспрашивает, а затем оставляет исходный.
            return BoostResult(
                prompt=_clean(payload.get("rewritten_prompt")) or "",
                wh_ratio=_clean(payload.get("wh_ratio")),
                ratio_follow=_clean(payload.get("ratio_follow")),
                raw=text,
            )

    LOGGER.warning("Rewriter response is not JSON; using the text as is")
    return BoostResult(prompt=raw, raw=text)


def off_language(instruction: str, rewritten: str) -> bool:
    """Описание не по-английски при инструкции без иероглифов.

    Оба системных промта требуют: инструкция на китайском — описание на
    китайском, на любом другом языке — по-английски. Модель изредка
    нарушает правило: отвечает по-китайски на английскую инструкцию
    (увидено вживую на qwen3.8-27b) или на языке самой инструкции — русском
    и любом другом. Поэтому проверяется не «нет иероглифов», а «все буквы
    латинские»: английский узнаётся по письменности, а чужих языков много.
    Текст в кавычках не считается: это надписи, которые модель изображения
    нарисует, и их язык решается отдельным правилом (B).
    """
    if _CJK.search(instruction):
        return False
    prose = _QUOTED.sub("", rewritten)
    return any(char.isalpha() and not _latin(char) for char in prose)


def _latin(char: str) -> bool:
    return unicodedata.name(char, "").startswith("LATIN")


def build_user_message(prompt: str, reference_count: int, tags: list[str] | None = None) -> str:
    """Собирает сообщение пользователя для переписывателя.

    При двух и более референсах спецификация Qwen требует адресовать их тегами
    ``<imageN>``; при единственном теги запрещены.

    ``tags`` — теги изображений, как их увидит модель, если они идут не
    подряд. Так бывает при правке с маской: исходник — ``<image1>``, маска —
    ``<image2>`` (переписывателю её не показывают), референсы — с
    ``<image3>``. Нумерация подряд назвала бы первый референс ``<image2>``, и
    переписанный промт ссылался бы на маску.
    """
    if tags is None:
        tags = [f"<image{index}>" for index in range(1, reference_count + 1)]
    if len(tags) < 2:
        return prompt
    return f"Input images: {' '.join(tags)}\n\nInstruction: {prompt}"


def _read_system_prompt(prompt_dir: Path, filename: str) -> str:
    path = prompt_dir / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise FileNotFoundError(f"System prompt not found {path}: {error}") from error


def boost(
    client: LlmClient,
    prompt: str,
    *,
    mode: str,
    prompt_dir: Path,
    references: list[Image.Image] | None = None,
    send_images: bool = True,
    tags: list[str] | None = None,
    annotation: bool = False,
) -> BoostResult:
    """Переписывает промт; изображения — подспорье, а не условие.

    Картинки помогают переписывателю правки понять, что на кадре, но промт
    переписывается и без них. Текстовая модель отвечает на запрос с
    изображением ошибкой сервера — тогда тот же запрос уходит одним текстом.
    Раньше отказ считался провалом буста целиком, и с текстовой моделью AI
    буст при правке не работал вовсе. ``send_images=False`` — модель уже
    известна как текстовая: незачем каждый раз платить заведомо неудачным
    запросом с картинками.

    Теги ``<imageN>`` в сообщении остаются и без картинок: это адреса
    изображений для самой Qwen-Image, и переписанный промт обязан их
    сохранить. К сообщению добавляется только ``BLIND_NOTE``.

    ``annotation`` — правка в режиме «Аннотация»: пометки на исходнике
    объясняются переписывателю (``ANNOTATION_NOTE``).
    """
    system = _read_system_prompt(prompt_dir, _PROMPT_FILES[mode])
    if mode == MODE_T2I:
        system += T2I_NOTE
    user = build_user_message(prompt, len(references or []), tags)
    if annotation and mode == MODE_EDIT:
        user += ANNOTATION_NOTE
    # Референсы уходят в модель только в режиме редактирования: переписывателю
    # T2I смотреть не на что, а лишние изображения удлиняют запрос.
    images = references if mode == MODE_EDIT else None
    if images and send_images:
        try:
            return _ask(client, system, user, prompt, images)
        except LlmImagesRejected as error:
            LOGGER.warning("Language model rejected the images, rewriting from text only: %s", error)
    if not images:
        return _ask(client, system, user, prompt)
    # Изображения были, но переписыватель их не увидит — и узнаёт об этом.
    return replace(_ask(client, system, user + BLIND_NOTE, prompt), images_skipped=True)


def _ask(
    client: LlmClient,
    system: str,
    user: str,
    instruction: str,
    images: list[Image.Image] | None = None,
) -> BoostResult:
    """Один запрос и, если описание пришло пустым или не на том языке, один повтор.

    Повтор ровно один: сбой редкий, и второй ответ почти наверняка верный;
    если нет — лучше отдать его, чем задерживать генерацию дальше. Пустой
    второй ответ заменяется исходной инструкцией: генерация с пустым промтом
    рисовала бы что попало.
    """
    result = parse_response(client.complete(system, user, images=images))
    if not result.prompt:
        LOGGER.warning("Rewriter returned an empty prompt, retrying")
        result = parse_response(client.complete(system, user, images=images))
    elif off_language(instruction, result.prompt):
        LOGGER.warning("Rewritten prompt is not in English, retrying")
        result = parse_response(client.complete(system, user, images=images))
    if not result.prompt:
        LOGGER.warning("Rewriter returned an empty prompt again, keeping the original")
        return replace(result, prompt=instruction)
    return result


def describe(client: LlmClient, image: Image.Image, prompt_dir: Path) -> str:
    system = _read_system_prompt(prompt_dir, _DESCRIBE_FILE)
    return client.complete(system, "Describe this image.", images=[image]).strip()

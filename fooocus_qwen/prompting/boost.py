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
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”|「[^」]*」|『[^』]*』')

# Пометка к запросу правки, когда изображений переписыватель не видит.
# Системный промт правки целиком построен на чтении картинки; пометка говорит
# модели правду о том, что картинки нет. Опыт tools/experiments/
# boost_text_only.py (qwen3.8-27b, 80 ответов на вариант): выдумок и чужого
# языка не нашлось ни с пометкой, ни без — редкий сбой, увиденный вживую,
# она не лечит (это делает проверка языка ниже); зато ответ короче и быстрее,
# 4.5 с против 5.4. Пометка по-английски: так написан системный промт.
BLIND_NOTE = (
    "\n\nNote: the input image is NOT available to you — you cannot see it. "
    "Do not guess, describe or name anything the image might contain. Rewrite "
    "only the instruction itself: make the requested change concrete and say "
    "that everything else in the image stays exactly as it is, without listing "
    "what that is. Write the description in the language of the instruction."
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
            rewritten = _clean(payload.get("rewritten_prompt"))
            if rewritten:
                return BoostResult(
                    prompt=rewritten,
                    wh_ratio=_clean(payload.get("wh_ratio")),
                    ratio_follow=_clean(payload.get("ratio_follow")),
                    raw=text,
                )

    LOGGER.warning("Ответ переписывателя не содержит поля rewritten_prompt, беру текст как есть")
    return BoostResult(prompt=raw, raw=text)


def off_language(instruction: str, rewritten: str) -> bool:
    """Описание по-китайски при инструкции без иероглифов.

    Оба системных промта требуют: инструкция на китайском — описание на
    китайском, на любом другом языке — по-английски. Модель изредка
    нарушает правило и отвечает по-китайски на английскую инструкцию
    (увидено вживую на qwen3.8-27b). Текст в кавычках не считается: это
    надписи, которые модель изображения нарисует, и их язык решается
    отдельным правилом (B).
    """
    if _CJK.search(instruction):
        return False
    return bool(_CJK.search(_QUOTED.sub("", rewritten)))


def build_user_message(prompt: str, reference_count: int) -> str:
    """Собирает сообщение пользователя для переписывателя.

    При двух и более референсах спецификация Qwen требует адресовать их тегами
    ``<imageN>``; при единственном теги запрещены.
    """
    if reference_count < 2:
        return prompt
    tags = " ".join(f"<image{index}>" for index in range(1, reference_count + 1))
    return f"Input images: {tags}\n\nInstruction: {prompt}"


def _read_system_prompt(prompt_dir: Path, filename: str) -> str:
    path = prompt_dir / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise FileNotFoundError(f"Не найден системный промт {path}: {error}") from error


def boost(
    client: LlmClient,
    prompt: str,
    *,
    mode: str,
    prompt_dir: Path,
    references: list[Image.Image] | None = None,
    send_images: bool = True,
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
    """
    system = _read_system_prompt(prompt_dir, _PROMPT_FILES[mode])
    user = build_user_message(prompt, len(references or []))
    # Референсы уходят в модель только в режиме редактирования: переписывателю
    # T2I смотреть не на что, а лишние изображения удлиняют запрос.
    images = references if mode == MODE_EDIT else None
    if images and send_images:
        try:
            return _ask(client, system, user, prompt, images)
        except LlmImagesRejected as error:
            LOGGER.warning("Языковая модель не приняла изображения, переписываю по тексту: %s", error)
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
    """Один запрос и, если описание пришло не на том языке, один повтор.

    Повтор ровно один: сбой редкий, и второй ответ почти наверняка верный;
    если нет — лучше отдать его, чем задерживать генерацию дальше.
    """
    result = parse_response(client.complete(system, user, images=images))
    if off_language(instruction, result.prompt):
        LOGGER.warning("Переписанный промт пришёл не на языке инструкции, спрашиваю ещё раз")
        result = parse_response(client.complete(system, user, images=images))
    return result


def describe(client: LlmClient, image: Image.Image, prompt_dir: Path) -> str:
    system = _read_system_prompt(prompt_dir, _DESCRIBE_FILE)
    return client.complete(system, "Describe this image.", images=[image]).strip()

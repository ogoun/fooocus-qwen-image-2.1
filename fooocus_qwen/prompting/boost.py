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
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..llm import LlmClient

LOGGER = logging.getLogger(__name__)

MODE_T2I = "t2i"
MODE_EDIT = "edit"

_PROMPT_FILES = {
    MODE_T2I: "system_prompt_t2i.txt",
    MODE_EDIT: "system_prompt_edit.txt",
}
_DESCRIBE_FILE = "system_prompt_describe.txt"

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


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


def _clean(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def parse_response(text: str) -> BoostResult:
    raw = text.strip()
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
) -> BoostResult:
    system = _read_system_prompt(prompt_dir, _PROMPT_FILES[mode])
    user = build_user_message(prompt, len(references or []))
    # Референсы уходят в модель только в режиме редактирования: переписывателю
    # T2I смотреть не на что, а лишние изображения удлиняют запрос.
    images = references if mode == MODE_EDIT else None
    answer = client.complete(system, user, images=images)
    return parse_response(answer)


def describe(client: LlmClient, image: Image.Image, prompt_dir: Path) -> str:
    system = _read_system_prompt(prompt_dir, _DESCRIBE_FILE)
    return client.complete(system, "Describe this image.", images=[image]).strip()

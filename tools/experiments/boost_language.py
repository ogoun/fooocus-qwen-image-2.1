r"""Опыт: на каком языке AI буст переписывает русский промт.

Оба системных промта требуют описание по-английски для любой инструкции,
кроме китайской (t2i: «The description is always in English»; правка:
правило (A)). Вживую буст на вкладке генерации возвращал русский текст.

Сравниваем пути, которыми промт уходит к переписывателю:

  ``t2i``         — генерация без референсов;
  ``edit-images`` — правка/генерация с референсами, картинки видны модели;
  ``edit-blind``  — то же без картинок (текстовая модель) с пометкой
                    ``boost.BLIND_NOTE``, как она стоит в коде;
  ``edit-plain``  — без картинок и без пометки: контроль к ``edit-blind``,
                    отличает действие пометки от свойства самой модели.

Мера — доля ответов, где вне кавычек есть кириллица (текст в кавычках —
надпись на картинке, её язык решается отдельно).

Запуск (нужен настроенный llm_endpoint.txt):
    .venv\Scripts\python tools\experiments\boost_language.py --runs 4
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from fooocus_qwen import config
from fooocus_qwen.llm import LlmImagesRejected
from fooocus_qwen.logging_setup import use_utf8_console
from fooocus_qwen.prompting import boost
from fooocus_qwen.ui.state import Studio

INSTRUCTIONS = [
    "рыжая лиса в заснеженном берёзовом лесу на закате",
    "сделай небо закатным",
    "замени фон на морской берег",
    "кот в очках читает газету на кухне",
]

CYRILLIC = re.compile(r"[А-Яа-яЁё]")
QUOTED = re.compile(r'"[^"]*"|“[^”]*”|«[^»]*»')


def cyrillic_outside_quotes(text: str) -> bool:
    return bool(CYRILLIC.search(QUOTED.sub("", text)))


def sample_image() -> Image.Image:
    """Гладкий градиент: смысловой нагрузки нет, язык ответа от него не зависит."""
    image = Image.new("RGB", (512, 384))
    image.putdata([(x // 2, y * 255 // 384, 128) for y in range(384) for x in range(512)])
    return image


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=4, help="повторов на инструкцию и вариант")
    parser.add_argument("--note", default=None,
                        help="текст пометки для варианта edit-blind вместо boost.BLIND_NOTE")
    args = parser.parse_args()
    note = args.note if args.note is not None else boost.BLIND_NOTE

    studio = Studio(config.AppConfig())
    client = studio.llm_client()
    prompts = {
        name: (config.SYSTEM_PROMPT_DIR / f"system_prompt_{name}.txt").read_text(encoding="utf-8")
        for name in ("t2i", "edit")
    }
    image = sample_image()
    print(f"модель: {client.model or '(без имени)'}\n")

    variants = {
        "t2i": ("t2i", lambda text: text, None),
        "edit-images": ("edit", lambda text: text, [image]),
        "edit-blind": ("edit", lambda text: text + note, None),
        "edit-plain": ("edit", lambda text: text, None),
    }
    totals = {name: {"n": 0, "ru": 0, "seconds": 0.0} for name in variants}
    for instruction in INSTRUCTIONS:
        for name, (system, make, images) in variants.items():
            for _ in range(args.runs):
                started = time.time()
                try:
                    raw = client.complete(prompts[system], make(instruction), images=images)
                except LlmImagesRejected:
                    print(f"[{name}] модель не принимает изображения — вариант пропущен")
                    break
                answer = boost.parse_response(raw).prompt
                row = totals[name]
                row["n"] += 1
                row["seconds"] += time.time() - started
                russian = cyrillic_outside_quotes(answer)
                row["ru"] += russian
                print(f"[{name}] {instruction!r}: {'RU ' if russian else ''}\n    {answer[:140]!r}")

    print("\nвариант     | ответов | кириллица вне кавычек | среднее, с")
    for name, row in totals.items():
        n = row["n"]
        if n:
            print(f"{name:11} | {n:7} | {row['ru'] / n:21.0%} | {row['seconds'] / n:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

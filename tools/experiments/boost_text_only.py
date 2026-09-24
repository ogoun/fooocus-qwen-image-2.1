r"""Опыт: AI буст правки у языковой модели без зрения.

Текстовая модель не принимает изображения, и буст правки повторяет запрос
одним текстом. Системный промт правки при этом целиком построен вокруг
чтения картинки («прочти весь текст на изображении…»), и живой прогон на
qwen3.8-27b показал два сбоя такого запроса:

* описание приходит по-китайски при английской инструкции — вопреки правилу
  (A) самого системного промта;
* переписыватель выдумывает содержимое кадра («сохранить здания, людей,
  машины»), которого он не видел. Для правки по маске это прямой вред:
  модель изображения получает указания про несуществующие предметы.

Сравниваем два варианта сообщения пользователя:

  ``plain`` — как есть: инструкция без картинок;
  ``blind`` — та же инструкция плюс пометка, что изображений переписыватель
              не видит и описывать их не должен.

Меры: доля ответов не на языке инструкции (иероглифы при английской
инструкции) и доля ответов, называющих предметы, которых нет в инструкции
(выдуманное содержимое кадра). Словарь предметов — общий для сцен, на
которые модель чаще всего «догадывается».

Запуск (нужен настроенный llm_endpoint.txt):
    .venv\Scripts\python tools\experiments\boost_text_only.py --runs 4
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fooocus_qwen import config
from fooocus_qwen.logging_setup import use_utf8_console
from fooocus_qwen.prompting import boost
from fooocus_qwen.ui.state import Studio

INSTRUCTIONS = [
    "replace the sky with a sunset",
    "make the car red",
    "remove the person on the left",
    "add a small wooden boat on the water",
    "turn it into a watercolor painting",
]

# Предметы, которые переписыватель вставляет «на всякий случай», не видя
# кадра. Только те, что из инструкций опыта никак не следуют: облака и
# горизонт у заката или вода у лодки — законный вывод, а не выдумка.
# Сравнение целыми словами (с формой множественного числа): «ground» внутри
# «background» выдумкой не считается.
INVENTED = [
    "building", "people", "person", "vehicle", "car", "tree", "road", "street",
    "mountain", "house", "animal", "dog", "bird", "sign",
]
INVENTED_CJK = ["建筑", "人物", "车辆", "树", "山", "房屋", "道路"]
CJK = re.compile(r"[一-鿿]")


def invented(instruction: str, answer: str) -> list[str]:
    said = instruction.lower()
    text = answer.lower()
    found = [
        word for word in INVENTED
        if word not in said and re.search(rf"{word}s?", text)
    ]
    return found + [word for word in INVENTED_CJK if word in answer]


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=4, help="повторов на инструкцию и вариант")
    parser.add_argument("--instruction", default=None, help="только эта инструкция вместо набора")
    args = parser.parse_args()
    instructions = [args.instruction] if args.instruction else INSTRUCTIONS

    studio = Studio(config.AppConfig())
    client = studio.llm_client()
    system = (config.SYSTEM_PROMPT_DIR / "system_prompt_edit.txt").read_text(encoding="utf-8")
    print(f"модель: {client.model or '(без имени)'}\n")

    variants = {
        "plain": lambda text: boost.build_user_message(text, 1),
        "blind": lambda text: boost.build_user_message(text, 1) + boost.BLIND_NOTE,
    }
    totals = {name: {"n": 0, "cjk": 0, "invented": 0, "seconds": 0.0} for name in variants}
    for instruction in instructions:
        for name, make in variants.items():
            for _ in range(args.runs):
                started = time.time()
                answer = boost.parse_response(client.complete(system, make(instruction))).prompt
                row = totals[name]
                row["n"] += 1
                row["seconds"] += time.time() - started
                row["cjk"] += bool(CJK.search(answer))
                extra = invented(instruction, answer)
                row["invented"] += bool(extra)
                print(f"[{name}] {instruction!r}: {'CJK ' if CJK.search(answer) else ''}"
                      f"{('выдумано: ' + ', '.join(extra)) if extra else ''}\n    {answer[:150]!r}")

    print("\nвариант | ответов | не на языке инструкции | выдуманное содержимое | среднее, с")
    for name, row in totals.items():
        n = row["n"]
        print(f"{name:7} | {n:7} | {row['cjk'] / n:22.0%} | {row['invented'] / n:21.0%} | "
              f"{row['seconds'] / n:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

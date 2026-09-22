r"""Граф прохождения запроса с временем каждого узла.

Инструмент отвечает на один вопрос: куда уходит время от нажатия кнопки до
записи PNG. Узлы графа — не «все функции подряд», а те места, между
которыми время может распределиться по-разному: подготовка маски,
кодирование промта, перестановка весов, цикл денойзинга, декодирование,
склейка, запись файла.

Измеряется с синхронизацией CUDA на каждой границе. Без неё вызовы ядер
возвращаются немедленно, и всё время стекает в первый же узел, который
случайно дождался устройства: картина получается не просто неточной, а
качественно неверной.

Узлы вложены, и время родителя включает время детей. В сводке показаны оба
числа — полное и собственное, за вычетом детей. Тяжёлым считается узел с
большим **собственным** временем.

Запуск (из корня проекта):
    .venv\Scripts\python tools\profile_pipeline.py --scenario t2i
    .venv\Scripts\python tools\profile_pipeline.py --scenario mask --preset MiddleQuality
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fooocus_qwen.logging_setup import use_utf8_console

use_utf8_console()

import argparse
import json
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
from PIL import Image, ImageDraw

from fooocus_qwen import config, logging_setup

OUT = config.LOG_DIR / "profile"


class Recorder:
    """Стек вызовов с временем и числом заходов в каждый узел."""

    def __init__(self, sync: bool = True) -> None:
        self._sync = sync and torch.cuda.is_available()
        self._stack: list[str] = []
        self.total: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.children: dict[str, float] = defaultdict(float)
        self.parent: dict[str, str] = {}

    def _now(self) -> float:
        if self._sync:
            torch.cuda.synchronize()
        return time.perf_counter()

    @contextmanager
    def node(self, name: str):
        path = "/".join([*self._stack, name])
        if self._stack:
            self.parent.setdefault(path, "/".join(self._stack))
        started = self._now()
        self._stack.append(name)
        try:
            yield
        finally:
            self._stack.pop()
            spent = self._now() - started
            self.total[path] += spent
            self.calls[path] += 1
            if path in self.parent:
                self.children[self.parent[path]] += spent

    def wrap(self, owner, attribute: str, name: str) -> None:
        """Оборачивает метод объекта узлом графа."""
        original = getattr(owner, attribute, None)
        if original is None or getattr(original, "_profiler_original", None) is not None:
            return

        def measured(*args, **kwargs):
            with self.node(name):
                return original(*args, **kwargs)

        measured._profiler_original = original
        setattr(owner, attribute, measured)

    def report(self) -> list[dict]:
        rows = []
        for path, spent in self.total.items():
            own = spent - self.children.get(path, 0.0)
            rows.append({
                "node": path,
                "total_s": round(spent, 3),
                "own_s": round(own, 3),
                "calls": self.calls[path],
                "per_call_ms": round(1000 * spent / self.calls[path], 1),
            })
        return sorted(rows, key=lambda row: -row["own_s"])


def instrument(recorder: Recorder, pipe, engine) -> None:
    """Расставляет узлы по всему пути запроса.

    Список узлов и есть граф прохождения. Он составлен по коду, а не по
    догадкам: каждый пункт соответствует месту, где запрос переходит из
    одной подсистемы в другую.
    """
    from fooocus_qwen.engine import generator as gen
    from fooocus_qwen.imaging import masking, metadata

    recorder.wrap(engine, "_prepare", "подготовка/маска и вырезка")
    recorder.wrap(engine, "_finish", "склейка результата")
    recorder.wrap(gen, "apply_styles", "подготовка/стили")
    recorder.wrap(gen, "build_conditions", "подготовка/условные изображения")
    recorder.wrap(masking, "refine", "подготовка/refine маски")
    recorder.wrap(masking, "as_condition", "подготовка/маска в условное")
    recorder.wrap(masking, "blend", "склейка/blend")
    recorder.wrap(masking, "stitch", "склейка/stitch")
    recorder.wrap(masking, "paste_region", "склейка/paste_region")
    recorder.wrap(masking, "clipped_share", "склейка/доля обрезанного")
    recorder.wrap(metadata, "save_png", "запись PNG")

    cache = getattr(pipe, "_studio_cache", None)
    if cache is not None:
        recorder.wrap(cache, "key", "кодирование/ключ кэша (хеш картинок)")
    residency = getattr(pipe, "_studio_residency", None)
    if residency is not None:
        for attribute in ("_transformer", "_text_encoder"):
            staged = getattr(residency, attribute, None)
            if staged is not None:
                short = "трансформер" if "transformer" in attribute else "энкодер"
                recorder.wrap(staged, "to_host", f"перестановка/{short} на хост")
                recorder.wrap(staged, "to_device", f"перестановка/{short} на карту")

    recorder.wrap(pipe, "_get_qwen_prompt_embeds", "кодирование промта")
    recorder.wrap(pipe.transformer, "forward", "денойзинг/шаг трансформера")
    recorder.wrap(pipe.vae, "encode", "VAE/кодирование условных")
    recorder.wrap(pipe.vae, "decode", "VAE/декодирование кадра")
    recorder.wrap(pipe.image_processor, "preprocess", "подготовка/preprocess условных")
    recorder.wrap(pipe.image_processor, "resize", "подготовка/resize условных")
    recorder.wrap(pipe.image_processor, "postprocess", "постобработка в PIL")


def unwrap(pipe, engine) -> None:
    """Снимает обёртки, чтобы следующий прогон не мерил через два слоя."""
    from fooocus_qwen.engine import generator as gen
    from fooocus_qwen.imaging import masking, metadata

    owners = [engine, gen, masking, metadata, pipe, pipe.transformer, pipe.vae,
              pipe.image_processor]
    residency = getattr(pipe, "_studio_residency", None)
    if residency is not None:
        owners += [s for s in (getattr(residency, "_transformer", None),
                               getattr(residency, "_text_encoder", None)) if s is not None]
    cache = getattr(pipe, "_studio_cache", None)
    if cache is not None:
        owners.append(cache)

    for owner in owners:
        for attribute in list(vars(owner)) if not isinstance(owner, type) else dir(owner):
            try:
                value = getattr(owner, attribute)
            except Exception:  # noqa: BLE001 — свойства могут бросать
                continue
            original = getattr(value, "_profiler_original", None)
            if original is not None:
                setattr(owner, attribute, original)


def scenario_request(kind: str, preset_name: str, source_path: Path):
    from fooocus_qwen.engine import presets
    from fooocus_qwen.engine.generator import MASK_MASK, MASK_NONE, GenerationRequest
    from fooocus_qwen.imaging import aspect

    preset = presets.get(preset_name)
    prompt = "a wooden desk with a brass lamp and a cup of tea, warm afternoon light"

    if kind == "t2i":
        return GenerationRequest(prompt=prompt, preset=preset, seed=7)

    source = Image.open(source_path).convert("RGB")
    if kind == "edit":
        return GenerationRequest(
            prompt="make the background a quiet autumn park", preset=preset,
            aspect=aspect.FOLLOW_REFERENCE, source=source, mask_mode=MASK_NONE, seed=7,
        )
    if kind == "mask":
        width, height = source.size
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).ellipse(
            (width // 4, height // 4, width * 3 // 4, height * 3 // 4), fill=255
        )
        return GenerationRequest(
            prompt="a small red rose tucked into her hair", preset=preset,
            aspect=aspect.FOLLOW_REFERENCE, source=source, mask=mask,
            mask_mode=MASK_MASK, keep_outside=True, seed=7,
        )
    raise ValueError(f"неизвестный сценарий: {kind}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Граф прохождения со временем узлов")
    parser.add_argument("--scenario", choices=("t2i", "edit", "mask"), default="t2i")
    parser.add_argument("--preset", default="MiddleQuality")
    parser.add_argument("--steps", type=int, default=8,
                        help="меньше шагов: доля денойзинга линейна по ним, "
                             "а остальные узлы от числа шагов не зависят")
    parser.add_argument("--repeat", type=int, default=2,
                        help="первый прогон несёт разовую перестановку весов и прогрев ядер")
    parser.add_argument("--source", type=str, default=None)
    args = parser.parse_args()

    logging_setup.setup_logging(False)
    OUT.mkdir(parents=True, exist_ok=True)

    from fooocus_qwen.engine import loader, presets
    from fooocus_qwen.engine.generator import Generator
    from fooocus_qwen.engine.presets import QualityPreset
    from fooocus_qwen.imaging import metadata
    from fooocus_qwen.prompting.styles import load_styles
    from fooocus_qwen.storage import gallery

    started = time.perf_counter()
    pipe, residency, cache = loader.load(config.MODEL_DIR)
    load_seconds = time.perf_counter() - started
    engine = Generator(pipe, residency, cache, load_styles(config.STYLES_DIR))

    source_path = Path(args.source) if args.source else config.LOG_DIR / "smoke" / "source.png"
    request = scenario_request(args.scenario, args.preset, source_path)
    preset = presets.get(args.preset)
    request.preset = QualityPreset(preset.name, preset.output_resolution, args.steps)

    reports = []
    for attempt in range(args.repeat):
        recorder = Recorder()
        instrument(recorder, pipe, engine)
        request.seed = 100 + attempt

        with recorder.node("ВСЁ"):
            produced = engine.generate(request)
            if produced:
                with recorder.node("запись PNG"):
                    destination = gallery.next_path(OUT)
                    metadata.save_png(produced[0].image, destination, produced[0].parameters)

        unwrap(pipe, engine)
        reports.append(recorder.report())

    final = reports[-1]
    name = f"{args.scenario}-{args.preset}-{args.steps}steps"
    (OUT / f"{name}.json").write_text(
        json.dumps({"warmup": reports[0], "steady": final,
                    "model_load_s": round(load_seconds, 1)},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    whole = next((row["total_s"] for row in final if row["node"] == "ВСЁ"), 0.0) or 1.0
    print(f"\n=== {name}: установившийся прогон, всего {whole:.1f} с ===")
    print(f"{'узел':>42} | {'всего, с':>8} | {'своё, с':>7} | {'раз':>4} | "
          f"{'на заход, мс':>12} | доля")
    for row in final:
        if row["node"] == "ВСЁ":
            continue
        print(f"{row['node']:>42} | {row['total_s']:>8} | {row['own_s']:>7} | "
              f"{row['calls']:>4} | {row['per_call_ms']:>12} | "
              f"{100 * row['own_s'] / whole:5.1f}%")
    print(f"\nзагрузка модели (разово): {load_seconds:.1f} с")
    print(f"json: {OUT / (name + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

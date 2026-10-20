"""Замеры видео: время и пик памяти по режимам и разрешениям H3 (FreeVideo).

Каждый случай идёт той же дорогой, что и кнопка вкладки «Видео» —
``Studio.run_video``: запрос, драйвер, оркестратор FreeVideo, ролик в
галерее. Параллельно раз в полсекунды пишутся занятая видеопамять всей карты
(NVML) и сумма RSS процессов FreeVideo (драйвер, CLI, энкодер, рабочий
процесс модели). Под Windows (WDDM) NVML не отдаёт память по процессу,
поэтому видеопамять — разность с фоном до запуска: окна, браузеры и прочие
программы карты в неё не входят. Отдельно читаются собственные отчёты
движка (``*.request.json``) — его план размещения и пики.

Случаи ``budget8`` — проверка заявки FreeVideo «от 8 ГБ VRAM и 16 ГБ RAM»:
тот же запрос с номинальным бюджетом ``--vram-gib 8 --ram-gib 16``, по
которому движок планирует размещение, как на маленькой машине.

Запуск:
    .venv\\Scripts\\python tools\\video_benchmark.py
    .venv\\Scripts\\python tools\\video_benchmark.py --only t2v_480 budget8_480

Результаты — JSON и таблица Markdown в ``tmp/video_bench/``; ролики — в
галерее, как обычные.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fooocus_qwen.logging_setup import use_utf8_console

use_utf8_console()

import argparse
import json
import threading
import time
from datetime import datetime

import numpy as np
import psutil
from PIL import Image

from fooocus_qwen import config, logging_setup
from fooocus_qwen.video import geometry, prompt_h3
from fooocus_qwen.video.request import Reference, VideoRequest, read_parameters

PROMPT_T2V = (
    "integrated_multimodal_description: [Shot 1] A warm, softly lit kitchen in the morning. An elderly man in a "
    "grey knitted cardigan pours steaming tea from a white porcelain teapot into a glass cup on a wooden table; "
    "steam curls up through a beam of sunlight from the window. The camera slowly pushes in toward the cup. "
    "He looks up and smiles gently.\n\n"
    "overall_soundscape: The quiet gurgle of tea being poured, a soft clink of porcelain on glass, distant birds "
    "outside the window.\n\n"
    "non_diegetic_music: Gentle solo acoustic guitar, slow tempo, soft dynamics."
)
PROMPT_BUDGET = (
    "integrated_multimodal_description: [Shot 1] A red fox trots across a snowy forest clearing at dawn, its "
    "breath visible in the cold air; pale pink light filters between dark pine trunks. The camera tracks it "
    "slowly from the side at a medium distance.\n\n"
    "overall_soundscape: Soft crunch of paws on fresh snow, a faint wind in the pine branches.\n\n"
    "non_diegetic_music: N/A"
)
PROMPT_BUDGET_LONG = (
    "integrated_multimodal_description: [Shot 1] A small wooden sailboat glides across a calm turquoise bay "
    "under a bright afternoon sun; a woman in a straw hat adjusts the sail rope. [Shot 2] At 00:05.000, the "
    "camera cuts to a close view of the bow cutting through clear water, sunlight sparkling on the ripples.\n\n"
    "overall_soundscape: Gentle lapping of water against the hull, the flutter of the sail, distant gulls.\n\n"
    "non_diegetic_music: Light acoustic ukulele, relaxed tempo, soft dynamics."
)
PROMPT_I2V = (
    "The scene in the picture comes alive: soft wind moves through it, light shifts slowly across the surfaces, "
    "and the camera drifts gently forward. Ambient sound of the place, no music."
)
PROMPT_FLF = (
    "A smooth, continuous transformation from the first picture to the last one, with a slow camera move; "
    "everything changes gradually and naturally. Soft ambient sound."
)
PROMPT_REF = (
    "The subject from <Picture 1> walks slowly through a sunlit park, looking around with curiosity; the camera "
    "follows at a medium distance. Birdsong and footsteps on gravel."
)


def gradient(size: tuple[int, int], flip: bool = False) -> Image.Image:
    width, height = size
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    red = np.tile(x[::-1] if flip else x, (height, 1))
    green = np.tile(y[:, None], (1, width))
    return Image.fromarray(np.stack([red, green, np.full((height, width), 140, np.uint8)], axis=2))


INPUTS = config.PROJECT_ROOT / "tmp" / "video_bench" / "inputs"


def sample_images(count: int = 2) -> list[Path]:
    """Входные кадры замеров: ``tmp/video_bench/inputs/first.png`` и ``last.png``.

    Из галереи их не берём: там личные снимки человека (люди на фото — не
    материал для тестовых роликов) и, до починки conftest, мусор тестов.
    Положите сюда нейтральные кадры — например, кадры уже снятых роликов
    (``assemble.first_frame``/``last_frame``); без них — градиенты.
    """
    names = ("first.png", "last.png")[:count]
    return [INPUTS / name for name in names if (INPUTS / name).is_file()]


def cases() -> dict[str, dict]:
    pictures = sample_images(2)
    first = Image.open(pictures[0]).convert("RGB") if pictures else gradient((1024, 576))
    last = Image.open(pictures[1]).convert("RGB") if len(pictures) > 1 else gradient((1024, 576), flip=True)
    reference = str(pictures[0]) if pictures else None
    table = {
        "t2v_480": dict(prompt=PROMPT_T2V, mode=prompt_h3.MODE_T2V, resolution="480p", seconds=5.0),
        "i2v_480": dict(prompt=PROMPT_I2V, mode=prompt_h3.MODE_I2V, resolution="480p", seconds=5.0,
                        aspect=geometry.FOLLOW_IMAGE, first=first),
        "flf2v_480": dict(prompt=PROMPT_FLF, mode=prompt_h3.MODE_FLF2V, resolution="480p", seconds=5.0,
                          first=first, last=last),
        "t2v_768_10s": dict(prompt=PROMPT_T2V, mode=prompt_h3.MODE_T2V, resolution="768p", seconds=10.0),
        # Повтор с другим сидом: первый прогон новой геометрии несёт
        # автонастройку ядер Triton, а с тем же сидом ответил бы кэш результатов.
        "t2v_768_10s_repeat": dict(prompt=PROMPT_T2V, mode=prompt_h3.MODE_T2V, resolution="768p", seconds=10.0,
                                   seed=7),
        # Промт бюджетных случаев — свой: с тем же промтом движок взял бы
        # готовое кодирование из кэша, и энкодер под бюджетом не проверялся бы.
        "budget8_480": dict(prompt=PROMPT_BUDGET, mode=prompt_h3.MODE_T2V, resolution="480p", seconds=5.0,
                            vram_gib=8.0, ram_gib=16.0),
        "budget8_768_10s": dict(prompt=PROMPT_BUDGET_LONG, mode=prompt_h3.MODE_T2V, resolution="768p",
                                seconds=10.0, vram_gib=8.0, ram_gib=16.0),
        "draft_480": dict(prompt=PROMPT_T2V, mode=prompt_h3.MODE_T2V, resolution="480p", seconds=5.0,
                          quality="draft"),
    }
    if reference:
        table["ref_480"] = dict(prompt=PROMPT_REF, mode=prompt_h3.MODE_REF, resolution="480p", seconds=5.0,
                                references=(Reference("image", reference),))
    return table


class Sampler:
    """Пики видеопамяти карты и RSS процессов FreeVideo, раз в полсекунды."""

    def __init__(self, root: Path) -> None:
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._root = str(root).lower()
        self.baseline = self.vram()
        self.peak_vram = self.baseline
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def vram(self) -> int:
        return self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used

    def rss(self) -> int:
        total = 0
        for process in psutil.process_iter(["cmdline", "memory_info"]):
            try:
                line = " ".join(process.info["cmdline"] or []).lower()
                if self._root in line and process.info["memory_info"] is not None:
                    total += process.info["memory_info"].rss
            except (psutil.Error, TypeError):
                continue
        return total

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_vram = max(self.peak_vram, self.vram())
            self.peak_rss = max(self.peak_rss, self.rss())
            self._stop.wait(0.5)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()


def engine_report(video: Path) -> dict:
    """Что движок сам записал о запросе: план размещения и пики."""
    parameters = read_parameters(video)
    output = parameters.get("engine_output")
    if not output:
        return {}
    report_path = (config.VIDEO_RUNS_DIR / output).with_suffix(".request.json")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    def value(*path):
        item = report
        for key in path:
            item = item.get(key) if isinstance(item, dict) else None
        return item

    def gib(number):
        return round(number / 2**30, 2) if isinstance(number, (int, float)) else None

    encoding = report.get("encoding") or {}
    # Кодирование из кэша несёт метрики того прогона, что кодировал: в этом
    # запросе энкодер не работал вовсе.
    encoded_here = not encoding.get("cache_source_metrics")
    return {
        "report": str(report_path),
        "engine_gpu_budget_gib": gib(value("profile", "policy", "gpu_budget_bytes")),
        "engine_ram_budget_gib": gib(value("profile", "policy", "ram_budget_bytes")),
        "resident_blocks": value("profile", "policy", "engine", "resident_blocks"),
        "low_memory_mode": report.get("low_memory_mode"),
        "dit_torch_peak_reserved_gib": gib(value("video", "torch_peak_reserved_bytes")),
        "decode_torch_peak_reserved_gib": gib(value("video", "final_stage_peak_reserved_bytes")),
        "encoder_ran": encoded_here,
        "encoder_torch_peak_reserved_gib": gib(encoding.get("torch_peak_reserved_bytes")) if encoded_here else None,
        "private_working_set_peak_gib": gib(value("resources", "ram", "process_tree_peak_private_working_set_bytes")),
        "private_commit_peak_gib": gib(value("resources", "ram", "process_tree_peak_private_commit_bytes")),
        "whole_gpu_peak_gib": gib(value("resources", "gpu", "gpu_peak_bytes")),
        "request_seconds": report.get("request_seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Video timing and memory by mode and resolution")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    logging_setup.setup_logging(False)
    config.ensure_directories()

    from fooocus_qwen.ui.state import Studio

    studio = Studio(config.AppConfig(preload=False))
    if not studio.video_status().ready:
        print(f"Video engine is not ready: {studio.video_status().reason}")
        return 1
    table = cases()
    chosen = args.only or list(table)
    out_dir = config.PROJECT_ROOT / "tmp" / "video_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results = []
    try:
        for name in chosen:
            values = dict(table[name])
            seed = values.pop("seed", 20261019)
            request = VideoRequest(seed=seed, **values)
            print(f"\n== {name}: {request.mode} {request.resolution} {request.seconds} s "
                  f"quality={request.quality} budget={request.vram_gib}/{request.ram_gib}", flush=True)
            last_label = [""]

            def on_event(event, _last=last_label):
                from fooocus_qwen.video.engine import progress_view

                view = progress_view(event)
                if view and view.label != _last[0]:
                    _last[0] = view.label
                    print(f"   {view.label}", flush=True)

            # Модель предыдущего случая отпускается: фон меряется по пустой карте.
            studio.release_video()
            time.sleep(3)
            with Sampler(config.FREEVIDEO_DIR) as sampler:
                started = time.perf_counter()
                path, parameters, failure = studio.run_video(request, "en", on_event)
                seconds = time.perf_counter() - started
            row = {
                "case": name, "mode": request.mode, "resolution": request.resolution,
                "width": parameters.get("width"), "height": parameters.get("height"),
                "frames": parameters.get("frames"), "video_seconds": parameters.get("seconds"),
                "quality": request.quality, "vram_gib_budget": request.vram_gib, "ram_gib_budget": request.ram_gib,
                "wall_seconds": round(seconds, 1),
                "vram_peak_gib": round((sampler.peak_vram - sampler.baseline) / 2**30, 2),
                "vram_baseline_gib": round(sampler.baseline / 2**30, 2),
                "rss_peak_gib": round(sampler.peak_rss / 2**30, 2),
                "failure": failure, "video": str(path) if path else None,
            }
            if path:
                row.update(engine_report(Path(path)))
            print(json.dumps(row, ensure_ascii=False), flush=True)
            results.append(row)
            (out_dir / f"{stamp}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        studio.close()

    lines = [
        "| Случай | Кадр | Кадров | Время, с | Бюджет | Энкодер, ГиБ | DiT, ГиБ | VAE, ГиБ "
        "| Частная RAM, ГиБ | Вся карта, ГиБ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        budget = f"{row['vram_gib_budget']:g}/{row['ram_gib_budget']:g}" if row["vram_gib_budget"] else "авто"
        encoder = row.get("encoder_torch_peak_reserved_gib") if row.get("encoder_ran") else "кэш"
        lines.append(
            f"| {row['case']} | {row['width']}×{row['height']} | {row['frames']} | {row['wall_seconds']} | {budget} "
            f"| {encoder} | {row.get('dit_torch_peak_reserved_gib')} | {row.get('decode_torch_peak_reserved_gib')} "
            f"| {row.get('private_working_set_peak_gib')} | {row.get('whole_gpu_peak_gib')} |"
            + (f" сбой: {row['failure']}" if row["failure"] else "")
        )
    (out_dir / f"{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return sum(1 for row in results if row["failure"])


if __name__ == "__main__":
    raise SystemExit(main())

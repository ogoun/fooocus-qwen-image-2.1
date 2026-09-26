r"""Опыт: рисует ли Qwen-Image 2.1 человека по скелету позы из референса.

Нужно для плитки «Добавить позу»: поза, распознанная на фотографии, получает
плитку в стиле библиотеки openposes.com (иллюстрация, одна и та же героиня).
Сравниваются две подачи:

  ``style+pose`` — два референса: плитка библиотеки (образ и стиль) и скелет;
  ``pose-only``  — один референс-скелет, образ и стиль описаны текстом.

Мера соблюдения позы не зависит от генератора: DWPose распознаёт позу на
результате, и она сравнивается с заданным скелетом после приведения обоих к
общей рамке (центр и размер). Ошибка — средняя по видимым точкам, в процентах
размера рамки; ``зеркально`` — та же мера с переставленными левой и правой
сторонами (перепутанные стороны — частый сбой, его стоит видеть отдельно).
Стиль и образ оцениваются глазами по листу ``tmp/pose_tile/sheet.png``.

Позы — отложенные из библиотеки: для них есть и скелет, и эталонная плитка.

Запуск (нужны веса модели, DWPose и плитки библиотеки в tmp/openposes):
    .venv\Scripts\python tools\experiments\pose_tile.py --preset Turbo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from fooocus_qwen import config  # noqa: E402
from fooocus_qwen.engine import presets  # noqa: E402
from fooocus_qwen.engine.generator import GenerationRequest  # noqa: E402
from fooocus_qwen.logging_setup import use_utf8_console  # noqa: E402
from fooocus_qwen.poses import detect, skeleton  # noqa: E402

WORK = ROOT / "tmp" / "pose_tile"
TILES = ROOT / "tmp" / "openposes"
POSES = ROOT / "tmp" / "poses_x"
STYLE_TILE = "standing_01"
HELD_OUT = ["jumping_03", "sitting_05", "laying_03", "dance_02"]

STYLE_POSE = (
    "Full-body digital illustration of the same woman as in <image1>: the same face, hair, "
    "outfit and painterly style, pastel flowers swirling around her, light grey background. "
    "She is posed exactly like the pose skeleton in <image2>: the colored lines mark her head, "
    "shoulders, arms, hips and legs. Do not draw the skeleton lines."
)
POSE_ONLY = (
    "Full-body digital illustration of a young woman with long wavy auburn hair, a dark red long "
    "coat over a white t-shirt, dark blue jeans and white sneakers, painterly style, pastel flowers "
    "swirling around her, light grey background. She is posed exactly like the pose skeleton in the "
    "reference image: the colored lines mark her head, shoulders, arms, hips and legs. Do not draw "
    "the skeleton lines."
)
# Стиль каталога openposes.com словами: живопись крупным мазком, фон —
# абстрактные разводы краски, одежда разная, героиня одна.
PAINTERLY = (
    "Full-body digital painting of Emma Watson, a young woman with shoulder-length wavy light "
    "brown hair, in a casual outfit, loose expressive painterly brushstrokes, soft cinematic light, "
    "the background is an abstract swirl of paint splashes in teal, blue and violet on a pale grey "
    "canvas. She is posed exactly like the pose skeleton in the reference image: the colored lines "
    "mark her head, shoulders, arms, hips and legs. Do not draw the skeleton lines."
)
PAINTERLY_ANON = PAINTERLY.replace("Emma Watson, a young woman", "a young woman")

VARIANTS = {
    "style+pose": (STYLE_POSE, True),
    "pose-only": (POSE_ONLY, False),
    "painterly": (PAINTERLY, False),
    "painterly-anon": (PAINTERLY_ANON, False),
}

# Перестановка левых и правых точек BODY_18.
_MIRROR = {2: 5, 3: 6, 4: 7, 8: 11, 9: 12, 10: 13, 14: 15, 16: 17}
_MIRROR.update({v: k for k, v in _MIRROR.items()})


def normalized(points: np.ndarray, visible: np.ndarray) -> np.ndarray:
    seen = points[visible]
    low, high = seen.min(0), seen.max(0)
    return (points - (low + high) / 2) / max(float((high - low).max()), 1.0)


def pose_error(target: skeleton.Pose, result: skeleton.Pose) -> tuple[float, float]:
    a = np.array([p[:2] for p in target.points])
    b = np.array([p[:2] for p in result.points])
    va = np.array([p[2] > 0 for p in target.points])
    vb = np.array([p[2] > 0 for p in result.points])
    both = va & vb
    if both.sum() < 4:
        return float("nan"), float("nan")
    na, nb = normalized(a, va), normalized(b, vb)
    direct = float(np.linalg.norm(na[both] - nb[both], axis=1).mean() * 100)
    order = [_MIRROR.get(i, i) for i in range(skeleton.POINTS)]
    nm, vm = nb[order], vb[order]
    both_m = va & vm
    mirrored = float(np.linalg.norm(na[both_m] - nm[both_m], axis=1).mean() * 100)
    return direct, mirrored


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", default="Turbo")
    parser.add_argument("--seeds", type=int, nargs="*", default=[1])
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    args = parser.parse_args()

    from fooocus_qwen.ui.state import Studio

    WORK.mkdir(parents=True, exist_ok=True)
    studio = Studio(config.AppConfig())
    preset = presets.get(args.preset)
    missing = studio.weights_for(preset, "ru")
    if missing:
        print(missing)
        return 1
    detector = detect.PoseDetector(config.DWPOSE_DIR)
    style = Image.open(TILES / f"{STYLE_TILE}.jpg").convert("RGB")

    rows = []
    for name in HELD_OUT:
        target = skeleton.load(POSES / f"{name}.json")
        bones = skeleton.render(target)
        row = [Image.open(TILES / f"{name}.jpg").convert("RGB"), bones]
        for variant in args.variants:
            prompt, with_style = VARIANTS[variant]
            refs = (style, bones) if with_style else (bones,)
            for seed in args.seeds:
                started = time.time()
                produced, failure = studio.run_generation(GenerationRequest(
                    prompt=prompt, prompt_original=prompt, preset=preset, references=refs,
                    aspect="1:1", seed=seed,
                ), "ru")
                if failure:
                    print(failure)
                    return 1
                image = produced[0].image
                image.save(WORK / f"{name}_{variant}_{seed}.png")
                try:
                    found = detect.to_pose(detector.detect(image))
                    direct, mirrored = pose_error(target, found)
                except detect.NoPersonFound:
                    direct = mirrored = float("nan")
                print(f"{name:11} {variant:10} seed {seed}: ошибка позы {direct:5.1f} % "
                      f"(зеркально {mirrored:5.1f} %), {time.time() - started:.0f} с", flush=True)
                row.append(image)
        rows.append(row)

    side = 256
    sheet = Image.new("RGB", (side * max(len(r) for r in rows), side * len(rows)), "white")
    for y, row in enumerate(rows):
        for x, image in enumerate(row):
            sheet.paste(image.resize((side, side)), (x * side, y * side))
    sheet.save(WORK / "sheet.png")
    print(f"лист: {WORK / 'sheet.png'} (столбцы: эталон, скелет, затем варианты по порядку)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

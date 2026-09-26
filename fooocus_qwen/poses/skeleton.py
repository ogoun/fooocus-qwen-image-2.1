"""Скелет позы в формате OpenPose: чтение, запись и отрисовка.

Формат — «тело из восемнадцати точек» (BODY_18, он же COCO-18 с шеей), в
котором хранит позы openposes.com и который понимают все ControlNet-подобные
модели: JSON ``[{"people": [{"pose_keypoints_2d": [x, y, c, …]}],
"canvas_width": W, "canvas_height": H}]``. Точка с нулевой уверенностью —
«не видна», её не рисуют.

Цвета и порядок костей — эталон ControlNet (``annotator/openpose/util.py``),
а начертание — как у openposes.com: кость — полупрозрачная полоса
постоянной ширины (у ControlNet — сужающийся эллипс), точки — поверх костей.
Совпадение с картинками openposes.com проверяется тестом по файлам их же
архива — картинка и JSON там лежат парой.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

POINTS = 18

# Порядок точек BODY_18.
NOSE, NECK = 0, 1
R_SHOULDER, R_ELBOW, R_WRIST = 2, 3, 4
L_SHOULDER, L_ELBOW, L_WRIST = 5, 6, 7
R_HIP, R_KNEE, R_ANKLE = 8, 9, 10
L_HIP, L_KNEE, L_ANKLE = 11, 12, 13
R_EYE, L_EYE, R_EAR, L_EAR = 14, 15, 16, 17

# Кости в порядке ControlNet: порядок задаёт цвет, поэтому он неприкосновенен.
LIMBS: tuple[tuple[int, int], ...] = (
    (NECK, R_SHOULDER), (NECK, L_SHOULDER),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (NECK, R_HIP), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
    (NECK, L_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (NECK, NOSE),
    (NOSE, R_EYE), (R_EYE, R_EAR),
    (NOSE, L_EYE), (L_EYE, L_EAR),
)

COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 85), (255, 0, 85),
)

# Холст библиотеки openposes.com — квадрат 768; толщины заданы для него и
# масштабируются вместе с холстом.
BASE_CANVAS = 768
LIMB_WIDTH = 8
POINT_RADIUS = 6
LIMB_ALPHA = 0.6


@dataclass(frozen=True)
class Pose:
    """Поза одного человека: 18 точек ``(x, y, уверенность)`` на холсте."""

    points: tuple[tuple[float, float, float], ...]
    width: int
    height: int

    def visible(self, index: int) -> bool:
        return self.points[index][2] > 0

    def to_json(self) -> str:
        flat = [value for point in self.points for value in point]
        return json.dumps([{
            "people": [{"pose_keypoints_2d": flat}],
            "canvas_width": self.width,
            "canvas_height": self.height,
        }])


def load(path: Path) -> Pose:
    return parse(Path(path).read_text(encoding="utf-8"))


def parse(text: str) -> Pose:
    """Разбирает JSON OpenPose; берёт первого человека на холсте."""
    data = json.loads(text)
    if isinstance(data, list):
        data = data[0]
    people = data.get("people") or []
    if not people:
        raise ValueError("в позе нет ни одного человека")
    flat = people[0]["pose_keypoints_2d"]
    if len(flat) < POINTS * 3:
        raise ValueError(f"в позе {len(flat) // 3} точек вместо {POINTS}")
    points = tuple(
        (float(flat[3 * i]), float(flat[3 * i + 1]), float(flat[3 * i + 2])) for i in range(POINTS)
    )
    return Pose(points, int(data.get("canvas_width", BASE_CANVAS)), int(data.get("canvas_height", BASE_CANVAS)))


def render(pose: Pose) -> Image.Image:
    """Рисует скелет на чёрном холсте размера позы — так, как рисует openposes.com."""
    canvas = np.zeros((pose.height, pose.width, 3), dtype=np.uint8)
    scale = min(pose.width, pose.height) / BASE_CANVAS
    radius = max(1, round(POINT_RADIUS * scale))
    thickness = max(1, round(LIMB_WIDTH * scale))

    for (start, end), color in zip(LIMBS, COLORS):
        if not (pose.visible(start) and pose.visible(end)):
            continue
        (x1, y1, _), (x2, y2, _) = pose.points[start], pose.points[end]
        layer = canvas.copy()
        cv2.line(layer, (round(x1), round(y1)), (round(x2), round(y2)), color, thickness, cv2.LINE_AA)
        canvas = cv2.addWeighted(canvas, 1 - LIMB_ALPHA, layer, LIMB_ALPHA, 0)

    for index in range(POINTS):
        if pose.visible(index):
            x, y, _ = pose.points[index]
            cv2.circle(canvas, (round(x), round(y)), radius, COLORS[index], -1, cv2.LINE_AA)

    return Image.fromarray(canvas, "RGB")

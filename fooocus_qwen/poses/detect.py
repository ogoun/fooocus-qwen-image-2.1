"""Распознавание позы на фотографии: DWPose через onnxruntime.

Выбор детектора (разбор — в ``docs/research/2026-09-26-pozy.md``):

* **CMU OpenPose** — первоисточник формата, но это C++/Caffe: под Windows
  сборка с CUDA, готовые сборки 2020 года под CUDA 10, лицензия
  некоммерческая. Для оболочки на Python не годится.
* **ControlNet OpenPose** (``lllyasviel/Annotators``) — тот же OpenPose,
  перенесённый в PyTorch; заметно ошибается на сложных позах.
* **DWPose** (``yzd-v/DWPose``, Apache-2.0) — дистиллированный RTMPose,
  точнее обоих на COCO; им распознают позы предобработчики ControlNet в
  A1111 и ComfyUI. Две ONNX-модели: YOLOX-L ищет людей, DW-LL (384×288)
  ставит 133 точки, из которых телу нужны первые 17 (COCO).

Выбран DWPose. Работает на процессоре: одна фотография — доли секунды, а
видеопамять целиком остаётся Qwen-Image. Библиотека ``rtmlib`` делает то
же, но тянет ``opencv-python`` и ``opencv-contrib-python``, которые
перезаписывают уже стоящий ``opencv-python-headless`` — поэтому здесь своя
тонкая обвязка по их же алгоритму (``rtmlib``, ``controlnet_aux.dwpose``).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import skeleton

LOGGER = logging.getLogger(__name__)

REPO = "yzd-v/DWPose"
DETECTOR_FILE = "yolox_l.onnx"
POSE_FILE = "dw-ll_ucoco_384.onnx"
FILES = (DETECTOR_FILE, POSE_FILE)

DETECTOR_INPUT = (640, 640)          # высота, ширина
POSE_INPUT = (288, 384)              # ширина, высота
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
SIMCC_SPLIT = 2.0
BOX_PADDING = 1.25

PERSON_SCORE = 0.3
NMS_IOU = 0.45
# Точка считается видимой от этой уверенности — порог DWPose в ControlNet.
POINT_SCORE = 0.3

# COCO-17 → BODY_18 (шея достраивается серединой плеч).
_COCO_TO_BODY18 = {
    0: skeleton.NOSE, 1: skeleton.L_EYE, 2: skeleton.R_EYE, 3: skeleton.L_EAR, 4: skeleton.R_EAR,
    5: skeleton.L_SHOULDER, 6: skeleton.R_SHOULDER, 7: skeleton.L_ELBOW, 8: skeleton.R_ELBOW,
    9: skeleton.L_WRIST, 10: skeleton.R_WRIST, 11: skeleton.L_HIP, 12: skeleton.R_HIP,
    13: skeleton.L_KNEE, 14: skeleton.R_KNEE, 15: skeleton.L_ANKLE, 16: skeleton.R_ANKLE,
}


class NoPersonFound(ValueError):
    """На фотографии не нашлось человека, чью позу можно взять."""


@dataclass(frozen=True)
class Detection:
    """Поза с координатами в пикселях исходной фотографии."""

    points: np.ndarray        # (17, 2)
    scores: np.ndarray        # (17,)
    box: np.ndarray           # x1, y1, x2, y2


class PoseDetector:
    """Две сессии onnxruntime; создаются при первом вызове и живут до конца процесса."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._sessions: tuple | None = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._sessions is None:
                import onnxruntime

                from ..engine import fetch

                # Веса качает установка (--fetch-model); если их всё же нет —
                # 350 МБ при первом распознавании, как адаптер Turbo при
                # первом выборе пресета; докачивается только недостающее.
                fetch.ensure_files(self.model_dir, REPO, FILES)

                options = onnxruntime.SessionOptions()
                options.log_severity_level = 3
                providers = ["CPUExecutionProvider"]
                self._sessions = tuple(
                    onnxruntime.InferenceSession(str(self.model_dir / name), options, providers=providers)
                    for name in FILES
                )
            return self._sessions

    def detect(self, image: Image.Image) -> Detection:
        """Поза главного человека кадра — самого крупного из найденных."""
        detector, estimator = self._load()
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        boxes = find_people(detector, rgb)
        if len(boxes) == 0:
            raise NoPersonFound("на фотографии не найден человек")
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        box = boxes[int(np.argmax(areas))]
        points, scores = estimate(estimator, rgb, box)
        return Detection(points, scores, box)


def find_people(session, rgb: np.ndarray) -> np.ndarray:
    """Рамки людей ``(N, 4)`` в пикселях кадра — YOLOX, как в DWPose."""
    height, width = DETECTOR_INPUT
    ratio = min(height / rgb.shape[0], width / rgb.shape[1])
    resized = cv2.resize(
        rgb[:, :, ::-1], (int(rgb.shape[1] * ratio), int(rgb.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR
    )
    padded = np.full((height, width, 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
    output = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
    output = _decode_yolox(output, DETECTOR_INPUT)

    centers, sizes = output[:, :2], output[:, 2:4]
    boxes = np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1) / ratio
    scores = output[:, 4] * output[:, 5]          # класс 0 — человек
    keep = scores > PERSON_SCORE
    boxes, scores = boxes[keep], scores[keep]
    if len(boxes) == 0:
        return boxes
    chosen = cv2.dnn.NMSBoxes(
        [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes],
        scores.tolist(), PERSON_SCORE, NMS_IOU,
    )
    return boxes[np.array(chosen).reshape(-1)]


def _decode_yolox(output: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    grids, strides = [], []
    for stride in (8, 16, 32):
        rows, cols = size[0] // stride, size[1] // stride
        xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
        grids.append(np.stack((xs, ys), 2).reshape(-1, 2))
        strides.append(np.full((rows * cols, 1), stride))
    grid, stride = np.concatenate(grids), np.concatenate(strides)
    output = output.copy()
    output[:, :2] = (output[:, :2] + grid) * stride
    output[:, 2:4] = np.exp(output[:, 2:4]) * stride
    return output


def estimate(session, rgb: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """17 точек тела в пикселях кадра и их уверенность — RTMPose/SimCC."""
    width, height = POSE_INPUT
    center = (box[:2] + box[2:]) / 2
    scale = (box[2:] - box[:2]) * BOX_PADDING
    aspect = width / height
    if scale[0] > scale[1] * aspect:
        scale = np.array([scale[0], scale[0] / aspect])
    else:
        scale = np.array([scale[1] * aspect, scale[1]])

    matrix = _warp_matrix(center, scale, (width, height))
    crop = cv2.warpAffine(rgb, matrix, (width, height), flags=cv2.INTER_LINEAR)
    blob = ((crop.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
    simcc_x, simcc_y = session.run(None, {session.get_inputs()[0].name: blob})

    x_index, y_index = simcc_x[0].argmax(1), simcc_y[0].argmax(1)
    scores = 0.5 * (simcc_x[0].max(1) + simcc_y[0].max(1))
    locations = np.stack([x_index, y_index], 1).astype(np.float32) / SIMCC_SPLIT
    points = locations / np.array([width, height]) * scale + center - scale / 2
    return points[:17], scores[:17]


def _warp_matrix(center: np.ndarray, scale: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Аффинное преобразование рамки ``center ± scale/2`` в кадр ``size``."""
    source = np.array([
        center,
        center + np.array([0.0, -scale[0] * 0.5]),
        center + np.array([-scale[0] * 0.5, -scale[0] * 0.5]),
    ], dtype=np.float32)
    width, height = size
    target = np.array([
        [width * 0.5, height * 0.5],
        [width * 0.5, height * 0.5 - width * 0.5],
        [0.0, height * 0.5 - width * 0.5],
    ], dtype=np.float32)
    return cv2.getAffineTransform(source, target)


def to_pose(detection: Detection, canvas: int = skeleton.BASE_CANVAS, margin: float = 0.08) -> skeleton.Pose:
    """Поза на квадратном холсте библиотеки: человек вписан с полями, по центру.

    Холст — как у поз openposes.com (768×768): плитки библиотеки одинаковы
    по форме, и скелет с фотографии встаёт в тот же ряд.
    """
    body = np.full((skeleton.POINTS, 3), 0.0)
    for coco, index in _COCO_TO_BODY18.items():
        if detection.scores[coco] >= POINT_SCORE:
            body[index] = (*detection.points[coco], 1.0)
    if body[skeleton.L_SHOULDER, 2] and body[skeleton.R_SHOULDER, 2]:
        body[skeleton.NECK] = (*((body[skeleton.L_SHOULDER, :2] + body[skeleton.R_SHOULDER, :2]) / 2), 1.0)

    seen = body[body[:, 2] > 0, :2]
    if len(seen) < 2:
        raise NoPersonFound("поза не распознана: видно меньше двух точек тела")
    low, high = seen.min(0), seen.max(0)
    extent = max(float((high - low).max()), 1.0)
    factor = canvas * (1 - 2 * margin) / extent
    offset = canvas / 2 - (low + high) / 2 * factor
    points = tuple(
        (float(x * factor + offset[0]), float(y * factor + offset[1]), 1.0) if c > 0 else (0.0, 0.0, 0.0)
        for x, y, c in body
    )
    return skeleton.Pose(points, canvas, canvas)

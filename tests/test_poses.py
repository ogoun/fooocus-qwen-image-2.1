"""Позы: скелет OpenPose, приведение распознанной позы, библиотека, плитка.

Каталог openposes.com и веса DWPose в репозиторий не входят; тесты, которым
они нужны, проверяют настоящие файлы, если те скачаны, и пропускаются, если
нет. Остальное — на синтетических данных.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from fooocus_qwen import config
from fooocus_qwen.poses import detect, library, skeleton, tile


def _pose(points: dict[int, tuple[float, float]], side: int = 768) -> skeleton.Pose:
    full = tuple((*points[i], 1.0) if i in points else (0.0, 0.0, 0.0) for i in range(skeleton.POINTS))
    return skeleton.Pose(full, side, side)


STANDING = {
    skeleton.NOSE: (384, 120), skeleton.NECK: (384, 200),
    skeleton.R_SHOULDER: (330, 200), skeleton.L_SHOULDER: (438, 200),
    skeleton.R_ELBOW: (310, 300), skeleton.L_ELBOW: (458, 300),
    skeleton.R_WRIST: (300, 390), skeleton.L_WRIST: (468, 390),
    skeleton.R_HIP: (350, 400), skeleton.L_HIP: (418, 400),
    skeleton.R_KNEE: (345, 540), skeleton.L_KNEE: (423, 540),
    skeleton.R_ANKLE: (340, 680), skeleton.L_ANKLE: (428, 680),
}


# --- скелет ---


def test_json_round_trip_keeps_points_and_canvas():
    pose = _pose(STANDING, 640)
    again = skeleton.parse(pose.to_json())
    assert again == pose


def test_parse_accepts_the_openposes_file_shape():
    flat = [v for i in range(skeleton.POINTS) for v in (float(i), float(2 * i), 1.0)]
    text = json.dumps([{"people": [{"pose_keypoints_2d": flat}], "canvas_width": 512, "canvas_height": 400}])
    pose = skeleton.parse(text)
    assert (pose.width, pose.height) == (512, 400)
    assert pose.points[3] == (3.0, 6.0, 1.0)


def test_parse_rejects_an_empty_canvas():
    with pytest.raises(ValueError):
        skeleton.parse(json.dumps({"people": []}))


def test_render_draws_limbs_in_controlnet_colours_on_black():
    image = np.asarray(skeleton.render(_pose(STANDING)))
    assert image.shape == (768, 768, 3)
    assert image[20, 20].tolist() == [0, 0, 0], "фон — чёрный"
    # Середина кости «шея — правое плечо»: красная (первый цвет ControlNet),
    # приглушённая прозрачностью кости.
    r, g, b = image[200, 357]
    assert r > 120 and g < 40 and b < 40
    # Сустав — поверх кости, полным цветом своей точки.
    assert image[300, 310].tolist() == list(skeleton.COLORS[skeleton.R_ELBOW])


def test_invisible_points_are_not_drawn():
    points = dict(STANDING)
    del points[skeleton.L_WRIST]
    image = np.asarray(skeleton.render(_pose(points)))
    assert image[390, 468].sum() == 0, "невидимое запястье — пустое место"
    assert image[345, 463].sum() == 0, "и кости к нему нет"


def test_render_matches_the_openposes_catalogue():
    """Своя отрисовка против картинок каталога: JSON и PNG там лежат парой."""
    pairs = sorted(config.POSE_LIBRARY_DIR.glob("*.json"))[:12]
    if not pairs:
        pytest.skip("каталог поз не скачан (tools/fetch_poses.py)")
    ious = []
    for path in pairs:
        reference = np.asarray(Image.open(path.with_suffix(".png")).convert("RGB")).astype(int).sum(2) > 60
        mine = np.asarray(skeleton.render(skeleton.load(path))).astype(int).sum(2) > 60
        ious.append((reference & mine).sum() / (reference | mine).sum())
    assert min(ious) > 0.85, f"совпадение с каталогом по площади: {min(ious):.2f}"


# --- распознанная поза → холст библиотеки ---


def _detection(scale: float = 1.0, shift: float = 0.0, low: tuple[int, ...] = ()) -> detect.Detection:
    coco = {
        0: (500, 100), 1: (510, 90), 2: (490, 90), 3: (520, 95), 4: (480, 95),
        5: (540, 160), 6: (460, 160), 7: (560, 240), 8: (440, 240), 9: (570, 320), 10: (430, 320),
        11: (530, 330), 12: (470, 330), 13: (535, 450), 14: (465, 450), 15: (540, 570), 16: (460, 570),
    }
    points = np.array([coco[i] for i in range(17)], dtype=float) * scale + shift
    scores = np.array([0.1 if i in low else 0.9 for i in range(17)])
    return detect.Detection(points, scores, np.array([400, 50, 600, 600], dtype=float))


def test_neck_is_the_middle_of_the_shoulders():
    pose = detect.to_pose(_detection())
    neck, right, left = (pose.points[i] for i in (skeleton.NECK, skeleton.R_SHOULDER, skeleton.L_SHOULDER))
    assert neck[2] == 1.0
    assert neck[0] == pytest.approx((right[0] + left[0]) / 2)


def test_left_and_right_follow_body18_order():
    """COCO 5 — левое плечо, 6 — правое; в BODY_18 правое плечо — точка 2."""
    pose = detect.to_pose(_detection())
    assert pose.points[skeleton.R_SHOULDER][0] < pose.points[skeleton.L_SHOULDER][0]


def test_the_figure_is_centred_and_fits_with_margins():
    for scale, shift in ((1.0, 0.0), (3.0, 900.0), (0.2, -40.0)):
        pose = detect.to_pose(_detection(scale, shift))
        seen = np.array([p[:2] for p in pose.points if p[2] > 0])
        assert (pose.width, pose.height) == (768, 768)
        assert seen.min() >= 768 * 0.08 - 1 and seen.max() <= 768 * 0.92 + 1
        centre = (seen.min(0) + seen.max(0)) / 2
        assert np.allclose(centre, 384, atol=1)


def test_low_confidence_points_are_invisible():
    pose = detect.to_pose(_detection(low=(9, 10)))
    assert not pose.visible(skeleton.L_WRIST) and not pose.visible(skeleton.R_WRIST)
    assert pose.visible(skeleton.L_ELBOW)


def test_no_neck_without_both_shoulders():
    pose = detect.to_pose(_detection(low=(5,)))
    assert not pose.visible(skeleton.NECK)


def test_too_few_points_is_no_person():
    with pytest.raises(detect.NoPersonFound):
        detect.to_pose(_detection(low=tuple(range(1, 17))))


def test_detector_finds_the_catalogue_pose():
    """Настоящий DWPose на плитке каталога против её же JSON."""
    weights = [config.DWPOSE_DIR / name for name in detect.FILES]
    tile_path = config.POSE_LIBRARY_DIR / "sitting_05.jpg"
    if not all(path.exists() for path in weights) or not tile_path.exists():
        pytest.skip("нет весов DWPose или каталога поз")
    truth = skeleton.load(tile_path.with_suffix(".json"))
    image = Image.open(tile_path)
    found = detect.PoseDetector(config.DWPOSE_DIR).detect(image)
    k = image.width / truth.width
    errors = [
        np.hypot(*(found.points[coco] - np.array(truth.points[index][:2]) * k)) / image.width
        for coco, index in detect._COCO_TO_BODY18.items()
        if truth.visible(index)
    ]
    assert np.median(errors) < 0.02, f"медианная ошибка {np.median(errors):.3f} ширины"


# --- библиотека ---


def _catalog_zip(names: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name in names:
            bundle.writestr(f"{name}.json", _pose(STANDING).to_json())
            png = io.BytesIO()
            skeleton.render(_pose(STANDING)).save(png, format="PNG")
            bundle.writestr(f"{name}.png", png.getvalue())
    return buffer.getvalue()


def _tile_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1024, 1024), "teal").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_fetch_catalog_downloads_archive_and_tiles(tmp_path):
    asked = []

    def downloader(url: str) -> bytes:
        asked.append(url)
        return _catalog_zip(["standing_01", "sitting_02"]) if url == library.ARCHIVE_URL else _tile_bytes()

    count = library.fetch_catalog(tmp_path, downloader=downloader)
    assert count == 2
    assert all(entry.thumb.exists() for entry in library.list_poses(tmp_path, tmp_path / "none"))
    assert library.TILE_URL.format(name="sitting_02") in asked
    thumb = Image.open(tmp_path / "standing_01.thumb.jpg")
    assert max(thumb.size) == library.THUMB_SIDE


def test_fetch_catalog_resumes_without_downloading_twice(tmp_path):
    asked = []

    def downloader(url: str) -> bytes:
        asked.append(url)
        return _catalog_zip(["a_01", "b_01"]) if url == library.ARCHIVE_URL else _tile_bytes()

    library.fetch_catalog(tmp_path, downloader=downloader)
    (tmp_path / "b_01.thumb.jpg").unlink()
    asked.clear()
    library.fetch_catalog(tmp_path, downloader=downloader)
    assert asked == [], "архив и плитка уже на диске — качать нечего, миниатюра строится заново"
    assert (tmp_path / "b_01.thumb.jpg").exists()


def test_fetch_catalog_takes_a_local_archive(tmp_path):
    archive = tmp_path / "poses.zip"
    archive.write_bytes(_catalog_zip(["x_01"]))
    asked = []
    library.fetch_catalog(tmp_path / "catalog", archive=archive, downloader=lambda url: asked.append(url) or _tile_bytes())
    assert library.ARCHIVE_URL not in asked


def test_custom_poses_follow_the_catalogue_and_show_their_skeleton_until_the_tile(tmp_path):
    catalog, user = tmp_path / "catalog", tmp_path / "user"
    library.fetch_catalog(catalog, archive=None, downloader=lambda url: (
        _catalog_zip(["z_01"]) if url == library.ARCHIVE_URL else _tile_bytes()))
    first = library.add_custom(user, _pose(STANDING))
    second = library.add_custom(user, _pose(STANDING))
    assert first.name != second.name, "две позы за секунду не затирают друг друга"
    entries = library.list_poses(catalog, user)
    assert [entry.name for entry in entries] == ["z_01", first.name, second.name]
    assert first.preview() == first.skeleton
    library.set_tile(first, Image.new("RGB", (1024, 1024), "red"))
    assert first.preview() == first.thumb


def test_list_poses_skips_a_pose_without_its_skeleton(tmp_path):
    (tmp_path / "broken.json").write_text(_pose(STANDING).to_json(), encoding="utf-8")
    assert library.list_poses(tmp_path, tmp_path / "none") == []


def test_the_catalogue_ships_with_the_repository():
    """Каталог — часть поставки: 46 поз openposes.com, у каждой четыре файла."""
    entries = library.list_poses(config.POSE_LIBRARY_DIR, config.POSE_LIBRARY_DIR / "none")
    assert len(entries) == 46
    for entry in entries:
        assert entry.keypoints.exists() and entry.skeleton.exists(), entry.name
        assert entry.tile.exists() and entry.thumb.exists(), entry.name
    assert config.POSE_LIBRARY_DIR.is_relative_to(config.RESOURCES_DIR)


def test_own_poses_live_with_the_generations(monkeypatch, tmp_path):
    """Свои позы — данные пользователя: в каталоге генераций, а не в поставке."""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "outputs")
    assert config.user_pose_dir() == tmp_path / "outputs" / "poses"
    entry = library.add_custom(config.user_pose_dir(), _pose(STANDING))
    assert entry.skeleton.parent == tmp_path / "outputs" / "poses"


def test_the_gallery_does_not_show_own_poses(monkeypatch, tmp_path):
    """В каталоге генераций рядом с днями лежат позы — это не результаты."""
    from fooocus_qwen.storage import gallery

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    day = tmp_path / "2026-09-26"
    day.mkdir()
    Image.new("RGB", (8, 8)).save(day / "12-00-00.png")
    library.add_custom(config.user_pose_dir(), _pose(STANDING))
    assert list((tmp_path / "poses").glob("*.png")), "поза сохранилась как PNG"
    assert gallery.recent(tmp_path) == [day / "12-00-00.png"]


# --- плитка ---


def test_tile_request_is_the_skeleton_alone_at_a_square():
    bones = skeleton.render(_pose(STANDING))
    request = tile.request(bones, turbo_ready=True)
    assert request.preset.name == "Turbo"
    assert len(request.references) == 1 and request.references[0].size == bones.size
    assert request.aspect == "1:1"
    assert "<image" not in request.prompt, "единственный референс — без тега"


def test_tile_does_not_download_turbo_for_an_icon():
    request = tile.request(skeleton.render(_pose(STANDING)), turbo_ready=False)
    assert request.preset.name == tile.FALLBACK_PRESET

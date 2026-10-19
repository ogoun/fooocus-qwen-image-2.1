"""Раскладка результатов по каталогам дат."""

from datetime import datetime

from PIL import Image

from fooocus_qwen.storage import gallery


def test_path_is_inside_a_dated_directory(tmp_path):
    path = gallery.next_path(tmp_path, when=datetime(2026, 9, 21, 15, 4, 5))
    assert path.parent.name == "2026-09-21"
    assert path.suffix == ".png"


def test_names_do_not_collide(tmp_path):
    when = datetime(2026, 9, 21, 15, 4, 5)
    first = gallery.next_path(tmp_path, when=when)
    first.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"")
    second = gallery.next_path(tmp_path, when=when)
    assert first != second


def test_recent_returns_newest_first(tmp_path):
    for index, day in enumerate((19, 20, 21)):
        path = gallery.next_path(tmp_path, when=datetime(2026, 9, day, 12, 0, index))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(path)

    recent = gallery.recent(tmp_path)
    assert len(recent) == 3
    assert recent[0].parent.name == "2026-09-21"
    assert recent[-1].parent.name == "2026-09-19"


def test_recent_respects_the_limit(tmp_path):
    for second in range(5):
        path = gallery.next_path(tmp_path, when=datetime(2026, 9, 21, 12, 0, second))
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(path)
    assert len(gallery.recent(tmp_path, limit=2)) == 2


def test_recent_on_empty_directory_is_empty(tmp_path):
    assert gallery.recent(tmp_path) == []


# --- ролики вкладки Video ---


def test_a_video_does_not_take_the_name_of_a_picture(tmp_path):
    # Иначе у 12-00-00.png и 12-00-00.mp4 был бы общий JSON параметров.
    when = datetime(2026, 10, 19, 12, 0, 0)
    picture = gallery.next_path(tmp_path, when=when)
    picture.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4)).save(picture)

    video = gallery.next_path(tmp_path, when=when, suffix=".mp4")
    assert video.suffix == ".mp4"
    assert video.stem != picture.stem


def test_a_picture_does_not_take_the_name_of_a_video(tmp_path):
    when = datetime(2026, 10, 19, 12, 0, 0)
    video = gallery.next_path(tmp_path, when=when, suffix=".mp4")
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"")

    picture = gallery.next_path(tmp_path, when=when)
    assert picture.suffix == ".png"
    assert picture.stem != video.stem


def test_recent_lists_pictures_and_videos_but_not_their_parameters(tmp_path):
    day = tmp_path / "2026-10-19"
    day.mkdir()
    Image.new("RGB", (4, 4)).save(day / "12-00-00.png")
    (day / "12-00-01.mp4").write_bytes(b"")
    (day / "12-00-01.json").write_text("{}", encoding="utf-8")
    (day / "notes.txt").write_text("", encoding="utf-8")

    assert [path.name for path in gallery.recent(tmp_path)] == ["12-00-01.mp4", "12-00-00.png"]


def test_the_storyboard_contact_sheet_is_not_a_result(tmp_path):
    """Контактный лист лежит рядом с роликом, но в галерею не попадает."""
    day = tmp_path / "2026-10-19"
    day.mkdir()
    for name in ("12-00-00.mp4", "12-00-00.json", "12-00-00.sheet.png", "11-00-00.png"):
        (day / name).write_bytes(b"x")
    assert [path.name for path in gallery.recent(tmp_path)] == ["12-00-00.mp4", "11-00-00.png"]

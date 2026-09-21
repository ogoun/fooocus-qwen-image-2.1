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

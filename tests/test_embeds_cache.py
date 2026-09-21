"""Кэш эмбеддингов промта."""

import torch
from PIL import Image

from fooocus_qwen.engine.embeds_cache import EmbedsCache, fingerprint


def sample():
    return (torch.zeros(1, 4, 8), torch.ones(1, 4, dtype=torch.long), torch.zeros(1, 4, dtype=torch.bool))


def test_identical_images_have_identical_fingerprints():
    first = Image.new("RGB", (8, 8), "red")
    second = Image.new("RGB", (8, 8), "red")
    assert fingerprint(first) == fingerprint(second)


def test_different_pixels_change_the_fingerprint():
    assert fingerprint(Image.new("RGB", (8, 8), "red")) != fingerprint(Image.new("RGB", (8, 8), "blue"))


def test_different_size_changes_the_fingerprint():
    assert fingerprint(Image.new("RGB", (8, 8), "red")) != fingerprint(Image.new("RGB", (16, 16), "red"))


def test_miss_then_hit():
    cache = EmbedsCache()
    key = cache.key("кот", None)

    assert cache.get(key) is None
    assert cache.misses == 1

    cache.put(key, sample())
    assert cache.get(key) is not None
    assert cache.hits == 1


def test_key_depends_on_prompt_and_images():
    cache = EmbedsCache()
    image = Image.new("RGB", (8, 8), "red")
    assert cache.key("кот", None) != cache.key("пёс", None)
    assert cache.key("кот", None) != cache.key("кот", [image])


def test_prompt_may_be_a_string_or_a_list():
    cache = EmbedsCache()
    assert cache.key("кот", None) == cache.key(["кот"], None)


def test_capacity_evicts_the_least_recently_used():
    cache = EmbedsCache(capacity=2)
    first, second, third = (cache.key(text, None) for text in ("a", "b", "c"))

    cache.put(first, sample())
    cache.put(second, sample())
    cache.get(first)          # first становится свежим, вытеснить должно second
    cache.put(third, sample())

    assert cache.get(first) is not None
    assert cache.get(second) is None
    assert cache.size == 2


def test_capacity_of_one_keeps_only_the_freshest_entry():
    # Вырожденный случай цикла вытеснения: с capacity=1 каждая новая запись
    # должна вытеснять единственную существующую, а не накапливаться рядом.
    cache = EmbedsCache(capacity=1)
    first, second = (cache.key(text, None) for text in ("a", "b"))

    cache.put(first, sample())
    cache.put(second, sample())

    assert cache.get(first) is None
    assert cache.get(second) is not None
    assert cache.size == 1


def test_stored_tensors_live_on_the_host():
    # Кэш не должен занимать видеопамять: с десятью референсами запись весит
    # десятки мегабайт, а записей несколько.
    cache = EmbedsCache()
    key = cache.key("кот", None)
    cache.put(key, sample())
    for tensor in cache.get(key):
        assert tensor.device.type == "cpu"


def test_clear_empties_the_cache_and_counters():
    cache = EmbedsCache()
    key = cache.key("кот", None)
    cache.put(key, sample())
    cache.clear()
    assert cache.size == 0
    assert cache.hits == 0 and cache.misses == 0

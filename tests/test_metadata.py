"""Запись параметров генерации в PNG и их чтение обратно."""

from PIL import Image

from fooocus_qwen.imaging import metadata

PARAMS = {
    "app_version": "0.1.0",
    "prompt": "кот в шляпе",
    "prompt_boosted": "a cat wearing a hat, studio light",
    "negative_prompt": "",
    "styles": ["sai-anime"],
    "seed": 12345,
    "steps": 28,
    "width": 1024,
    "height": 1024,
    "output_resolution": 1024,
    "true_cfg_scale": 1.0,
    "use_kv_cache": True,
    "mask_mode": "none",
    "references": 0,
    "seconds": 31.4,
}


def test_parameters_survive_a_round_trip(tmp_path):
    path = metadata.save_png(Image.new("RGB", (16, 16), "red"), tmp_path / "a.png", PARAMS)
    assert metadata.read_png(path) == PARAMS


def test_rgba_transparency_is_preserved(tmp_path):
    image = Image.new("RGBA", (8, 8), (255, 0, 0, 0))
    path = metadata.save_png(image, tmp_path / "t.png", PARAMS)
    assert Image.open(path).mode == "RGBA"
    assert Image.open(path).getpixel((4, 4))[3] == 0


def test_human_readable_chunk_is_written_too(tmp_path):
    # Сторонние просмотрщики умеют читать только ключ parameters.
    path = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "b.png", PARAMS)
    text = Image.open(path).text
    assert metadata.CHUNK_KEY in text
    assert metadata.LEGACY_KEY in text
    assert "кот в шляпе" in text[metadata.LEGACY_KEY]


def test_foreign_png_returns_none(tmp_path):
    path = tmp_path / "foreign.png"
    Image.new("RGB", (8, 8), "blue").save(path)
    assert metadata.read_png(path) is None


def test_broken_json_returns_none(tmp_path):
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text(metadata.CHUNK_KEY, "{это не json")
    path = tmp_path / "broken.png"
    Image.new("RGB", (8, 8)).save(path, pnginfo=info)
    assert metadata.read_png(path) is None


def test_missing_file_returns_none(tmp_path):
    assert metadata.read_png(tmp_path / "нет.png") is None


def test_readable_form_lists_key_parameters():
    text = metadata.to_readable(PARAMS)
    assert "Seed: 12345" in text
    assert "Steps: 28" in text
    assert "Size: 1024x1024" in text


def test_non_ascii_keys_survive(tmp_path):
    params = dict(PARAMS, prompt="Тест «кавычки» и emoji 🐈")
    path = metadata.save_png(Image.new("RGB", (8, 8)), tmp_path / "u.png", params)
    assert metadata.read_png(path)["prompt"] == "Тест «кавычки» и emoji 🐈"

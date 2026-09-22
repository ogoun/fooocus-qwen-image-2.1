"""Веса приезжают сами, а «уже скачано» проверяется по индексу шардов.

Тридцать три гигабайта — не та величина, о которой можно спросить «а не
скачать ли ещё раз». Поэтому наличие проверяется строго: не по каталогу и
даже не по `model_index.json`, а по тем файлам, которые сама модель
перечисляет в своих `*.safetensors.index.json`. Оборванная закачка так
отличается от полной, а полная не перекачивается.
"""

from __future__ import annotations

import json

import pytest

from fooocus_qwen.engine import fetch


def _model(root, *, shards: int = 2, drop: tuple[str, ...] = ()) -> None:
    """Строит подобие каталога весов: индекс, шарды трансформера, VAE."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))

    transformer = root / "transformer"
    transformer.mkdir(exist_ok=True)
    names = [f"diffusion_pytorch_model-{i + 1:05d}-of-{shards:05d}.safetensors" for i in range(shards)]
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"block.{i}": name for i, name in enumerate(names)}})
    )
    for name in names:
        if name not in drop:
            (transformer / name).write_bytes(b"weights")

    vae = root / "vae"
    vae.mkdir(exist_ok=True)
    (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")


def test_a_complete_model_is_not_downloaded_again(tmp_path):
    model_dir = tmp_path / "Qwen-Image-2.1"
    _model(model_dir)

    calls = []
    downloaded = fetch.ensure_model(model_dir, downloader=lambda **kwargs: calls.append(kwargs))

    assert downloaded is False, "полные веса перекачивать нельзя"
    assert calls == []


def test_an_absent_model_is_downloaded(tmp_path):
    model_dir = tmp_path / "Qwen-Image-2.1"
    calls = []

    def downloader(**kwargs):
        calls.append(kwargs)
        _model(model_dir)

    assert fetch.ensure_model(model_dir, downloader=downloader) is True
    assert calls and calls[0]["repo_id"] == fetch.REPO_ID
    assert calls[0]["local_dir"] == model_dir


def test_an_interrupted_download_is_finished(tmp_path):
    """Половина шардов на диске — это не «скачано»."""
    model_dir = tmp_path / "Qwen-Image-2.1"
    _model(model_dir, drop=("diffusion_pytorch_model-00002-of-00002.safetensors",))
    assert fetch.missing_files(model_dir), "неполная закачка обязана обнаруживаться"

    def downloader(**_kwargs):
        _model(model_dir)

    assert fetch.ensure_model(model_dir, downloader=downloader) is True
    assert fetch.missing_files(model_dir) == []


def test_an_empty_shard_counts_as_missing(tmp_path):
    """Файл нулевой длины остаётся от оборванной записи и весами не является."""
    model_dir = tmp_path / "Qwen-Image-2.1"
    _model(model_dir)
    (model_dir / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"")

    assert "vae/diffusion_pytorch_model.safetensors" in fetch.missing_files(model_dir)


def test_a_download_that_brought_nothing_is_an_error(tmp_path):
    """Молча вернуть «готово» с неполными весами — худшее из возможного."""
    model_dir = tmp_path / "Qwen-Image-2.1"

    with pytest.raises(fetch.ModelDownloadError) as error:
        fetch.ensure_model(model_dir, downloader=lambda **_kwargs: None)

    assert "model_index.json" in str(error.value)

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dinoplay.index import EmbeddingIndex


def test_empty_dir_produces_empty_index(tmp_path: Path, fake_encoder, cache_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    idx = EmbeddingIndex.build_or_load(
        images_dir=empty_dir,
        cache_path=cache_path,
        encoder=fake_encoder,
        model_id=fake_encoder.model_id,
        extensions=frozenset({".png"}),
    )
    assert idx.is_empty
    assert len(idx) == 0


def test_build_encodes_all_then_caches(image_dir: Path, fake_encoder, cache_path):
    idx = EmbeddingIndex.build_or_load(
        images_dir=image_dir,
        cache_path=cache_path,
        encoder=fake_encoder,
        model_id=fake_encoder.model_id,
        extensions=frozenset({".png"}),
    )
    assert len(idx) == 4
    assert fake_encoder.calls == 4
    assert cache_path.exists()


def test_rebuild_reuses_cache(image_dir: Path, fake_encoder, cache_path):
    EmbeddingIndex.build_or_load(
        images_dir=image_dir,
        cache_path=cache_path,
        encoder=fake_encoder,
        model_id=fake_encoder.model_id,
        extensions=frozenset({".png"}),
    )
    fake_encoder.calls = 0
    idx2 = EmbeddingIndex.build_or_load(
        images_dir=image_dir,
        cache_path=cache_path,
        encoder=fake_encoder,
        model_id=fake_encoder.model_id,
        extensions=frozenset({".png"}),
    )
    assert len(idx2) == 4
    assert fake_encoder.calls == 0


def test_changed_file_is_reencoded(image_dir: Path, fake_encoder, cache_path):
    EmbeddingIndex.build_or_load(
        images_dir=image_dir,
        cache_path=cache_path,
        encoder=fake_encoder,
        model_id=fake_encoder.model_id,
        extensions=frozenset({".png"}),
    )
    target = image_dir / "img_0.png"
    Image.new("RGB", (16, 16), color=(123, 45, 67)).save(target)
    future = time.time() + 5
    os.utime(target, (future, future))

    fake_encoder.calls = 0
    idx2 = EmbeddingIndex.build_or_load(
        images_dir=image_dir,
        cache_path=cache_path,
        encoder=fake_encoder,
        model_id=fake_encoder.model_id,
        extensions=frozenset({".png"}),
    )
    assert len(idx2) == 4
    assert fake_encoder.calls == 1


def test_model_id_change_invalidates_cache(image_dir: Path, fake_encoder, cache_path):
    EmbeddingIndex.build_or_load(
        images_dir=image_dir,
        cache_path=cache_path,
        encoder=fake_encoder,
        model_id="model-a",
        extensions=frozenset({".png"}),
    )
    fake_encoder.calls = 0
    EmbeddingIndex.build_or_load(
        images_dir=image_dir,
        cache_path=cache_path,
        encoder=fake_encoder,
        model_id="model-b",
        extensions=frozenset({".png"}),
    )
    assert fake_encoder.calls == 4


def test_corrupt_cache_is_recovered(image_dir: Path, fake_encoder, cache_path):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"not a real npz file")
    idx = EmbeddingIndex.build_or_load(
        images_dir=image_dir,
        cache_path=cache_path,
        encoder=fake_encoder,
        model_id=fake_encoder.model_id,
        extensions=frozenset({".png"}),
    )
    assert len(idx) == 4
    assert fake_encoder.calls == 4


def test_search_self_match_is_top(image_dir: Path, fake_encoder, cache_path):
    idx = EmbeddingIndex.build_or_load(
        images_dir=image_dir,
        cache_path=cache_path,
        encoder=fake_encoder,
        model_id=fake_encoder.model_id,
        extensions=frozenset({".png"}),
    )
    img = Image.open(image_dir / "img_2.png").convert("RGB")
    q = fake_encoder.encode([img])
    hits = idx.search(q[0], k=4)

    assert len(hits) == 4
    assert Path(hits[0].path).name == "img_2.png"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_unreadable_file_is_skipped(image_dir: Path, fake_encoder, cache_path, caplog):
    bad = image_dir / "broken.png"
    bad.write_bytes(b"not an image")
    idx = EmbeddingIndex.build_or_load(
        images_dir=image_dir,
        cache_path=cache_path,
        encoder=fake_encoder,
        model_id=fake_encoder.model_id,
        extensions=frozenset({".png"}),
    )
    assert len(idx) == 4
    assert any("broken.png" in rec.message for rec in caplog.records)


def test_recursive_scan_finds_one_level_deep(tmp_path: Path, fake_encoder, cache_path):
    root = tmp_path / "labels"
    (root / "mug").mkdir(parents=True)
    (root / "keyboard").mkdir(parents=True)
    (root / "mug" / "nested").mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(root / "mug" / "a.png")
    Image.new("RGB", (8, 8), color=(0, 255, 0)).save(root / "mug" / "b.png")
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(root / "keyboard" / "c.png")
    # File two levels deep should NOT be found.
    Image.new("RGB", (8, 8), color=(255, 255, 0)).save(root / "mug" / "nested" / "deep.png")
    # File at the top level (no subfolder) should NOT be found in recursive mode.
    Image.new("RGB", (8, 8), color=(0, 255, 255)).save(root / "loose.png")

    idx = EmbeddingIndex.build_or_load(
        images_dir=root,
        cache_path=cache_path,
        encoder=fake_encoder,
        model_id=fake_encoder.model_id,
        extensions=frozenset({".png"}),
        recursive=True,
    )
    rels = sorted(str(Path(p).relative_to(root)) for p in idx.paths)
    assert rels == ["keyboard/c.png", "mug/a.png", "mug/b.png"]

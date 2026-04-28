from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
from PIL import Image

from dinoplay.labels import LabelIndex, Prediction


class StubEncoder:
    """L2-normalized encoder. Each call returns vectors from a queue, in order."""

    DIM = 4
    model_id = "stub-model"

    def __init__(self, queue: list[np.ndarray] | None = None) -> None:
        self.queue: list[np.ndarray] = list(queue or [])
        self.calls = 0

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.calls += len(images)
        if len(images) == 0:
            return np.zeros((0, self.DIM), dtype=np.float32)
        out = []
        for _ in images:
            v = self.queue.pop(0).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out, axis=0)


@pytest.fixture
def labels_root(tmp_path: Path) -> Path:
    d = tmp_path / "labels"
    d.mkdir()
    return d


@pytest.fixture
def labels_cache(tmp_path: Path) -> Path:
    return tmp_path / "cache" / "labels.npz"


def test_empty_label_index(labels_root: Path, labels_cache: Path):
    enc = StubEncoder()
    idx = LabelIndex.build_or_load(
        labels_dir=labels_root,
        cache_path=labels_cache,
        encoder=enc,
        model_id=enc.model_id,
        extensions=frozenset({".jpg", ".png"}),
    )
    assert idx.is_empty
    assert idx.classes() == []
    assert idx.count("anything") == 0
    pred = idx.predict(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=5)
    assert pred == Prediction(label=None, confidence=0.0, hits=[])


def _write_labelled_files(root: Path, layout: dict[str, int]) -> None:
    """Create `count` PNG files under root/<class>/ for each class. Content varies by file."""
    for cls, count in layout.items():
        sub = root / cls
        sub.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            color = (i * 17 % 256, (i * 53 + ord(cls[0])) % 256, (i * 91) % 256)
            Image.new("RGB", (8, 8), color=color).save(sub / f"{cls}_{i}.png")


def _evec(dim: int, idx: int, scale: float = 1.0) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = scale
    return v


def test_predict_clean_signal(labels_root: Path, labels_cache: Path):
    # 3 classes, 3 examples each. Each class lives on its own basis axis,
    # so the closest k will all be from the queried class.
    _write_labelled_files(labels_root, {"apple": 3, "banana": 3, "cherry": 3})
    # Order of files seen by the index: alphabetical by (class, filename).
    # apple gets axis 0, banana axis 1, cherry axis 2.
    queue = (
        [_evec(StubEncoder.DIM, 0)] * 3
        + [_evec(StubEncoder.DIM, 1)] * 3
        + [_evec(StubEncoder.DIM, 2)] * 3
    )
    enc = StubEncoder(queue=queue)
    idx = LabelIndex.build_or_load(
        labels_dir=labels_root,
        cache_path=labels_cache,
        encoder=enc,
        model_id=enc.model_id,
        extensions=frozenset({".png"}),
    )
    pred = idx.predict(_evec(StubEncoder.DIM, 1), k=3)
    assert pred.label == "banana"
    assert pred.confidence == pytest.approx(1.0)
    assert len(pred.hits) == 3
    for h in pred.hits:
        assert "/banana/" in h.path


def test_predict_tie_break_alphabetical(labels_root: Path, labels_cache: Path):
    # 2 examples per class. Query is equidistant from apple-0 and banana-0,
    # and from apple-1 and banana-1 (so top-4 is 2 apple + 2 banana). Tie → "apple".
    _write_labelled_files(labels_root, {"apple": 2, "banana": 2})
    # Build embeddings so that for each class, the two examples have identical
    # similarity to the query. Query = [1, 1, 0, 0] / sqrt(2). All four labelled
    # vectors put equal weight on axes 0 and 1.
    same = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    queue = [same.copy() for _ in range(4)]
    enc = StubEncoder(queue=queue)
    idx = LabelIndex.build_or_load(
        labels_dir=labels_root,
        cache_path=labels_cache,
        encoder=enc,
        model_id=enc.model_id,
        extensions=frozenset({".png"}),
    )
    pred = idx.predict(np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32), k=4)
    assert pred.label == "apple"
    assert pred.confidence == pytest.approx(0.5)


def test_predict_clamps_k_to_index_size(labels_root: Path, labels_cache: Path):
    _write_labelled_files(labels_root, {"apple": 2})
    enc = StubEncoder(queue=[_evec(StubEncoder.DIM, 0)] * 2)
    idx = LabelIndex.build_or_load(
        labels_dir=labels_root,
        cache_path=labels_cache,
        encoder=enc,
        model_id=enc.model_id,
        extensions=frozenset({".png"}),
    )
    pred = idx.predict(_evec(StubEncoder.DIM, 0), k=10)
    assert pred.label == "apple"
    assert len(pred.hits) == 2
    assert pred.confidence == pytest.approx(1.0)  # 2 votes / 2 effective k


def test_add_writes_file_and_grows_index(labels_root: Path, labels_cache: Path):
    enc = StubEncoder(queue=[_evec(StubEncoder.DIM, 0)])
    idx = LabelIndex.build_or_load(
        labels_dir=labels_root,
        cache_path=labels_cache,
        encoder=enc,
        model_id=enc.model_id,
        extensions=frozenset({".png"}),
    )
    assert idx.is_empty
    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    idx.add("mug", [img])

    assert idx.count("mug") == 1
    assert idx.classes() == ["mug"]
    files = list((labels_root / "mug").iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".png"

    # Cache file should exist now.
    assert labels_cache.exists()


def test_add_persists_across_reload(labels_root: Path, labels_cache: Path):
    enc = StubEncoder(queue=[_evec(StubEncoder.DIM, 0), _evec(StubEncoder.DIM, 1)])
    idx = LabelIndex.build_or_load(
        labels_dir=labels_root,
        cache_path=labels_cache,
        encoder=enc,
        model_id=enc.model_id,
        extensions=frozenset({".png"}),
    )
    idx.add("mug", [Image.new("RGB", (8, 8), color=(10, 20, 30))])
    idx.add("keyboard", [Image.new("RGB", (8, 8), color=(40, 50, 60))])

    # Rebuild from disk; should not re-encode (cache hit) and should see both.
    enc2 = StubEncoder(queue=[])  # empty queue: any encode call would IndexError
    idx2 = LabelIndex.build_or_load(
        labels_dir=labels_root,
        cache_path=labels_cache,
        encoder=enc2,
        model_id=enc2.model_id,
        extensions=frozenset({".png"}),
    )
    assert sorted(idx2.classes()) == ["keyboard", "mug"]
    assert idx2.count("mug") == 1
    assert idx2.count("keyboard") == 1
    assert enc2.calls == 0


def test_add_two_to_same_class_no_collision(labels_root: Path, labels_cache: Path):
    enc = StubEncoder(queue=[_evec(StubEncoder.DIM, 0), _evec(StubEncoder.DIM, 0)])
    idx = LabelIndex.build_or_load(
        labels_dir=labels_root,
        cache_path=labels_cache,
        encoder=enc,
        model_id=enc.model_id,
        extensions=frozenset({".png"}),
    )
    idx.add("mug", [Image.new("RGB", (8, 8), color=(10, 20, 30))])
    idx.add("mug", [Image.new("RGB", (8, 8), color=(40, 50, 60))])
    assert idx.count("mug") == 2
    files = sorted((labels_root / "mug").iterdir())
    assert len(files) == 2
    assert files[0].name != files[1].name

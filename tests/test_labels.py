from __future__ import annotations

import hashlib
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

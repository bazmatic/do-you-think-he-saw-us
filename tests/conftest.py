from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
from PIL import Image


class FakeEncoder:
    """Deterministic L2-normalized fake encoder. Counts calls per image hash."""

    DIM = 8
    model_id = "fake-model"

    def __init__(self) -> None:
        self.calls = 0
        self.calls_per_hash: dict[str, int] = {}

    def _vec_for(self, img: Image.Image) -> np.ndarray:
        h = hashlib.sha256(img.tobytes()).hexdigest()
        self.calls_per_hash[h] = self.calls_per_hash.get(h, 0) + 1
        seed = int(h[:8], 16) % (2**31 - 1)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self.DIM).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.calls += len(images)
        if len(images) == 0:
            return np.zeros((0, self.DIM), dtype=np.float32)
        return np.stack([self._vec_for(im) for im in images], axis=0).astype(np.float32)


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """Creates 4 small distinct PNG files in a temp dir."""
    d = tmp_path / "imgs"
    d.mkdir()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, c in enumerate(colors):
        Image.new("RGB", (8, 8), color=c).save(d / f"img_{i}.png")
    return d


@pytest.fixture
def cache_path(tmp_path: Path) -> Path:
    return tmp_path / "cache" / "embeddings.npz"


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from dinoplay.index import EmbeddingIndex, SearchHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    label: str | None
    confidence: float
    hits: list[SearchHit]


class _EncoderLike(Protocol):
    model_id: str

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray: ...


def _class_of(relpath: str) -> str:
    return relpath.split("/", 1)[0]


class LabelIndex:
    def __init__(
        self,
        labels_dir: Path,
        cache_path: Path,
        encoder: _EncoderLike,
        model_id: str,
        extensions: frozenset[str],
        inner: EmbeddingIndex,
    ) -> None:
        self._labels_dir = Path(labels_dir)
        self._cache_path = Path(cache_path)
        self._encoder = encoder
        self._model_id = model_id
        self._extensions = extensions
        self._inner = inner

    @classmethod
    def build_or_load(
        cls,
        labels_dir: Path,
        cache_path: Path,
        encoder: _EncoderLike,
        model_id: str,
        extensions: frozenset[str],
    ) -> "LabelIndex":
        labels_dir = Path(labels_dir)
        labels_dir.mkdir(parents=True, exist_ok=True)
        inner = EmbeddingIndex.build_or_load(
            images_dir=labels_dir,
            cache_path=cache_path,
            encoder=encoder,
            model_id=model_id,
            extensions=extensions,
            recursive=True,
        )
        return cls(
            labels_dir=labels_dir,
            cache_path=cache_path,
            encoder=encoder,
            model_id=model_id,
            extensions=extensions,
            inner=inner,
        )

    @property
    def is_empty(self) -> bool:
        return self._inner.is_empty

    def classes(self) -> list[str]:
        seen: set[str] = set()
        for full in self._inner.paths:
            rel = Path(full).relative_to(self._labels_dir).as_posix()
            seen.add(_class_of(rel))
        return sorted(seen)

    def count(self, label: str) -> int:
        n = 0
        for full in self._inner.paths:
            rel = Path(full).relative_to(self._labels_dir).as_posix()
            if _class_of(rel) == label:
                n += 1
        return n

    def predict(self, query_emb: np.ndarray, k: int = 5) -> Prediction:
        if self._inner.is_empty:
            return Prediction(label=None, confidence=0.0, hits=[])
        effective_k = min(k, len(self._inner))
        hits = self._inner.search(query_emb, k=effective_k)
        votes: dict[str, int] = {}
        for h in hits:
            rel = Path(h.path).relative_to(self._labels_dir).as_posix()
            cls = _class_of(rel)
            votes[cls] = votes.get(cls, 0) + 1
        # Sort by (-votes, class_name) so ties break alphabetically.
        winner = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        label, count = winner
        confidence = count / effective_k
        return Prediction(label=label, confidence=confidence, hits=hits)

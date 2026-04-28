from __future__ import annotations

import logging
import re
import shutil
import time
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


_VALID_NAME = re.compile(r"^[a-z0-9_-]+$")


def sanitize_class_name(raw: str) -> str | None:
    """Normalize a user-entered class name to lowercase, snake_case, [a-z0-9_-].

    Returns the sanitized name, or None if the result would be invalid.
    """
    if raw is None:
        return None
    s = raw.strip().lower()
    s = re.sub(r"\s+", "_", s)
    if not s:
        return None
    if not _VALID_NAME.match(s):
        return None
    return s


def _class_of(relpath: str) -> str:
    return relpath.split("/", 1)[0]


def _resolve_class_dir(labels_dir: Path, label: str) -> Path:
    """Return labels_dir/label, raising ValueError if it escapes labels_dir."""
    class_dir = labels_dir / label
    try:
        class_dir.resolve().relative_to(labels_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"label {label!r} escapes the labels directory") from exc
    return class_dir


def _next_filename(class_dir: Path, ext: str) -> Path:
    """Return a fresh path under class_dir with format YYYYMMDD-HHMMSS-NNN<ext>."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for n in range(1, 1000):
        candidate = class_dir / f"{stamp}-{n:03d}{ext}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(
        f"more than 999 files written to {class_dir} in one second; refusing to extend NNN past three digits"
    )


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

    def add(self, label: str, images: list[Image.Image]) -> None:
        if not images:
            return
        class_dir = _resolve_class_dir(self._labels_dir, label)
        class_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []
        for img in images:
            out = _next_filename(class_dir, ".png")
            img.convert("RGB").save(out, format="PNG")
            saved_paths.append(out)
        # Reload the inner index from disk; cache reuse will skip re-encoding
        # the unchanged files and only encode the just-written ones.
        self._inner = EmbeddingIndex.build_or_load(
            images_dir=self._labels_dir,
            cache_path=self._cache_path,
            encoder=self._encoder,
            model_id=self._model_id,
            extensions=self._extensions,
            recursive=True,
        )

    def delete_class(self, label: str) -> None:
        class_dir = _resolve_class_dir(self._labels_dir, label)
        if class_dir.exists() and class_dir.is_dir():
            shutil.rmtree(class_dir)
        # Reload the inner index from disk so cache + arrays reflect the deletion.
        self._inner = EmbeddingIndex.build_or_load(
            images_dir=self._labels_dir,
            cache_path=self._cache_path,
            encoder=self._encoder,
            model_id=self._model_id,
            extensions=self._extensions,
            recursive=True,
        )

    def predict(self, query_emb: np.ndarray, k: int = 5) -> Prediction:
        if self._inner.is_empty:
            return Prediction(label=None, confidence=0.0, hits=[])
        # Clamp k locally so the confidence denominator below matches len(hits).
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

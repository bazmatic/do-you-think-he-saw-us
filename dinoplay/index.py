from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    path: str
    score: float


class _EncoderLike(Protocol):
    def encode(self, images: Sequence[Image.Image]) -> np.ndarray: ...


@dataclass(frozen=True)
class _Entry:
    relpath: str
    mtime: float
    size: int


def _scan(images_dir: Path, extensions: frozenset[str]) -> list[_Entry]:
    if not images_dir.exists() or not images_dir.is_dir():
        return []
    entries: list[_Entry] = []
    for name in sorted(os.listdir(images_dir)):
        path = images_dir / name
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        st = path.stat()
        entries.append(_Entry(relpath=name, mtime=st.st_mtime, size=st.st_size))
    return entries


def _load_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            return {
                "paths": data["paths"].tolist(),
                "mtimes": data["mtimes"],
                "sizes": data["sizes"],
                "embeddings": data["embeddings"],
                "model_id": str(data["model_id"]),
            }
    except Exception as exc:
        logger.warning("Failed to read cache at %s (%s); rebuilding.", cache_path, exc)
        try:
            cache_path.unlink()
        except OSError:
            pass
        return None


def _save_cache(
    cache_path: Path,
    paths: list[str],
    mtimes: np.ndarray,
    sizes: np.ndarray,
    embeddings: np.ndarray,
    model_id: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    # Use Unicode-string arrays so the cache loads with allow_pickle=False.
    # Pass an open file handle so numpy doesn't auto-append ".npz" to our .tmp path.
    with open(tmp, "wb") as f:
        np.savez(
            f,
            paths=np.array(paths, dtype=np.str_),
            mtimes=mtimes.astype(np.float64),
            sizes=sizes.astype(np.int64),
            embeddings=embeddings.astype(np.float32),
            model_id=np.array(model_id, dtype=np.str_),
        )
    os.replace(tmp, cache_path)


class EmbeddingIndex:
    def __init__(
        self,
        images_dir: Path,
        paths: list[str],
        embeddings: np.ndarray,
    ) -> None:
        self.images_dir = images_dir
        self._paths = paths
        self._embeddings = embeddings

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def is_empty(self) -> bool:
        return len(self._paths) == 0

    @property
    def paths(self) -> list[str]:
        return [str(self.images_dir / p) for p in self._paths]

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    def search(self, query_emb: np.ndarray, k: int) -> list[SearchHit]:
        if self.is_empty:
            return []
        q = query_emb.reshape(-1).astype(np.float32)
        scores = self._embeddings @ q
        k = min(k, len(scores))
        top_idx = np.argpartition(-scores, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            SearchHit(path=str(self.images_dir / self._paths[i]), score=float(scores[i]))
            for i in top_idx
        ]

    @classmethod
    def build_or_load(
        cls,
        images_dir: Path,
        cache_path: Path,
        encoder: _EncoderLike,
        model_id: str,
        extensions: frozenset[str],
    ) -> "EmbeddingIndex":
        images_dir = Path(images_dir)
        cache_path = Path(cache_path)
        entries = _scan(images_dir, extensions)

        if not entries:
            return cls(images_dir=images_dir, paths=[], embeddings=np.zeros((0, 1), dtype=np.float32))

        cache = _load_cache(cache_path)
        cached_lookup: dict[str, tuple[float, int, np.ndarray]] = {}
        if cache is not None and cache["model_id"] == model_id:
            for i, p in enumerate(cache["paths"]):
                cached_lookup[p] = (
                    float(cache["mtimes"][i]),
                    int(cache["sizes"][i]),
                    cache["embeddings"][i],
                )

        to_encode_indices: list[int] = []
        reused: dict[int, np.ndarray] = {}
        for i, entry in enumerate(entries):
            cached = cached_lookup.get(entry.relpath)
            if cached is not None and cached[0] == entry.mtime and cached[1] == entry.size:
                reused[i] = cached[2]
            else:
                to_encode_indices.append(i)

        loaded: list[tuple[int, Image.Image]] = []
        for i in to_encode_indices:
            path = images_dir / entries[i].relpath
            try:
                img = Image.open(path).convert("RGB")
                img.load()
                loaded.append((i, img))
            except Exception as exc:
                logger.warning("Skipping unreadable image %s: %s", path, exc)

        new_embeds: dict[int, np.ndarray] = {}
        if loaded:
            arr = encoder.encode([img for _, img in loaded])
            for (i, _), vec in zip(loaded, arr):
                new_embeds[i] = vec

        final_paths: list[str] = []
        final_mtimes: list[float] = []
        final_sizes: list[int] = []
        final_embeds: list[np.ndarray] = []
        for i, entry in enumerate(entries):
            if i in reused:
                vec = reused[i]
            elif i in new_embeds:
                vec = new_embeds[i]
            else:
                continue
            final_paths.append(entry.relpath)
            final_mtimes.append(entry.mtime)
            final_sizes.append(entry.size)
            final_embeds.append(vec)

        if final_embeds:
            embeddings = np.stack(final_embeds, axis=0).astype(np.float32)
        else:
            embeddings = np.zeros((0, 1), dtype=np.float32)

        if final_paths:
            _save_cache(
                cache_path,
                paths=final_paths,
                mtimes=np.array(final_mtimes, dtype=np.float64),
                sizes=np.array(final_sizes, dtype=np.int64),
                embeddings=embeddings,
                model_id=model_id,
            )

        return cls(images_dir=images_dir, paths=final_paths, embeddings=embeddings)

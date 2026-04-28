# Labels and Label Live Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Label Capture tab that lets the user save webcam photos under class names, and a Label Live tab that predicts the class of the live webcam frame using k-NN majority vote over the labelled set.

**Architecture:** Reuse the existing `EmbeddingIndex` (extended with optional one-level recursion) for the labelled set. A new `LabelIndex` wraps it, derives classes from `<class>/<file>` relpaths, and adds `add()`, `delete_class()`, and `predict()`. Two new Gradio tabs (Label Capture, Label Live) are added after the existing Live tab. The label index is built once at startup alongside the existing image index and shared between both tabs.

**Tech Stack:** Python 3.11, NumPy, Pillow, Gradio, PyTorch (already in use). pytest for tests. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-29-labels-and-label-live-design.md`

---

## File Structure

- **Modify** `dinoplay/index.py` — add `recursive: bool = False` to `_scan` (and thread it through `EmbeddingIndex.build_or_load`). One-level recursion only.
- **Create** `dinoplay/labels.py` — `Prediction` dataclass and `LabelIndex` class.
- **Modify** `dinoplay/app.py` — accept `label_index` parameter, add Label Capture and Label Live tabs after the existing Live tab.
- **Modify** `dinoplay/cli.py` — build the `LabelIndex` at startup and pass it to `build_app`.
- **Modify** `dinoplay/config.py` — add `labels_dir` setting and `labels_cache_path` property.
- **Modify** `dinoplay/__init__.py` — export `LabelIndex`, `Prediction`.
- **Modify** `.gitignore` — ignore `labels/*` (with a `.gitkeep` exception).
- **Create** `labels/.gitkeep` — placeholder.
- **Create** `tests/test_labels.py` — unit tests for `LabelIndex`.
- **Modify** `tests/test_index.py` — add a recursive-scan test.

---

## Task 1: Add `recursive` flag to `_scan` and `EmbeddingIndex.build_or_load`

**Files:**
- Modify: `dinoplay/index.py`
- Modify: `tests/test_index.py`

The current `_scan` walks only the top level of `images_dir`. We need optional one-level recursion so `LabelIndex` can scan `labels/<class>/<file>`. Default stays `False` so the existing `images/` index is unaffected.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_index.py` (at the end):

```python
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
    rels = sorted(p.replace(str(root) + "/", "") for p in idx.paths)
    assert rels == ["keyboard/c.png", "mug/a.png", "mug/b.png"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_index.py::test_recursive_scan_finds_one_level_deep -v`
Expected: FAIL — `build_or_load` doesn't accept `recursive` keyword.

- [ ] **Step 3: Update `_scan` and `build_or_load`**

In `dinoplay/index.py`, replace the existing `_scan` function with:

```python
def _scan(
    images_dir: Path,
    extensions: frozenset[str],
    recursive: bool = False,
) -> list[_Entry]:
    if not images_dir.exists() or not images_dir.is_dir():
        return []
    entries: list[_Entry] = []
    if recursive:
        # Exactly one level deep: <images_dir>/<subdir>/<file>.
        for subname in sorted(os.listdir(images_dir)):
            sub = images_dir / subname
            if not sub.is_dir():
                continue
            for name in sorted(os.listdir(sub)):
                path = sub / name
                if not path.is_file():
                    continue
                if path.suffix.lower() not in extensions:
                    continue
                st = path.stat()
                entries.append(_Entry(relpath=f"{subname}/{name}", mtime=st.st_mtime, size=st.st_size))
    else:
        for name in sorted(os.listdir(images_dir)):
            path = images_dir / name
            if not path.is_file():
                continue
            if path.suffix.lower() not in extensions:
                continue
            st = path.stat()
            entries.append(_Entry(relpath=name, mtime=st.st_mtime, size=st.st_size))
    return entries
```

Then in `EmbeddingIndex.build_or_load`, change the signature and the single call to `_scan`:

```python
@classmethod
def build_or_load(
    cls,
    images_dir: Path,
    cache_path: Path,
    encoder: _EncoderLike,
    model_id: str,
    extensions: frozenset[str],
    recursive: bool = False,
) -> "EmbeddingIndex":
    images_dir = Path(images_dir)
    cache_path = Path(cache_path)
    entries = _scan(images_dir, extensions, recursive=recursive)
    ...  # rest unchanged
```

(The rest of the method body stays the same — `_Entry.relpath` is already used opaquely, so a `<sub>/<file>` relpath flows through transparently. `Image.open(images_dir / entries[i].relpath)` still works because `Path` joins relative segments.)

- [ ] **Step 4: Run all index tests**

Run: `.venv/bin/pytest tests/test_index.py -v`
Expected: PASS, including the new recursive test and all existing non-recursive tests (regression check).

- [ ] **Step 5: Commit**

```bash
git add dinoplay/index.py tests/test_index.py
git commit -m "feat(index): optional one-level recursive scan"
```

---

## Task 2: `Prediction` dataclass and skeleton `LabelIndex`

**Files:**
- Create: `dinoplay/labels.py`
- Create: `tests/test_labels.py`

We start with the smallest unit: a `LabelIndex` that can be built from an empty `labels/` dir, exposes `is_empty`, `classes()`, `count()`, and `predict()` returning the empty-state `Prediction`. Persistence and mutation come in later tasks.

- [ ] **Step 1: Write the failing test**

Create `tests/test_labels.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_labels.py::test_empty_label_index -v`
Expected: FAIL — module `dinoplay.labels` does not exist.

- [ ] **Step 3: Create `dinoplay/labels.py`**

```python
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
        # Implementation comes in Task 3.
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_labels.py::test_empty_label_index -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dinoplay/labels.py tests/test_labels.py
git commit -m "feat(labels): scaffold LabelIndex with empty-state predict"
```

---

## Task 3: Implement `predict()` (k-NN majority vote)

**Files:**
- Modify: `dinoplay/labels.py`
- Modify: `tests/test_labels.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_labels.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_labels.py -v`
Expected: three new tests FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `predict`**

Replace the body of `LabelIndex.predict` in `dinoplay/labels.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_labels.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dinoplay/labels.py tests/test_labels.py
git commit -m "feat(labels): k-NN majority vote predict with alphabetical tie-break"
```

---

## Task 4: Implement `add()`

**Files:**
- Modify: `dinoplay/labels.py`
- Modify: `tests/test_labels.py`

`add(label, images)` writes each image to disk under `labels/<label>/<timestamp>-NNN.<ext>`, encodes them, appends to the in-memory arrays, and persists the cache. Filename collisions are avoided by adding a per-call counter and falling back to checking existence.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_labels.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_labels.py -v`
Expected: three new tests FAIL — `LabelIndex` has no `add` method.

- [ ] **Step 3: Implement `add`**

Add to `dinoplay/labels.py`:

At the top of the file, add imports:

```python
import time
```

Add a private helper at module level (above the class):

```python
def _next_filename(class_dir: Path, ext: str) -> Path:
    """Return a fresh path under class_dir with format YYYYMMDD-HHMMSS-NNN<ext>."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    n = 1
    while True:
        candidate = class_dir / f"{stamp}-{n:03d}{ext}"
        if not candidate.exists():
            return candidate
        n += 1
```

Add the `add` method to `LabelIndex`:

```python
def add(self, label: str, images: list[Image.Image]) -> None:
    if not images:
        return
    class_dir = self._labels_dir / label
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
```

(Always saving as `.png` keeps the on-disk format consistent and avoids JPEG re-encoding artifacts; the `extensions` filter still includes whatever the caller configured.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_labels.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dinoplay/labels.py tests/test_labels.py
git commit -m "feat(labels): add() saves image and grows index"
```

---

## Task 5: Implement `delete_class()`

**Files:**
- Modify: `dinoplay/labels.py`
- Modify: `tests/test_labels.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_labels.py`:

```python
def test_delete_class_removes_folder_and_entries(labels_root: Path, labels_cache: Path):
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
    assert sorted(idx.classes()) == ["keyboard", "mug"]

    idx.delete_class("mug")

    assert idx.classes() == ["keyboard"]
    assert idx.count("mug") == 0
    assert not (labels_root / "mug").exists()


def test_delete_class_idempotent_on_missing(labels_root: Path, labels_cache: Path):
    enc = StubEncoder(queue=[])
    idx = LabelIndex.build_or_load(
        labels_dir=labels_root,
        cache_path=labels_cache,
        encoder=enc,
        model_id=enc.model_id,
        extensions=frozenset({".png"}),
    )
    # Should not raise.
    idx.delete_class("nonexistent")
    assert idx.classes() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_labels.py -v`
Expected: two new tests FAIL — no `delete_class` method.

- [ ] **Step 3: Implement `delete_class`**

Add to `dinoplay/labels.py`:

At the top of the file, add:

```python
import shutil
```

Add the method to `LabelIndex`:

```python
def delete_class(self, label: str) -> None:
    class_dir = self._labels_dir / label
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_labels.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dinoplay/labels.py tests/test_labels.py
git commit -m "feat(labels): delete_class removes folder and refreshes index"
```

---

## Task 6: Class-name sanitization helper

**Files:**
- Modify: `dinoplay/labels.py`
- Modify: `tests/test_labels.py`

A standalone module-level function so the UI layer can call it before `add()` and surface the result to the user. Returns the sanitized name, or `None` if invalid.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_labels.py`:

```python
from dinoplay.labels import sanitize_class_name


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("mug", "mug"),
        ("  mug  ", "mug"),
        ("Mug", "mug"),
        ("hot dog", "hot_dog"),
        ("hot   dog", "hot_dog"),
        ("Coffee Mug 2", "coffee_mug_2"),
        ("a-b_c", "a-b_c"),
    ],
)
def test_sanitize_class_name_accepts(raw, expected):
    assert sanitize_class_name(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "My Mug!",
        "mug?",
        "mug/keyboard",
        "café",
    ],
)
def test_sanitize_class_name_rejects(raw):
    assert sanitize_class_name(raw) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_labels.py -v`
Expected: FAIL — `sanitize_class_name` not defined.

- [ ] **Step 3: Implement `sanitize_class_name`**

Add to `dinoplay/labels.py` (at module level, near the top):

```python
import re

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_labels.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dinoplay/labels.py tests/test_labels.py
git commit -m "feat(labels): sanitize_class_name helper"
```

---

## Task 7: Wire `LabelIndex` into config and startup

**Files:**
- Modify: `dinoplay/config.py`
- Modify: `dinoplay/cli.py`
- Modify: `dinoplay/__init__.py`
- Modify: `.gitignore`
- Create: `labels/.gitkeep`
- Modify: `tests/test_config.py`

Add `labels_dir` (default `Path("labels")`) and a `labels_cache_path` property to `Settings`. Build the `LabelIndex` at startup; pass to `build_app` (signature change comes in Task 8). Export the new types.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_settings_labels_paths(monkeypatch):
    from dinoplay.config import Settings
    s = Settings.from_env()
    # Defaults.
    assert str(s.labels_dir) == "labels"
    assert str(s.labels_cache_path).endswith("labels.npz")

    # Env override.
    monkeypatch.setenv("DINOPLAY_LABELS_DIR", "my_labels")
    s2 = Settings.from_env()
    assert str(s2.labels_dir) == "my_labels"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config.py::test_settings_labels_paths -v`
Expected: FAIL — `labels_dir` attribute does not exist.

- [ ] **Step 3: Update `dinoplay/config.py`**

Replace the file's contents with:

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    device: str = "auto"
    images_dir: Path = field(default_factory=lambda: Path("images"))
    labels_dir: Path = field(default_factory=lambda: Path("labels"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    batch_size: int = 16
    top_k: int = 12
    image_extensions: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / "embeddings.npz"

    @property
    def labels_cache_path(self) -> Path:
        return self.cache_dir / "labels.npz"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_id=os.environ.get("DINOPLAY_MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
            device=os.environ.get("DINOPLAY_DEVICE", "auto"),
            images_dir=Path(os.environ.get("DINOPLAY_IMAGE_DIR", "images")),
            labels_dir=Path(os.environ.get("DINOPLAY_LABELS_DIR", "labels")),
            cache_dir=Path(os.environ.get("DINOPLAY_CACHE_DIR", "cache")),
            batch_size=int(os.environ.get("DINOPLAY_BATCH_SIZE", "16")),
            top_k=int(os.environ.get("DINOPLAY_TOP_K", "12")),
        )
```

- [ ] **Step 4: Update `dinoplay/__init__.py`**

Replace contents with:

```python
from dinoplay.config import Settings
from dinoplay.index import EmbeddingIndex, SearchHit
from dinoplay.labels import LabelIndex, Prediction, sanitize_class_name
from dinoplay.model import DinoEncoder

__all__ = [
    "Settings",
    "DinoEncoder",
    "EmbeddingIndex",
    "SearchHit",
    "LabelIndex",
    "Prediction",
    "sanitize_class_name",
]
```

- [ ] **Step 5: Update `dinoplay/cli.py`**

Replace contents with:

```python
from __future__ import annotations

import logging

from dinoplay.app import build_app
from dinoplay.config import Settings
from dinoplay.index import EmbeddingIndex
from dinoplay.labels import LabelIndex
from dinoplay.model import DinoEncoder


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("dinoplay")

    settings = Settings.from_env()
    log.info("Loading encoder: %s", settings.model_id)
    encoder = DinoEncoder(
        settings.model_id,
        device=settings.device,
        batch_size=settings.batch_size,
    )
    log.info("Encoder ready on device: %s", encoder.device)

    log.info("Building/loading image index from: %s", settings.images_dir)
    index = EmbeddingIndex.build_or_load(
        images_dir=settings.images_dir,
        cache_path=settings.cache_path,
        encoder=encoder,
        model_id=settings.model_id,
        extensions=settings.image_extensions,
    )
    log.info("Image index ready (%d images).", len(index))

    log.info("Building/loading label index from: %s", settings.labels_dir)
    label_index = LabelIndex.build_or_load(
        labels_dir=settings.labels_dir,
        cache_path=settings.labels_cache_path,
        encoder=encoder,
        model_id=settings.model_id,
        extensions=settings.image_extensions,
    )
    log.info("Label index ready (%d labelled images, %d classes).", len(label_index._inner), len(label_index.classes()))

    app = build_app(settings, encoder, index, label_index)
    app.launch()
    return 0
```

- [ ] **Step 6: Update `.gitignore`**

Add at the end:

```
labels/*
!labels/.gitkeep
```

- [ ] **Step 7: Create `labels/.gitkeep`**

```bash
mkdir -p labels && touch labels/.gitkeep
```

- [ ] **Step 8: Run config tests**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS.

Note: the app will not yet start because `build_app` doesn't accept `label_index` — that's Task 8. **Do not run the app between Task 7 and Task 8.**

- [ ] **Step 9: Commit**

```bash
git add dinoplay/config.py dinoplay/__init__.py dinoplay/cli.py .gitignore labels/.gitkeep tests/test_config.py
git commit -m "feat(config): labels_dir setting and LabelIndex startup wiring"
```

---

## Task 8: Add `label_index` parameter to `build_app` (no UI yet)

**Files:**
- Modify: `dinoplay/app.py`

This is a tiny "plumbing" task to unbreak Task 7. The new parameter is accepted but unused — the new tabs come in Tasks 9 and 10.

- [ ] **Step 1: Update the `build_app` signature**

In `dinoplay/app.py`, change the imports and signature:

```python
from dinoplay.labels import LabelIndex
```

```python
def build_app(
    settings: Settings,
    encoder: DinoEncoder,
    index: EmbeddingIndex,
    label_index: LabelIndex,
) -> gr.Blocks:
```

Leave the function body unchanged otherwise.

- [ ] **Step 2: Smoke-run the existing tests**

Run: `.venv/bin/pytest -v`
Expected: PASS — nothing imports `build_app` from tests, but the package import still works.

- [ ] **Step 3: Smoke-run the app**

Run (from repo root): `.venv/bin/python -m dinoplay`
Expected: app starts, the existing Search/Inspect/Live tabs work as before. The startup log should mention "Label index ready (0 labelled images, 0 classes)." Stop with Ctrl-C.

- [ ] **Step 4: Commit**

```bash
git add dinoplay/app.py
git commit -m "chore(app): accept label_index in build_app (no UI yet)"
```

---

## Task 9: Build the Label Capture tab

**Files:**
- Modify: `dinoplay/app.py`

We add the tab end-to-end here (no per-step TDD because it's UI glue against a tested core). Manual smoke test at the end.

- [ ] **Step 1: Add the Label Capture tab**

In `dinoplay/app.py`, immediately after the existing `with gr.Tab("Live"):` block (which ends near `# interval_slider.change is unused now ...`), and **before** the closing `return app`, add:

```python
        with gr.Tab("Label Capture"):
            gr.Markdown(
                "Capture webcam photos and label them. One click = one photo saved under "
                "`labels/<class>/`. Use the **Label Live** tab to predict labels of new frames."
            )

            def _classes_summary(li: LabelIndex) -> str:
                names = li.classes()
                if not names:
                    return "_No classes yet._"
                return "  •  ".join(f"`{n}` ({li.count(n)})" for n in names)

            with gr.Row():
                with gr.Column(scale=1):
                    capture_cam = gr.Image(
                        type="pil",
                        label="Camera",
                        sources=["webcam"],
                        streaming=True,
                        webcam_options={"facingMode": {"exact": "environment"}},
                    )
                with gr.Column(scale=1):
                    class_input = gr.Textbox(
                        label="Class name",
                        placeholder="e.g. mug",
                    )
                    classes_md = gr.Markdown(_classes_summary(label_index))
                    capture_btn = gr.Button("Capture", variant="primary")
                    done_btn = gr.Button("Done with this class")
                    status = gr.Markdown("")
                with gr.Column(scale=1):
                    staging = gr.Gallery(
                        label="Captured this session",
                        columns=2,
                        height="auto",
                    )

            # Streaming callback just keeps the latest frame in a state.
            latest_frame = gr.State(None)

            def _stash_frame(img):
                if img is None:
                    return gr.skip()
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                return img.convert("RGB")

            capture_cam.stream(_stash_frame, inputs=capture_cam, outputs=latest_frame)

            def do_capture(raw_label: str, frame, current_gallery):
                from dinoplay.labels import sanitize_class_name

                label = sanitize_class_name(raw_label or "")
                if label is None:
                    return (
                        gr.skip(),
                        "**Status:** enter a valid class name (`[a-z0-9_-]`, spaces become `_`).",
                        gr.skip(),
                    )
                if frame is None:
                    return (
                        gr.skip(),
                        "**Status:** no camera frame yet — wait a moment, then try again.",
                        gr.skip(),
                    )
                label_index.add(label, [frame])
                gallery = list(current_gallery or [])
                gallery.append((frame, f"{label} #{label_index.count(label)}"))
                msg = f"**Status:** captured photo {label_index.count(label)} for `{label}`."
                if label != raw_label.strip():
                    msg += f" _(used `{label}`)_"
                return gallery, msg, _classes_summary(label_index)

            capture_btn.click(
                do_capture,
                inputs=[class_input, latest_frame, staging],
                outputs=[staging, status, classes_md],
            )

            done_btn.click(
                lambda: ("", [], "**Status:** ready for the next class."),
                outputs=[class_input, staging, status],
            )

            with gr.Accordion("Manage classes", open=False):
                manage_md = gr.Markdown(_classes_summary(label_index))
                delete_input = gr.Textbox(
                    label="Class to delete (exact name)",
                    placeholder="e.g. mug",
                )
                delete_btn = gr.Button("Delete class", variant="stop")
                delete_status = gr.Markdown("")

                def do_delete(name: str):
                    name = (name or "").strip()
                    if not name:
                        return "**Status:** enter a class name.", gr.skip(), gr.skip()
                    if label_index.count(name) == 0:
                        return (
                            f"**Status:** no class named `{name}`.",
                            gr.skip(),
                            gr.skip(),
                        )
                    label_index.delete_class(name)
                    summary = _classes_summary(label_index)
                    return (
                        f"**Status:** deleted class `{name}`.",
                        summary,
                        summary,
                    )

                delete_btn.click(
                    do_delete,
                    inputs=delete_input,
                    outputs=[delete_status, manage_md, classes_md],
                )
```

- [ ] **Step 2: Run unit tests**

Run: `.venv/bin/pytest -v`
Expected: PASS (UI changes don't affect unit tests).

- [ ] **Step 3: Manual smoke test**

Run: `.venv/bin/python -m dinoplay`

Verify in browser:
1. "Label Capture" tab appears after "Live".
2. Enter `mug` → click Capture → status shows `captured photo 1 for mug`. Gallery shows the photo. Classes summary updates to include `mug (1)`.
3. Capture another → photo count increments.
4. Click "Done with this class" → textbox and gallery clear.
5. Enter `Hot Dog!` → click Capture → status rejects ("`a-z0-9_-`...").
6. Enter `Hot Dog` (no `!`) → status shows `(used 'hot_dog')` and saves.
7. Open "Manage classes" → enter `mug` → click Delete → status shows deleted; folder gone from `labels/`.
8. Stop with Ctrl-C.

- [ ] **Step 4: Commit**

```bash
git add dinoplay/app.py
git commit -m "feat(app): Label Capture tab"
```

---

## Task 10: Build the Label Live tab

**Files:**
- Modify: `dinoplay/app.py`

- [ ] **Step 1: Add the Label Live tab**

In `dinoplay/app.py`, after the Label Capture `with gr.Tab(...)` block, add:

```python
        with gr.Tab("Label Live"):
            gr.Markdown(
                "Live label prediction. Point the camera at something — the predicted "
                "label and the labelled photos that voted for it will appear below."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    live_cam = gr.Image(
                        type="pil",
                        label="Camera",
                        sources=["webcam"],
                        streaming=True,
                        webcam_options={"facingMode": {"exact": "environment"}},
                    )
                    with gr.Row():
                        live_start = gr.Button("Start", variant="primary")
                        live_stop = gr.Button("Stop", variant="stop", visible=False)
                with gr.Column(scale=1):
                    label_md = gr.Markdown("### —")
                    confidence_md = gr.Markdown("_idle_")
                    k_slider = gr.Slider(
                        minimum=1, maximum=15, value=5, step=1, label="k (neighbors)"
                    )
                    threshold_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.6, step=0.05, label="confidence threshold"
                    )
                with gr.Column(scale=2):
                    matches_gallery = gr.Gallery(
                        label="Top-k matches",
                        columns=5,
                        height="auto",
                    )

            live_running = gr.State(False)

            def do_live_predict(frame, running, k, threshold):
                if not running:
                    return gr.skip(), gr.skip(), gr.skip()
                if label_index.is_empty:
                    return (
                        "### —",
                        "_No labelled classes yet — capture some in the Label Capture tab._",
                        [],
                    )
                if frame is None:
                    return gr.skip(), gr.skip(), gr.skip()
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(frame)
                try:
                    emb = encoder.encode([frame.convert("RGB")])[0]
                    pred = label_index.predict(emb, k=int(k))
                except Exception as exc:  # noqa: BLE001
                    logger.error("Label Live prediction failed: %s", exc)
                    return gr.skip(), gr.skip(), gr.skip()

                if pred.confidence >= float(threshold) and pred.label is not None:
                    label_text = f"### {pred.label}"
                else:
                    label_text = "### —"
                conf_text = f"confidence: {pred.confidence:.2f} ({int(round(pred.confidence * len(pred.hits)))}/{len(pred.hits)})"
                gallery = [
                    (h.path, f"{Path(h.path).parent.name} — {h.score:.3f}")
                    for h in pred.hits
                ]
                return label_text, conf_text, gallery

            live_cam.stream(
                do_live_predict,
                inputs=[live_cam, live_running, k_slider, threshold_slider],
                outputs=[label_md, confidence_md, matches_gallery],
            )

            live_start.click(
                lambda: (gr.update(visible=False), gr.update(visible=True), True),
                outputs=[live_start, live_stop, live_running],
            )
            live_stop.click(
                lambda: (gr.update(visible=True), gr.update(visible=False), False),
                outputs=[live_start, live_stop, live_running],
            )
```

Also at the top of `dinoplay/app.py` add (after the existing `import logging` line, before the `import gradio as gr` line):

```python
from pathlib import Path
```

- [ ] **Step 2: Run unit tests**

Run: `.venv/bin/pytest -v`
Expected: PASS.

- [ ] **Step 3: Manual smoke test**

Run: `.venv/bin/python -m dinoplay`

1. Open the Label Capture tab. Capture ≥3 photos for two distinct classes (e.g. `mug` and `keyboard`) — point the camera at clearly different things for each.
2. Switch to the Label Live tab. Click Start.
3. Point at one of the captured objects → predicted label appears within a couple of frames; top-k gallery shows photos from that class.
4. Point at something the camera hasn't seen → either a low-confidence prediction shown as `—` (good), or a wrong label with low confidence (also shown as `—` if below threshold).
5. Drop the threshold to 0 → labels always show. Raise it near 1 → labels rarely show.
6. Click Stop → streaming stops updating outputs.
7. Stop the app with Ctrl-C.

- [ ] **Step 4: Commit**

```bash
git add dinoplay/app.py
git commit -m "feat(app): Label Live tab with k-NN majority vote"
```

---

## Task 11: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

In `README.md`, in the opening paragraph that begins "Sandbox for experimenting...", replace the bulleted feature list with:

```markdown
- **Search** — drop an image, see the most visually similar images from your folder.
- **Inspect** — see the raw embedding (shape, L2 norm, first values) and the 5 nearest neighbors.
- **Live** — webcam similarity search against `images/`.
- **Label Capture** — webcam-capture photos and tag them with a class (e.g. `mug`). Saved to `labels/<class>/`.
- **Label Live** — webcam predicts a class label using k-NN majority vote over the labelled set.
```

In the "Configuration" table, add the new env var row alphabetically near the other dirs:

```markdown
| `DINOPLAY_LABELS_DIR` | `labels` | Folder for labelled photos. Subfolder per class. |
```

In the "Project layout" code block, add a `labels/` line:

```
labels/          # Labelled photos (gitignored), one subfolder per class.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README covers Label Capture and Label Live tabs"
```

---

## Task 12: Final verification

- [ ] **Step 1: Full test run**

Run: `.venv/bin/pytest -v`
Expected: all tests PASS.

- [ ] **Step 2: End-to-end smoke**

Run: `.venv/bin/python -m dinoplay`

Verify:
1. App starts; logs show both image and label index loaded.
2. All five tabs work: Search, Inspect, Live, Label Capture, Label Live.
3. Capture in Label Capture, then predict in Label Live, with the correct class winning when pointed at the captured object.
4. Stop with Ctrl-C.

- [ ] **Step 3: Confirm no committed labelled data**

Run: `git status`
Expected: clean. Run: `ls labels/` — should show `.gitkeep` plus any captured class folders. Run: `git ls-files labels/` — should show only `labels/.gitkeep`.

---

## Self-Review Notes

**Spec coverage check:**
- Data layout (subfolder per class, timestamped filenames, gitignore) → Task 7 (gitignore + .gitkeep), Task 4 (timestamped writes).
- Recursive `_scan` → Task 1.
- `Prediction` + `LabelIndex` API (`classes`, `count`, `add`, `delete_class`, `predict`, `is_empty`) → Tasks 2, 3, 4, 5.
- Tie-break alphabetical, k clamping, empty-state → Task 3.
- Class-name sanitization → Task 6.
- Capture tab UI (webcam, class input, helper text with counts, capture button, status, done button, staging gallery, manage panel) → Task 9.
- Label Live tab UI (webcam, label panel, top-k gallery, k slider, threshold slider, start/stop) → Task 10.
- App wiring (label index built at startup, two new tabs after existing Live) → Tasks 7, 8, 9, 10.
- Atomic in-memory updates → handled by `LabelIndex.add` / `delete_class` rebuilding `self._inner` via `EmbeddingIndex.build_or_load`, which constructs new arrays before assignment.
- Cache file uses temp-rename → inherited from existing `_save_cache`.
- Empty-state for Label Live (handled in streaming callback for both startup-empty and mid-session-empty) → Task 10.

**Type / name consistency:** `LabelIndex`, `Prediction`, `sanitize_class_name`, `_class_of`, `_next_filename` — used consistently across tasks. `label_index` is the standard parameter name in `build_app` and the closures inside it.

**Notes for the implementer:**
- All new tests use a `StubEncoder` defined inside `tests/test_labels.py` rather than the global `FakeEncoder` from `conftest.py`, because we need controllable embeddings for prediction tests. The two coexist fine.
- `LabelIndex.add` writes only PNG files even though the extensions filter is broader. This is deliberate — keeps capture deterministic and avoids JPEG re-encode artifacts. If the user has hand-placed JPEGs in `labels/<class>/`, they'll still be indexed (the extensions filter accepts them); only newly captured files are PNG.
- `_class_of(rel)` assumes `rel` always has a slash. Since `LabelIndex` only ever loads via `recursive=True`, this holds; we never need to handle a file at the top level of `labels/`.
- `len(label_index._inner)` in `cli.py` uses a single-underscore attribute. That's a deliberate small accessor break for a log line; if it bothers a reviewer, add a `__len__` to `LabelIndex` that returns `len(self._inner)`.

# Labels and Label Live mode

Add the ability to capture photos of a thing, label them with a class name, and run a Live mode that predicts the label of whatever the webcam is pointed at by matching against the labelled set.

## Goals

- Build up a small per-class set of labelled webcam photos.
- Predict the class of a live frame using k-NN majority vote over labelled embeddings.
- Keep the new feature isolated from the existing `images/` index and Search/Inspect/Live tabs.

## Non-goals

- Burst capture. Capture is one-shot per click.
- In-app editing or relabelling of existing photos. Delete and recapture instead.
- Per-class confidence thresholds.
- Persisting `k` or confidence-threshold settings across sessions.
- Recursing more than one level under `labels/`.

## Data layout

```
labels/
  mug/
    20260429-141203-001.jpg
    20260429-141204-002.jpg
  keyboard/
    ...
cache/
  embeddings.npz   # existing, over images/
  labels.npz       # new, over labels/<class>/*
```

- One subfolder per class. Class name is the subfolder name.
- Filenames are timestamped on capture (`YYYYMMDD-HHMMSS-NNN.jpg`) so they sort and don't collide.
- `labels/` is gitignored.
- Deleting a class removes its folder. Removing one example deletes that file.

## Index layer

### Change to `dinoplay/index.py`

Add a `recursive: bool = False` parameter to `_scan` and to `EmbeddingIndex.build_or_load`. When recursive, walk one level deep and store each entry's relpath as `<subdir>/<file>`. The non-recursive path stays unchanged so the existing `images/` index is untouched.

### New `dinoplay/labels.py`

`LabelIndex` wraps an `EmbeddingIndex` built recursively over `labels/`. Its responsibilities:

```python
@dataclass(frozen=True)
class Prediction:
    label: str | None       # majority class among top-k, None if index empty
    confidence: float       # votes_for_label / k, in [0, 1]
    hits: list[SearchHit]   # top-k nearest labelled images (with class derivable from path)

class LabelIndex:
    @classmethod
    def build_or_load(cls, labels_dir: Path, cache_path: Path, encoder, model_id: str, extensions) -> "LabelIndex": ...
    def classes(self) -> list[str]: ...                            # sorted unique class names
    def count(self, label: str) -> int: ...                        # number of examples for a class
    def add(self, label: str, images: list[Image.Image]) -> None:  # encode, write files, update arrays + cache atomically
    def delete_class(self, label: str) -> None:                    # remove folder + entries, persist cache
    def predict(self, query_emb: np.ndarray, k: int = 5) -> Prediction: ...
    @property
    def is_empty(self) -> bool: ...
```

Class derivation lives in `LabelIndex` (it parses `relpath.split("/")[0]`). `EmbeddingIndex` stays generic.

### Prediction algorithm

1. Compute cosine similarity of `query_emb` against the full labelled embedding matrix (same dot-product as `EmbeddingIndex.search`, since vectors are L2-normalized by the encoder).
2. Take top-k by score.
3. Among the top-k, count votes per class.
4. Pick the class with the most votes. **Ties broken alphabetically** (deterministic).
5. Confidence = `winning_votes / k`.
6. Return `Prediction(label, confidence, hits)` where `hits` are the top-k `SearchHit`s in score order.

If the index is empty: return `Prediction(None, 0.0, [])`.

If `k > len(index)`: clamp `k` to `len(index)` (and confidence is computed against the clamped k).

## Capture tab

Layout, left to right:

1. **Webcam input** — `gr.Image(sources=["webcam"], streaming=True, webcam_options={"facingMode": {"exact": "environment"}})`. Same config as the existing Live tab so back-camera selection stays consistent on phones.
2. **Controls column**:
   - `Textbox` for class name. Helper text below shows existing classes and their counts (e.g. `mug (12), keyboard (8)`).
   - "Capture" button (primary).
   - Status line: most recent action result (`captured photo 7 for mug` or `enter a class name first`).
   - "Done with this class" button — clears the textbox and the staging gallery.
3. **Staging gallery** — photos captured in this session for the current class, in capture order (most recent appended at the end). The gallery is session-only — it shows what was captured *this run*, not the full set on disk. Each capture appends here and writes immediately to disk + index (no separate Save step).

Below the main row:

- **Manage classes** panel: a table of `(class, count)` and a per-row delete button. Confirmation prompt on delete (a Gradio confirmation dialog or a two-click pattern — implementer's choice).

### Class name sanitization

- Trim whitespace, lowercase, replace internal whitespace runs with `_`.
- After sanitization, only `[a-z0-9_-]` is allowed.
- If sanitization changes the input, surface what was applied (e.g. `using "my_mug"`).
- Empty after sanitization → reject with status `enter a class name first`.

### Capture flow per click

1. Sanitize and validate the label.
2. Read the latest webcam frame from the streaming input. If no frame yet, status `no camera frame available yet`.
3. `LabelIndex.add(label, [frame])` — encodes, writes the file under `labels/<label>/<timestamp>-NNN.jpg`, updates the in-memory arrays, persists the cache.
4. Append the captured image to the staging gallery.
5. Update status: `captured photo <count> for <label>`.

## Label Live tab

Layout:

1. **Webcam streaming input** — same config as the existing Live tab.
2. **Predicted-label panel** — a `gr.Markdown` rendered as either:
   - `### mug` + `confidence: 0.80 (4/5)` when `confidence >= threshold`, or
   - `### —` + `confidence: 0.40 (2/5)` when below threshold.
3. **Top-k thumbnails gallery** — the actual labelled photos that contributed to the vote, captioned with `<class> — <score>`. Always shown (useful debugging signal even when label is gated out).
4. **Settings row**:
   - `k` slider — default 5, range 1–15, step 1.
   - confidence threshold slider — default 0.6, range 0–1, step 0.05.
   - Both stored as `gr.State` and read by the streaming callback each frame.
5. **Start / Stop buttons** — mirror the existing Live tab. Running flag in `gr.State`.

### Streaming callback per frame

1. If not running, or labelled index is empty → `gr.skip()` for all outputs.
2. Encode frame → `LabelIndex.predict(emb, k)`.
3. Render the predicted-label panel:
   - If `prediction.confidence >= threshold` → show label and confidence.
   - Else → show `—` and the (sub-threshold) confidence so the user can see how close it was.
4. Always update the top-k gallery from `prediction.hits`.

### Empty-state

The tab UI is always built (streaming input + label panel + gallery + sliders). The streaming callback is the single point that handles emptiness: when `LabelIndex.is_empty` is true, it short-circuits to render `### —` with the message `No labelled classes yet — capture some in the Label Capture tab.` and an empty gallery, regardless of `running`. This handles both "started with no labels" and "deleted all labels mid-session" with one code path.

## App wiring

In `dinoplay/app.py`:

- At startup, build the `LabelIndex` alongside the existing `EmbeddingIndex`, sharing the same encoder. Both are passed into `build_app(...)`.
- Add the two new tabs **after** the existing Live tab. Final tab order: Search, Inspect, Live, Label Capture, Label Live.
- The existing Search / Inspect / Live tabs and their behavior over `images/` are unchanged.

## Concurrency

Capture writes happen on user clicks (rare, sequential within a single user). The Label Live streaming callback reads the embedding array on every frame.

- `LabelIndex.add` and `delete_class` rebuild the internal arrays and swap the reference atomically (build new arrays → assign to `self._embeddings` and `self._paths` in a single statement). A streaming read therefore sees either the old snapshot or the new one, never a half-written array.
- Cache writes use the existing temp-file-then-`os.replace` pattern from `index._save_cache`.

## Error handling

- Empty / unsanitizable class name on capture → status hint, no save.
- No webcam frame yet → status hint, no save.
- `predict` on empty index → `Prediction(None, 0.0, [])`; Live tab renders the empty-state hint.
- Unreadable image file under `labels/` on startup → log warning and skip (consistent with `EmbeddingIndex` today).
- `delete_class` on a missing class → no-op (idempotent).

## Testing

New `tests/test_labels.py`:

- **predict, clean signal**: three fake classes with synthetic, well-separated embeddings. Query near class A → returns `label="a"`, `confidence=1.0`, hits all from class A.
- **predict, tied vote**: construct a query equidistant from two classes such that top-k splits 2/2/1; assert tie-break is alphabetical.
- **predict, empty index**: returns `Prediction(None, 0.0, [])`.
- **predict, k clamped**: with `len(index)=3` and `k=5`, returns 3 hits and confidence is computed over 3.
- **add round-trip**: `add("mug", [img])` writes a file under `labels/mug/`, the in-memory arrays grow by one, and re-loading via `build_or_load` returns the same data (file path, embedding bytes equal).
- **delete_class**: removes folder and entries; `classes()` no longer lists it; cache reflects the removal.
- **class-name sanitization**: `"My Mug!"` → reject; `"  mug  "` → accept as `mug`; `"hot dog"` → accept as `hot_dog`.
- **recursive `_scan`**: a file at `labels/<class>/<file>.jpg` is found; a file at `labels/<class>/<sub>/<file>.jpg` is **not** found (we only recurse one level).

UI interactions are tested by hand — Gradio behavior is out of scope for the unit suite.

## Out of scope

- Burst capture (rejected during brainstorming in favor of one-at-a-time).
- Combining the new Label Live mode into the existing Live tab.
- Editing or relabelling captured photos in-app.
- Per-class thresholds, learned thresholds, or model fine-tuning.
- Cross-tab notifications when capture changes the labelled set (the Label Live tab will simply pick up the new state on the next frame because the index reference is shared).

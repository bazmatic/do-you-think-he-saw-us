# dino-play — Design

A small Python sandbox for experimenting with Meta's DINO vision foundation models on a folder of personal images. Two capabilities: feature extraction and similarity search, exposed through a Gradio UI.

## Goals

1. **Feature extraction** — produce a fixed-dimensional embedding for any image using a DINO model, with the embedding inspectable in the UI.
2. **Image similarity search** — given a query image, return the visually most similar images from a local folder, ranked by cosine similarity.

DINOv3 weights are gated on Hugging Face. The user has a HF account but has not yet been approved for `facebook/dinov3-*`. To stay unblocked, we start with **DINOv2** (open weights, same `transformers` API). The model id is a config value; switching to DINOv3 once access lands is a one-line change.

## Non-Goals

- No FAISS / vector DB. In-memory numpy with brute-force cosine search is sufficient for hundreds of images.
- No clustering, PCA visualization, or attention-map rendering.
- No multi-folder support, recursive folder watching, auth, or Docker packaging.
- No CLI search interface; UI is Gradio only.
- Not a production service; this is a sandbox.

## Constraints

- **Runtime:** macOS (Apple Silicon). Default device is MPS, with automatic fallback to CPU if MPS is unavailable. No CUDA assumptions.
- **Scale:** designed for ~10–500 images. Brute-force similarity over a `(N, D)` numpy array is the right default at this scale.
- **Python:** 3.11.
- **Model size:** default `facebook/dinov2-base` (ViT-B/14, ~350 MB). Configurable.
- **Network:** first run downloads the model from Hugging Face; afterward fully offline.

## Architecture

Three focused modules plus a thin entrypoint. Each module has one purpose and a narrow interface.

```
dino-play/
├── pyproject.toml
├── README.md
├── .python-version           # 3.11
├── .gitignore                # ignores images/, cache/, .venv/
├── images/                   # user drops images here
├── cache/                    # embedding cache lives here
├── dinoplay/
│   ├── __init__.py
│   ├── config.py             # Settings dataclass; env-var overrides
│   ├── model.py              # DinoEncoder
│   ├── index.py              # EmbeddingIndex
│   └── app.py                # Gradio UI wiring
├── scripts/
│   └── run.py                # entrypoint
└── tests/
    ├── conftest.py
    └── test_index.py
```

### Module responsibilities

- **`config.py`** — `Settings` dataclass with: `model_id` (default `facebook/dinov2-base`), `device` (`"mps" | "cpu" | "auto"`, default `"auto"`), `images_dir` (default `./images`), `cache_dir` (default `./cache`), `batch_size` (default 16), `top_k` (default 12), `image_extensions` (default `{".jpg", ".jpeg", ".png", ".webp"}`). Each field overridable via env var (`DINOPLAY_MODEL`, `DINOPLAY_DEVICE`, `DINOPLAY_IMAGE_DIR`, `DINOPLAY_CACHE_DIR`, `DINOPLAY_BATCH_SIZE`, `DINOPLAY_TOP_K`). Provides a derived `cache_path` property = `cache_dir / "embeddings.npz"`.

- **`model.py`** — `DinoEncoder` class.
  - `__init__(model_id: str, device: str)`: loads `AutoImageProcessor` and `AutoModel` from Hugging Face, moves model to device, sets eval mode.
  - `encode(images: list[PIL.Image.Image]) -> np.ndarray`: batches images through the processor and model, takes the CLS token from `last_hidden_state`, L2-normalizes, returns float32 array of shape `(N, D)`. Internal batching uses `Settings.batch_size`.
  - Knows nothing about disk, files, or Gradio. Pure: PIL in, numpy out.

- **`index.py`** — `EmbeddingIndex` class.
  - `build_or_load(images_dir, cache_path, encoder, model_id) -> EmbeddingIndex`: scans the folder, loads cache if present, encodes only new/changed files, saves cache, returns the index.
  - `search(query_emb: np.ndarray, k: int) -> list[SearchHit]`: dot-product against the embedding matrix (vectors are pre-normalized, so dot product == cosine), returns top-k as `SearchHit(path, score)` tuples.
  - `__len__`, `is_empty` for UI to render friendly states.
  - Knows numpy and the filesystem; nothing about Torch or Gradio.

- **`app.py`** — `build_app(settings) -> gr.Blocks`. Two tabs ("Search", "Inspect") wired to `DinoEncoder` and `EmbeddingIndex`. No model or filesystem logic.

- **`scripts/run.py`** — loads settings, builds the index (with a console progress bar via `tqdm`), launches Gradio. Also installable as `python -m dinoplay` via `__main__.py`.

### Data flow

**Startup:**
1. Load `Settings` (defaults + env overrides).
2. Construct `DinoEncoder(settings.model_id, settings.device)`. First run pulls weights from HF.
3. `EmbeddingIndex.build_or_load(settings.images_dir, settings.cache_path, encoder, settings.model_id)`:
   - Walk `images_dir` for supported extensions.
   - Compute key `(relpath, mtime, size)` for each.
   - Load `cache/embeddings.npz` if present; if its `model_id` field doesn't match the current model, discard the cache.
   - Reuse cached embeddings whose key matches; encode the rest in batches.
   - Atomically save the updated cache (write `.tmp`, then rename).
4. Launch Gradio at `127.0.0.1:7860`.

**Search query:**
1. User drops an image into the Search tab.
2. `encoder.encode([img])` → `(1, D)` query.
3. `index.search(query, k=settings.top_k)` → ranked list of paths + scores.
4. UI renders a `gr.Gallery` of top-k thumbnails, captioned `"{filename} — {score:.3f}"`.

**Inspect query:**
1. User drops an image into the Inspect tab.
2. `encoder.encode([img])` → `(1, D)` embedding.
3. UI shows: shape, L2 norm (sanity check, should be ~1.0), first 16 values, and a small gallery of top-5 nearest neighbors from the index.

## Cache format

Single file `cache/embeddings.npz` produced by `numpy.savez`:

| Key          | Type            | Description                                          |
|--------------|-----------------|------------------------------------------------------|
| `paths`      | array of str    | Image paths relative to `images_dir`.                |
| `mtimes`     | float64 array   | `os.stat().st_mtime` per file.                       |
| `sizes`      | int64 array     | `os.stat().st_size` per file.                        |
| `embeddings` | float32 array   | Shape `(N, D)`, L2-normalized.                       |
| `model_id`   | scalar str      | Identifies which model produced the embeddings.      |

A cache entry is reused only if `(relpath, mtime, size)` matches the current scan AND `model_id` matches current settings. Mismatch on `model_id` invalidates the entire cache (re-encode all). This makes model switching safe by construction.

## Error handling

Validate at boundaries; trust internal calls.

- **`images_dir` missing or empty** → `EmbeddingIndex.is_empty` is `True`; the UI shows: *"Drop images into `./images/` and restart the app."* No crash.
- **Unreadable image file** during build → log a warning with the path, skip it, continue. One corrupt JPEG must not abort the build.
- **MPS unavailable** with `device="auto"` → fall back to CPU; log one line at startup: *"MPS not available, using CPU."*
- **Model download fails** (network down, gated model without token) → catch the exception at `DinoEncoder.__init__`, raise a `RuntimeError` with a message pointing to the README section on Hugging Face authentication. The README must include a "Hugging Face authentication" section covering `huggingface-cli login` and the `HF_TOKEN` env var, so this error message has somewhere to point to.
- **Cache file corrupt** (NPZ load fails) → log a warning, delete the file, rebuild from scratch.

## Testing

Pragmatic, fast-by-default.

- **`tests/test_index.py`** (fast, no Torch):
  - Uses a `FakeEncoder` that returns deterministic L2-normalized vectors keyed by file content hash.
  - Builds an index over 4 fixture images, asserts cache file exists and has the right keys.
  - Touches one file's mtime, rebuilds, asserts only that one file was re-encoded (FakeEncoder counts calls).
  - Switches `model_id`, rebuilds, asserts all four are re-encoded (cache invalidation).
  - Searches with a known query, asserts ranking is stable.

- **`tests/test_model.py`** (slow, marked `@pytest.mark.slow`, opt-in via `pytest -m slow`):
  - Loads the real `facebook/dinov2-base`, encodes one tiny image, asserts output shape `(1, 768)` and L2 norm ≈ 1.0. Sanity check only.

- **No automated UI tests.** Gradio is exercised manually.

`pyproject.toml` configures pytest with `addopts = "-m 'not slow'"` so the default `pytest` run skips the slow test.

## Dependencies

Managed with `uv` (fast, modern; `uv pip install -e .` works for fallback).

| Package         | Purpose                                              |
|-----------------|------------------------------------------------------|
| `torch`         | Model runtime (with MPS support on Apple Silicon).   |
| `transformers`  | `AutoModel` / `AutoImageProcessor` for DINOv2 / v3.  |
| `pillow`        | Image loading.                                       |
| `numpy`         | Embedding storage and similarity math.               |
| `gradio`        | UI.                                                  |
| `safetensors`   | Faster, safer model weight loading via transformers. |
| `tqdm`          | Progress bar during index build.                     |
| `pytest`        | Test runner (dev dep).                               |

## Migrating to DINOv3

Once HF approves access:
1. `huggingface-cli login` with the user's token.
2. Set `DINOPLAY_MODEL=facebook/dinov3-vitb16` (or the chosen v3 variant) in the environment, or update `Settings.model_id` default.
3. Restart the app. Cache invalidates automatically (model_id check); embeddings rebuild.

No code changes required.

## Open questions / future work

- Add a "compare two images" tab that prints just the cosine score (useful for quick A/B tests).
- Add UMAP/PCA scatter of all embeddings as an exploration view.
- Render attention maps from the last block — useful and visually striking, but a meaningful chunk of work; deliberately out of scope here.
- Optional CLI (`python -m dinoplay.search --query path.jpg`) if interactive use ever feels limiting.

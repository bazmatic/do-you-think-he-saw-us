# dino-play

Sandbox for experimenting with Meta's DINO vision foundation models on a folder of your own images. Two things in one Gradio app:

- **Search** — drop an image, see the most visually similar images from your folder.
- **Inspect** — see the raw embedding (shape, L2 norm, first values) and the 5 nearest neighbors.

Defaults to **DINOv2** (open weights). Switch to DINOv3 once you have Hugging Face access (see below).

## Requirements

- macOS (Apple Silicon recommended; CPU works, just slower)
- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) (recommended) or plain `pip`

## Setup

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

(With `pip`: `python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.)

## Run

1. Drop some images (`.jpg`, `.jpeg`, `.png`, `.webp`) into `./images/`.
2. Launch the app:

```bash
.venv/bin/python scripts/run.py
# or:
.venv/bin/python -m dinoplay
```

3. Open the URL it prints (`http://127.0.0.1:7860`).

The first run downloads the model weights (~350 MB for `dinov2-base`) and builds an embedding cache in `./cache/embeddings.npz`. Subsequent runs reuse the cache and only re-encode files that changed.

## Configuration

All settings can be overridden via env vars:

| Variable | Default | Description |
|---|---|---|
| `DINOPLAY_MODEL` | `facebook/dinov2-base` | Hugging Face model id. |
| `DINOPLAY_DEVICE` | `auto` | `auto` (MPS if available else CPU), `mps`, or `cpu`. |
| `DINOPLAY_IMAGE_DIR` | `images` | Folder to index. |
| `DINOPLAY_CACHE_DIR` | `cache` | Where `embeddings.npz` lives. |
| `DINOPLAY_BATCH_SIZE` | `16` | Encode batch size. |
| `DINOPLAY_TOP_K` | `12` | Results in the Search tab. |

Example:

```bash
DINOPLAY_DEVICE=cpu DINOPLAY_TOP_K=20 .venv/bin/python scripts/run.py
```

### Available models

DINOv2 (open weights, no auth needed):
- `facebook/dinov2-small` — 86 MB, 384-dim, fastest
- `facebook/dinov2-base` — 350 MB, 768-dim **(default)**
- `facebook/dinov2-large` — 1.2 GB, 1024-dim
- `facebook/dinov2-giant` — 4.4 GB, 1536-dim

DINOv3 (gated; see auth section below):
- `facebook/dinov3-vits16-pretrain-lvd1689m` — small ViT, ~22M params
- `facebook/dinov3-vitb16-pretrain-lvd1689m` — base ViT, ~85M params (closest to `dinov2-base`)
- `facebook/dinov3-vitl16-pretrain-lvd1689m` — large ViT, ~300M params

## Hugging Face authentication (required for DINOv3)

DINOv3 weights (`facebook/dinov3-*`) are gated. To use them:

1. Create or sign in to your Hugging Face account.
2. Visit the model card and request access (button reads "Agree and access repository"):
   - <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>
   - Approval is typically fast for non-commercial use.
3. Once approved, log in locally:
   ```bash
   .venv/bin/huggingface-cli login
   ```
   Or set `HF_TOKEN` in your shell.
4. Switch model:
   ```bash
   DINOPLAY_MODEL=facebook/dinov3-vitb16-pretrain-lvd1689m .venv/bin/python scripts/run.py
   ```

The cache invalidates automatically when the model id changes; embeddings are rebuilt on first run with the new model.

## Tests

```bash
.venv/bin/pytest          # fast tests only
.venv/bin/pytest -m slow  # slow test that loads the real model (downloads weights)
```

## Troubleshooting

- **"MPS not available, using CPU."** — Normal on non-Apple-Silicon machines. CPU works fine, just slower.
- **First launch is slow.** — Building embeddings from scratch. Subsequent launches reuse the cache.
- **`OSError: You are trying to access a gated repo.`** — DINOv3 access not yet approved, or you haven't logged in. See "Hugging Face authentication".
- **`RuntimeError: Failed to load model ...`** — Either no network, or the model is gated and you haven't authenticated. See above.
- **Old embeddings in the UI after editing files.** — The cache uses `(path, mtime, size)`. Save the file properly (changing mtime). If a file's content changed but mtime didn't, `touch <file>` will force a re-encode.

## Project layout

```
dinoplay/        # Library code (config, model, index, app, cli).
scripts/run.py   # Launches the Gradio UI.
images/          # Your images go here (gitignored).
cache/           # Embedding cache (gitignored).
tests/           # pytest suite.
docs/superpowers # Spec + plan documents.
```

# dino-play

A Gradio app for exploring Meta's DINO vision models against a folder of your own images. Five tabs:

- **Search** — drop an image, find the most visually similar images in your collection
- **Inspect** — view raw embeddings and nearest neighbours for any image
- **Live** — real-time webcam similarity search
- **Label Capture** — capture and tag webcam photos by class (saved to `labels/<class>/`)
- **Label Live** — webcam predicts a class label using k-NN majority vote over your labelled set

## Requirements

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) (recommended)
- macOS with Apple Silicon recommended (CPU works, just slower)

## Setup

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

## Run

1. Drop images (`.jpg`, `.png`, `.webp`) into `./images/`
2. Start the app:

```bash
./run.sh
# or:
.venv/bin/python scripts/run.py
```

3. Open the URL printed in the terminal (default: `http://127.0.0.1:7860`)

First run downloads model weights (~350 MB) and builds an embedding cache. Subsequent runs reuse the cache.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DINOPLAY_MODEL` | `facebook/dinov2-base` | Hugging Face model ID |
| `DINOPLAY_DEVICE` | `auto` | `auto`, `mps`, or `cpu` |
| `DINOPLAY_IMAGE_DIR` | `images` | Folder to index |
| `DINOPLAY_LABELS_DIR` | `labels` | Folder for labelled photos |
| `DINOPLAY_TOP_K` | `12` | Number of results in Search tab |

Available models: `facebook/dinov2-small`, `facebook/dinov2-base`, `facebook/dinov2-large`, `facebook/dinov2-giant`

## Tests

```bash
.venv/bin/pytest
```

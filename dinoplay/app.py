from __future__ import annotations

import logging

import gradio as gr
import numpy as np
from PIL import Image

from dinoplay.config import Settings
from dinoplay.index import EmbeddingIndex
from dinoplay.labels import LabelIndex
from dinoplay.model import DinoEncoder

logger = logging.getLogger(__name__)


def _format_embedding_summary(emb: np.ndarray) -> str:
    norm = float(np.linalg.norm(emb))
    head = ", ".join(f"{x:+.4f}" for x in emb[:16])
    return (
        f"shape: {emb.shape}\n"
        f"dtype: {emb.dtype}\n"
        f"L2 norm: {norm:.6f}  (should be ~1.0)\n"
        f"first 16 values: [{head}]"
    )


def _empty_message(settings: Settings) -> str:
    return (
        f"No images found in `{settings.images_dir}`.\n"
        "Drop some `.jpg / .jpeg / .png / .webp` files in there and restart the app."
    )


def build_app(
    settings: Settings,
    encoder: DinoEncoder,
    index: EmbeddingIndex,
    label_index: LabelIndex,
) -> gr.Blocks:
    with gr.Blocks(title="dino-play") as app:
        gr.Markdown(
            f"# dino-play\n"
            f"**Model:** `{settings.model_id}`  •  "
            f"**Device:** `{encoder.device}`  •  "
            f"**Indexed images:** {len(index)}"
        )

        if index.is_empty:
            gr.Markdown(f"⚠️ {_empty_message(settings)}")
            return app

        with gr.Tab("Search"):
            gr.Markdown(
                f"Drop a query image; see the top {settings.top_k} most similar from your folder."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    search_input = gr.Image(
                        type="pil",
                        label="Query image",
                        sources=["upload", "clipboard"],
                    )
                    search_btn = gr.Button("Search", variant="primary")
                with gr.Column(scale=2):
                    search_gallery = gr.Gallery(
                        label="Top matches",
                        columns=4,
                        height="auto",
                        show_label=True,
                    )

            def do_search(img: Image.Image | None) -> list[tuple[str, str]]:
                if img is None:
                    return []
                q = encoder.encode([img.convert("RGB")])[0]
                hits = index.search(q, k=settings.top_k)
                return [(h.path, f"{h.path.split('/')[-1]} — {h.score:.3f}") for h in hits]

            search_btn.click(do_search, inputs=search_input, outputs=search_gallery)

        with gr.Tab("Inspect"):
            gr.Markdown(
                "Drop an image to see its embedding stats and 5 nearest neighbors from your folder."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    inspect_input = gr.Image(
                        type="pil",
                        label="Image",
                        sources=["upload", "clipboard"],
                    )
                    inspect_btn = gr.Button("Inspect", variant="primary")
                    inspect_text = gr.Textbox(label="Embedding", lines=8, interactive=False)
                with gr.Column(scale=2):
                    inspect_gallery = gr.Gallery(
                        label="Nearest neighbors", columns=5, height="auto"
                    )

            def do_inspect(img: Image.Image | None) -> tuple[str, list[tuple[str, str]]]:
                if img is None:
                    return "", []
                emb = encoder.encode([img.convert("RGB")])[0]
                hits = index.search(emb, k=5)
                neighbors = [(h.path, f"{h.path.split('/')[-1]} — {h.score:.3f}") for h in hits]
                return _format_embedding_summary(emb), neighbors

            inspect_btn.click(
                do_inspect, inputs=inspect_input, outputs=[inspect_text, inspect_gallery]
            )

        with gr.Tab("Live"):
            gr.Markdown(
                "Live similarity search using webcam. Start the feed to automatically poll your camera and find nearby matches."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    live_input = gr.Image(
                        type="pil",
                        label="Live Camera",
                        sources=["webcam"],
                        streaming=True,
                        webcam_options={"facingMode": {"exact": "environment"}},
                    )
                    with gr.Row():
                        start_btn = gr.Button("Start Live Search", variant="primary")
                        stop_btn = gr.Button("Stop", variant="stop", visible=False)
                with gr.Column(scale=1):
                    last_query_image = gr.Image(
                        type="pil",
                        label="Last Query",
                        interactive=False,
                    )
                with gr.Column(scale=2):
                    live_gallery = gr.Gallery(
                        label="Top match",
                        columns=1,
                        height="auto",
                        show_label=True,
                    )

            is_running = gr.State(False)
            is_running = gr.State(False)

            def do_live_search(
                img: Image.Image | None, 
                active: bool
            ):
                if img is None or not active:
                    return gr.skip(), gr.skip()
                    
                print("Sampling camera frame sequentially...", flush=True)
                try:
                    import numpy as np
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img)
                    q = encoder.encode([img.convert("RGB")])[0]
                    hits = index.search(q, k=1)
                    matches = [(h.path, f"{h.path.split('/')[-1]} — {h.score:.3f}") for h in hits]
                    print(f"Captured! Found {len(matches)} matches.", flush=True)
                    return img, matches
                except Exception as e:
                    print(f"Error during live search: {e}", flush=True)
                    logger.error(f"Error during live search: {e}")
                    return gr.skip(), gr.skip()

            live_input.stream(
                do_live_search,
                inputs=[live_input, is_running],
                outputs=[last_query_image, live_gallery]
            )

            start_btn.click(
                lambda: (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    True,
                ),
                outputs=[start_btn, stop_btn, is_running],
            )

            stop_btn.click(
                lambda: (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    False,
                ),
                outputs=[start_btn, stop_btn, is_running],
            )

            # interval_slider.change is unused now, because interval is dynamically pulled!

    return app

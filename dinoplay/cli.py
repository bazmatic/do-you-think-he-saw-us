from __future__ import annotations

import logging

from dinoplay.app import build_app
from dinoplay.config import Settings
from dinoplay.index import EmbeddingIndex
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

    log.info("Building/loading index from: %s", settings.images_dir)
    index = EmbeddingIndex.build_or_load(
        images_dir=settings.images_dir,
        cache_path=settings.cache_path,
        encoder=encoder,
        model_id=settings.model_id,
        extensions=settings.image_extensions,
    )
    log.info("Index ready (%d images).", len(index))

    app = build_app(settings, encoder, index)
    app.launch()
    return 0

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def resolve_device(preference: str) -> str:
    if preference == "auto":
        if _mps_available():
            return "mps"
        logger.info("MPS not available, using CPU.")
        return "cpu"
    return preference


def _batched(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class DinoEncoder:
    def __init__(self, model_id: str, device: str = "auto", batch_size: int = 16) -> None:
        self.model_id = model_id
        self.device = resolve_device(device)
        self.batch_size = batch_size
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{model_id}'. "
                "If this is a gated model (e.g. DINOv3), see the 'Hugging Face authentication' "
                "section in README.md for setup steps."
            ) from exc

    @torch.no_grad()
    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        if len(images) == 0:
            hidden = self.model.config.hidden_size
            return np.zeros((0, hidden), dtype=np.float32)

        chunks: list[np.ndarray] = []
        for batch in _batched(list(images), self.batch_size):
            inputs = self.processor(images=list(batch), return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            chunks.append(cls.detach().to("cpu", dtype=torch.float32).numpy())
        return np.concatenate(chunks, axis=0)

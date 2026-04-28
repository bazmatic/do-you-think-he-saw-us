from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    device: str = "auto"
    images_dir: Path = field(default_factory=lambda: Path("images"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    batch_size: int = 16
    top_k: int = 12
    image_extensions: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / "embeddings.npz"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_id=os.environ.get("DINOPLAY_MODEL", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
            device=os.environ.get("DINOPLAY_DEVICE", "auto"),
            images_dir=Path(os.environ.get("DINOPLAY_IMAGE_DIR", "images")),
            cache_dir=Path(os.environ.get("DINOPLAY_CACHE_DIR", "cache")),
            batch_size=int(os.environ.get("DINOPLAY_BATCH_SIZE", "16")),
            top_k=int(os.environ.get("DINOPLAY_TOP_K", "12")),
        )

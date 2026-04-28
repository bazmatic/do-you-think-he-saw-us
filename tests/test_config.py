from pathlib import Path

from dinoplay.config import Settings


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for var in [
        "DINOPLAY_MODEL",
        "DINOPLAY_DEVICE",
        "DINOPLAY_IMAGE_DIR",
        "DINOPLAY_CACHE_DIR",
        "DINOPLAY_BATCH_SIZE",
        "DINOPLAY_TOP_K",
    ]:
        monkeypatch.delenv(var, raising=False)

    s = Settings.from_env()

    assert s.model_id == "facebook/dinov2-base"
    assert s.device == "auto"
    assert s.images_dir == Path("images")
    assert s.cache_dir == Path("cache")
    assert s.batch_size == 16
    assert s.top_k == 12
    assert s.image_extensions == frozenset({".jpg", ".jpeg", ".png", ".webp"})
    assert s.cache_path == Path("cache") / "embeddings.npz"


def test_settings_env_overrides(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DINOPLAY_MODEL", "facebook/dinov3-vitb16")
    monkeypatch.setenv("DINOPLAY_DEVICE", "cpu")
    monkeypatch.setenv("DINOPLAY_IMAGE_DIR", "/tmp/imgs")
    monkeypatch.setenv("DINOPLAY_CACHE_DIR", "/tmp/cache")
    monkeypatch.setenv("DINOPLAY_BATCH_SIZE", "4")
    monkeypatch.setenv("DINOPLAY_TOP_K", "5")

    s = Settings.from_env()

    assert s.model_id == "facebook/dinov3-vitb16"
    assert s.device == "cpu"
    assert s.images_dir == Path("/tmp/imgs")
    assert s.cache_dir == Path("/tmp/cache")
    assert s.batch_size == 4
    assert s.top_k == 5
    assert s.cache_path == Path("/tmp/cache") / "embeddings.npz"

from unittest.mock import patch

import pytest

from dinoplay.model import resolve_device


def test_resolve_device_explicit_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_explicit_mps():
    assert resolve_device("mps") == "mps"


def test_resolve_device_auto_picks_mps_when_available():
    with patch("dinoplay.model._mps_available", return_value=True):
        assert resolve_device("auto") == "mps"


def test_resolve_device_auto_falls_back_to_cpu():
    with patch("dinoplay.model._mps_available", return_value=False):
        assert resolve_device("auto") == "cpu"


@pytest.mark.slow
def test_real_encoder_shape_and_norm():
    """Loads the actual DINOv2-base model. Slow; opt in with `pytest -m slow`."""
    import numpy as np
    from PIL import Image

    from dinoplay.model import DinoEncoder

    encoder = DinoEncoder("facebook/dinov2-base", device="cpu")
    img = Image.new("RGB", (224, 224), color=(128, 64, 200))
    out = encoder.encode([img])

    assert out.shape == (1, 768)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

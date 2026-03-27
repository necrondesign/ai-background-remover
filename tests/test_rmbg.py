"""
Unit tests for AI Background Remover.

Run with:
    python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestDeviceDetection:
    """FR-2.5 / FR-2.6: Device detection for MPS or CPU."""

    def test_get_device_returns_torch_device(self):
        from rmbg_app import get_device
        import torch

        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("mps", "cuda", "cpu")

    def test_get_device_cpu_fallback(self):
        """When MPS and CUDA are unavailable, should fallback to CPU."""
        import torch
        from rmbg_app import get_device

        with patch.object(torch.backends.mps, "is_available", return_value=False), \
             patch.object(torch.cuda, "is_available", return_value=False):
            device = get_device()
            assert device.type == "cpu"


class TestModelPath:
    """FR-2.1: Model loading path resolution."""

    def test_dev_mode_returns_hub_id(self):
        from rmbg_app import get_model_path, MODEL_ID

        # In dev mode (not frozen), should return HuggingFace model ID
        if not getattr(sys, "frozen", False):
            assert get_model_path() == MODEL_ID


class TestCheckerboard:
    """FR-4.2: Checkerboard transparency pattern."""

    def test_checkerboard_dimensions(self):
        from rmbg_app import build_checkerboard

        img = build_checkerboard(100, 80)
        assert img.size == (100, 80)
        assert img.mode == "RGB"

    def test_checkerboard_has_two_colors(self):
        from rmbg_app import build_checkerboard

        img = build_checkerboard(20, 20)
        pixels = set(img.getdata())
        # Should contain both light and dark squares
        assert len(pixels) >= 2

    def test_checkerboard_square_pattern(self):
        from rmbg_app import build_checkerboard, CHECKERBOARD_SIZE

        img = build_checkerboard(40, 40)
        arr = np.array(img)
        # Top-left pixel should be dark (first square)
        top_left = tuple(arr[0, 0])
        # Pixel at (CHECKERBOARD_SIZE, 0) should be different
        next_square = tuple(arr[0, CHECKERBOARD_SIZE])
        assert top_left != next_square


class TestThumbnail:
    """FR-1.4 / FR-1.5: Thumbnail generation preserving aspect ratio."""

    def test_fit_thumbnail_landscape(self):
        from rmbg_app import fit_thumbnail, PREVIEW_MAX

        img = Image.new("RGB", (800, 400))
        result = fit_thumbnail(img)
        assert result.size[0] <= PREVIEW_MAX
        assert result.size[1] <= PREVIEW_MAX
        # Landscape: width should be at max
        assert result.size[0] == PREVIEW_MAX
        # Aspect ratio preserved
        assert abs(result.size[0] / result.size[1] - 2.0) < 0.1

    def test_fit_thumbnail_portrait(self):
        from rmbg_app import fit_thumbnail, PREVIEW_MAX

        img = Image.new("RGB", (300, 600))
        result = fit_thumbnail(img)
        assert result.size[0] <= PREVIEW_MAX
        assert result.size[1] <= PREVIEW_MAX
        assert result.size[1] == PREVIEW_MAX

    def test_fit_thumbnail_small_image(self):
        from rmbg_app import fit_thumbnail, PREVIEW_MAX

        img = Image.new("RGB", (100, 100))
        result = fit_thumbnail(img)
        # Small images should not be upscaled
        assert result.size[0] <= 100
        assert result.size[1] <= 100


# ---------------------------------------------------------------------------
# Image preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    """FR-2.3: Image preprocessing pipeline."""

    def test_preprocess_output_shape(self):
        from rmbg_app import _preprocess, INPUT_RESOLUTION

        img = Image.new("RGB", (640, 480))
        tensor = _preprocess(img)
        assert tensor.shape == (3, INPUT_RESOLUTION[0], INPUT_RESOLUTION[1])

    def test_preprocess_normalized_range(self):
        from rmbg_app import _preprocess

        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        tensor = _preprocess(img)
        # After ImageNet normalization, values should be roughly centered around 0
        assert tensor.min() > -5.0
        assert tensor.max() < 5.0


# ---------------------------------------------------------------------------
# Threshold application tests
# ---------------------------------------------------------------------------

class TestThreshold:
    """FR-3: Threshold control logic."""

    def test_threshold_zero_all_transparent(self):
        """FR-3.5: threshold=0 → only pixels with matte > 0 are visible."""
        from rmbg_app import apply_threshold

        original = Image.new("RGB", (10, 10), color=(255, 0, 0))
        # Matte with all zeros
        matte = np.zeros((10, 10), dtype=np.float32)
        result = apply_threshold(original, matte, threshold=0)
        alpha = np.array(result.split()[-1])
        # All pixels with matte=0 are ≤ threshold=0, so transparent
        assert np.all(alpha == 0)

    def test_threshold_max_all_opaque_where_foreground(self):
        """FR-3.5: threshold=0 with max matte → all visible."""
        from rmbg_app import apply_threshold

        original = Image.new("RGB", (10, 10), color=(255, 0, 0))
        matte = np.full((10, 10), 255.0, dtype=np.float32)
        result = apply_threshold(original, matte, threshold=0)
        alpha = np.array(result.split()[-1])
        assert np.all(alpha == 255)

    def test_threshold_255_only_max_visible(self):
        """threshold=255 → only pixels with matte > 255 visible (none)."""
        from rmbg_app import apply_threshold

        original = Image.new("RGB", (10, 10), color=(255, 0, 0))
        matte = np.full((10, 10), 255.0, dtype=np.float32)
        result = apply_threshold(original, matte, threshold=255)
        alpha = np.array(result.split()[-1])
        # 255 > 255 is false, so all transparent
        assert np.all(alpha == 0)

    def test_threshold_splits_matte(self):
        """FR-3.5: Threshold correctly splits pixels."""
        from rmbg_app import apply_threshold

        original = Image.new("RGB", (4, 1), color=(100, 100, 100))
        # Matte: [50, 100, 150, 200]
        matte = np.array([[50, 100, 150, 200]], dtype=np.float32)
        result = apply_threshold(original, matte, threshold=128)
        alpha = np.array(result.split()[-1])
        # Pixels > 128 → opaque, pixels ≤ 128 → transparent
        expected = np.array([[0, 0, 255, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(alpha, expected)

    def test_result_is_rgba(self):
        """FR-5.4: Result must be RGBA."""
        from rmbg_app import apply_threshold

        original = Image.new("RGB", (10, 10))
        matte = np.full((10, 10), 128.0, dtype=np.float32)
        result = apply_threshold(original, matte, threshold=64)
        assert result.mode == "RGBA"

    def test_result_preserves_original_size(self):
        """FR-5.6: Preserve original resolution."""
        from rmbg_app import apply_threshold

        original = Image.new("RGB", (1920, 1080))
        matte = np.full((1080, 1920), 200.0, dtype=np.float32)
        result = apply_threshold(original, matte, threshold=128)
        assert result.size == (1920, 1080)


# ---------------------------------------------------------------------------
# Format validation tests
# ---------------------------------------------------------------------------

class TestSupportedFormats:
    """FR-1.3: Supported image format validation."""

    def test_supported_extensions(self):
        from rmbg_app import SUPPORTED_FORMATS

        for ext in (".png", ".jpeg", ".jpg", ".webp", ".bmp", ".tiff"):
            assert ext in SUPPORTED_FORMATS

    def test_unsupported_extension(self):
        from rmbg_app import SUPPORTED_FORMATS

        assert ".gif" not in SUPPORTED_FORMATS
        assert ".svg" not in SUPPORTED_FORMATS


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify application constants match spec."""

    def test_default_threshold(self):
        from rmbg_app import DEFAULT_THRESHOLD
        assert DEFAULT_THRESHOLD == 128

    def test_input_resolution(self):
        from rmbg_app import INPUT_RESOLUTION
        assert INPUT_RESOLUTION == (1024, 1024)

    def test_window_dimensions(self):
        from rmbg_app import WINDOW_WIDTH, WINDOW_HEIGHT
        assert WINDOW_WIDTH == 960
        assert WINDOW_HEIGHT == 720

    def test_preview_max(self):
        from rmbg_app import PREVIEW_MAX
        assert PREVIEW_MAX == 400

    def test_checkerboard_params(self):
        from rmbg_app import CHECKERBOARD_LIGHT, CHECKERBOARD_DARK, CHECKERBOARD_SIZE
        assert CHECKERBOARD_LIGHT == "#FFFFFF"
        assert CHECKERBOARD_DARK == "#C8C8C8"
        assert CHECKERBOARD_SIZE == 10

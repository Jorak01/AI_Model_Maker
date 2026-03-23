"""Tests for utils/image_tools.py — upscale, variations, color transfer."""

import os
import pytest

# Check if PIL is available
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from utils.image_tools import (
    upscale_image, generate_variations, color_transfer,
    merge_lora_weights, export_to_onnx, _check_deps,
)


# ---------------------------------------------------------------------------
# Dependency Check
# ---------------------------------------------------------------------------

class TestCheckDeps:
    def test_returns_dict(self):
        deps = _check_deps()
        assert isinstance(deps, dict)
        assert "PIL" in deps
        assert "torch" in deps


# ---------------------------------------------------------------------------
# Upscale
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
class TestUpscaleImage:
    @pytest.fixture
    def sample_image(self, tmp_path):
        img = Image.new("RGB", (32, 32), color="red")
        path = str(tmp_path / "test.png")
        img.save(path)
        return path

    def test_upscale_2x(self, sample_image, tmp_path):
        output = str(tmp_path / "upscaled.png")
        result = upscale_image(sample_image, output, scale=2)
        assert result == output
        assert os.path.exists(output)
        upscaled = Image.open(output)
        assert upscaled.size == (64, 64)

    def test_upscale_4x_default(self, sample_image, tmp_path):
        result = upscale_image(sample_image, scale=4)
        assert os.path.exists(result)
        upscaled = Image.open(result)
        assert upscaled.size == (128, 128)

    def test_upscale_missing_file(self, capsys):
        result = upscale_image("nonexistent.png")
        assert result == ""

    def test_upscale_auto_output_path(self, sample_image):
        result = upscale_image(sample_image, scale=2)
        assert "_upscaled" in result
        assert os.path.exists(result)
        os.remove(result)


# ---------------------------------------------------------------------------
# Variations
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PIL, reason="PIL not installed")
class TestGenerateVariations:
    @pytest.fixture
    def sample_image(self, tmp_path):
        img = Image.new("RGB", (64, 64), color="blue")
        path = str(tmp_path / "test.png")
        img.save(path)
        return path

    def test_generates_variations(self, sample_image, tmp_path):
        output_dir = str(tmp_path / "vars")
        results = generate_variations(sample_image, num_variations=3, output_dir=output_dir)
        assert len(results) > 0
        assert all(os.path.exists(p) for p in results)

    def test_missing_image(self, capsys):
        results = generate_variations("nonexistent.png")
        assert results == []

    def test_single_variation(self, sample_image, tmp_path):
        output_dir = str(tmp_path / "vars")
        results = generate_variations(sample_image, num_variations=1, output_dir=output_dir)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Color Transfer
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_PIL or not HAS_NUMPY, reason="PIL or numpy not installed")
class TestColorTransfer:
    @pytest.fixture
    def images(self, tmp_path):
        source = Image.new("RGB", (32, 32), color=(100, 50, 50))
        ref = Image.new("RGB", (32, 32), color=(50, 100, 200))
        src_path = str(tmp_path / "source.png")
        ref_path = str(tmp_path / "reference.png")
        source.save(src_path)
        ref.save(ref_path)
        return src_path, ref_path

    def test_basic_transfer(self, images, tmp_path):
        src, ref = images
        output = str(tmp_path / "styled.png")
        result = color_transfer(src, ref, output)
        assert result == output
        assert os.path.exists(output)

    def test_auto_output_path(self, images):
        src, ref = images
        result = color_transfer(src, ref)
        assert os.path.exists(result)
        os.remove(result)


# ---------------------------------------------------------------------------
# Scaffolds
# ---------------------------------------------------------------------------

class TestScaffolds:
    def test_merge_lora_returns_string(self, capsys):
        result = merge_lora_weights("base.pt", ["lora1.pt", "lora2.pt"])
        assert isinstance(result, str)

    def test_export_onnx_returns_string(self, capsys):
        result = export_to_onnx("model.pt")
        assert isinstance(result, str)

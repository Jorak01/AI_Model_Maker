"""Tests for the image generation module — tag processing, dataset creation, model lookup."""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_gen import (
    IMAGE_GEN_MODELS, TAG_SITE_FORMATS,
    normalize_tags, parse_tag_file, create_tag_dataset,
    list_image_models, get_image_model_by_number, get_image_model_by_name,
    _check_dependencies,
)


# ============================================================
# 1. Constants Tests
# ============================================================
class TestConstants:
    """Verify module constants are properly defined."""

    def test_image_gen_models_not_empty(self):
        assert len(IMAGE_GEN_MODELS) > 0

    def test_all_models_have_required_fields(self):
        for name, info in IMAGE_GEN_MODELS.items():
            assert "hf_id" in info, f"{name} missing hf_id"
            assert "desc" in info, f"{name} missing desc"
            assert "params" in info, f"{name} missing params"
            assert "resolution" in info, f"{name} missing resolution"
            assert "family" in info, f"{name} missing family"
            assert isinstance(info["resolution"], int)

    def test_tag_site_formats_not_empty(self):
        assert len(TAG_SITE_FORMATS) > 0

    def test_tag_formats_have_required_fields(self):
        for name, fmt in TAG_SITE_FORMATS.items():
            assert "separator" in fmt, f"{name} missing separator"
            assert "underscore_to_space" in fmt, f"{name} missing underscore_to_space"
            assert "desc" in fmt, f"{name} missing desc"

    def test_known_models_exist(self):
        assert "stable-diffusion-v1-5" in IMAGE_GEN_MODELS
        assert "stable-diffusion-xl" in IMAGE_GEN_MODELS

    def test_known_tag_formats_exist(self):
        assert "danbooru" in TAG_SITE_FORMATS
        assert "e621" in TAG_SITE_FORMATS
        assert "deviantart" in TAG_SITE_FORMATS
        assert "custom" in TAG_SITE_FORMATS


# ============================================================
# 2. Tag Normalization Tests
# ============================================================
class TestNormalizeTags:
    """Test normalize_tags for various formats."""

    def test_danbooru_basic(self):
        result = normalize_tags("1girl, blue_hair, school_uniform", "danbooru")
        assert "1girl" in result
        assert "blue hair" in result  # underscores replaced
        assert "school uniform" in result

    def test_danbooru_strips_weights(self):
        result = normalize_tags("1girl, (smile:1.2), blue_hair", "danbooru")
        assert "smile" in result
        assert "1.2" not in result
        assert "(" not in result

    def test_danbooru_strips_nested_parens(self):
        result = normalize_tags("(masterpiece), (best_quality:1.3)", "danbooru")
        assert "masterpiece" in result
        assert "best quality" in result
        assert "(" not in result

    def test_e621_space_separated(self):
        result = normalize_tags("blue_hair red_eyes solo", "e621")
        # e621 keeps underscores
        assert "blue_hair" in result
        assert "red_eyes" in result

    def test_deviantart_comma_separated(self):
        result = normalize_tags("digital art, fantasy, landscape", "deviantart")
        assert "digital art" in result
        assert "fantasy" in result
        assert "landscape" in result

    def test_custom_format(self):
        result = normalize_tags("tag_one, tag_two, tag_three", "custom")
        assert "tag one" in result  # underscores to spaces
        assert "tag two" in result

    def test_empty_string(self):
        result = normalize_tags("", "danbooru")
        assert result == ""

    def test_single_tag(self):
        result = normalize_tags("solo", "danbooru")
        assert result == "solo"

    def test_unknown_format_uses_custom(self):
        result = normalize_tags("hello_world", "unknown_format")
        assert "hello world" in result

    def test_strips_empty_tags(self):
        result = normalize_tags("tag1,, ,tag2", "danbooru")
        tags = [t.strip() for t in result.split(",")]
        assert "" not in tags


# ============================================================
# 3. Tag File Parsing Tests
# ============================================================
class TestParseTagFile:
    """Test parse_tag_file for various file formats."""

    def test_parse_json_with_string_tags(self, tmp_path):
        data = [
            {"image": "img1.png", "tags": "1girl, blue_hair"},
            {"image": "img2.png", "tags": "landscape, sunset"},
        ]
        path = str(tmp_path / "tags.json")
        with open(path, 'w') as f:
            json.dump(data, f)

        entries = parse_tag_file(path, "danbooru")
        assert len(entries) == 2
        assert entries[0]["image_path"] == "img1.png"
        assert "blue hair" in entries[0]["caption"]
        assert entries[0]["raw_tags"] == "1girl, blue_hair"

    def test_parse_json_with_list_tags(self, tmp_path):
        data = [
            {"image": "img1.png", "tags": ["1girl", "blue_hair"]},
        ]
        path = str(tmp_path / "tags.json")
        with open(path, 'w') as f:
            json.dump(data, f)

        entries = parse_tag_file(path, "danbooru")
        assert len(entries) == 1
        assert "blue hair" in entries[0]["caption"]

    def test_parse_json_with_image_path_key(self, tmp_path):
        data = [
            {"image_path": "img1.png", "caption": "a beautiful sunset"},
        ]
        path = str(tmp_path / "tags.json")
        with open(path, 'w') as f:
            json.dump(data, f)

        entries = parse_tag_file(path, "custom")
        assert len(entries) == 1
        assert entries[0]["image_path"] == "img1.png"

    def test_parse_txt_pipe_separated(self, tmp_path):
        content = "img1.png|1girl, blue_hair\nimg2.png|landscape, sunset\n"
        path = str(tmp_path / "tags.txt")
        with open(path, 'w') as f:
            f.write(content)

        entries = parse_tag_file(path, "danbooru")
        assert len(entries) == 2
        assert entries[0]["image_path"] == "img1.png"
        assert "blue hair" in entries[0]["caption"]

    def test_parse_txt_tags_only(self, tmp_path):
        content = "1girl, blue_hair\nlandscape, sunset\n"
        path = str(tmp_path / "tags.txt")
        with open(path, 'w') as f:
            f.write(content)

        entries = parse_tag_file(path, "danbooru")
        assert len(entries) == 2
        assert entries[0]["image_path"] == ""

    def test_parse_txt_skips_comments(self, tmp_path):
        content = "# this is a comment\n1girl, blue_hair\n# another comment\n"
        path = str(tmp_path / "tags.txt")
        with open(path, 'w') as f:
            f.write(content)

        entries = parse_tag_file(path, "danbooru")
        assert len(entries) == 1

    def test_parse_csv(self, tmp_path):
        content = "image,tags\nimg1.png,\"1girl, blue_hair\"\nimg2.png,\"landscape, sunset\"\n"
        path = str(tmp_path / "tags.csv")
        with open(path, 'w') as f:
            f.write(content)

        entries = parse_tag_file(path, "danbooru")
        assert len(entries) == 2
        assert entries[0]["image_path"] == "img1.png"

    def test_parse_nonexistent_file(self, capsys):
        entries = parse_tag_file("/nonexistent/file.json")
        assert entries == []
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()

    def test_parse_unsupported_format(self, tmp_path, capsys):
        path = str(tmp_path / "tags.xml")
        with open(path, 'w') as f:
            f.write("<tags/>")

        entries = parse_tag_file(path)
        assert entries == []
        captured = capsys.readouterr()
        assert "Unsupported" in captured.out


# ============================================================
# 4. Dataset Creation Tests
# ============================================================
class TestCreateTagDataset:
    """Test create_tag_dataset output."""

    def test_creates_valid_json(self, tmp_path, capsys):
        entries = [
            {"image_path": "img1.png", "caption": "1girl, blue hair", "raw_tags": "1girl, blue_hair"},
            {"image_path": "img2.png", "caption": "landscape", "raw_tags": "landscape"},
        ]
        output = str(tmp_path / "dataset.json")
        result = create_tag_dataset(entries, output)

        assert result == output
        assert os.path.exists(output)

        with open(output, 'r') as f:
            data = json.load(f)

        assert "metadata" in data
        assert "entries" in data
        assert data["metadata"]["count"] == 2
        assert len(data["entries"]) == 2

    def test_creates_parent_directories(self, tmp_path, capsys):
        output = str(tmp_path / "sub" / "dir" / "dataset.json")
        entries = [{"image_path": "img.png", "caption": "test", "raw_tags": "test"}]
        create_tag_dataset(entries, output)
        assert os.path.exists(output)

    def test_metadata_includes_format_info(self, tmp_path, capsys):
        entries = [{"image_path": "img.png", "caption": "test", "raw_tags": "test"}]
        output = str(tmp_path / "ds.json")
        create_tag_dataset(entries, output, site_format="e621")

        with open(output, 'r') as f:
            data = json.load(f)

        assert data["metadata"]["site_format"] == "e621"
        assert data["metadata"]["format"] == "image_caption_pairs"
        assert "created" in data["metadata"]


# ============================================================
# 5. Model Lookup Tests
# ============================================================
class TestModelLookup:
    """Test image model lookup functions."""

    def test_get_by_number_valid(self):
        result = get_image_model_by_number(1)
        assert result is not None
        name, info = result
        assert name in IMAGE_GEN_MODELS

    def test_get_by_number_last(self):
        result = get_image_model_by_number(len(IMAGE_GEN_MODELS))
        assert result is not None

    def test_get_by_number_zero(self):
        assert get_image_model_by_number(0) is None

    def test_get_by_number_out_of_range(self):
        assert get_image_model_by_number(999) is None

    def test_get_by_number_negative(self):
        assert get_image_model_by_number(-1) is None

    def test_get_by_name_exact(self):
        result = get_image_model_by_name("stable-diffusion-v1-5")
        assert result is not None
        name, info = result
        assert name == "stable-diffusion-v1-5"

    def test_get_by_name_case_insensitive(self):
        result = get_image_model_by_name("STABLE-DIFFUSION-V1-5")
        assert result is not None

    def test_get_by_name_nonexistent(self):
        assert get_image_model_by_name("nonexistent-model") is None

    def test_get_by_name_empty(self):
        assert get_image_model_by_name("") is None


# ============================================================
# 6. Display Tests
# ============================================================
class TestDisplay:
    """Test display functions."""

    def test_list_image_models_output(self, capsys):
        list_image_models()
        captured = capsys.readouterr()
        assert "Image Generation Models" in captured.out
        assert "stable-diffusion" in captured.out
        assert "Total:" in captured.out
        assert str(len(IMAGE_GEN_MODELS)) in captured.out

    def test_list_shows_all_models(self, capsys):
        list_image_models()
        captured = capsys.readouterr()
        for name in IMAGE_GEN_MODELS:
            assert name in captured.out


# ============================================================
# 7. Dependency Check Tests
# ============================================================
class TestDependencyCheck:
    """Test dependency checking functions."""

    def test_check_dependencies_returns_dict(self):
        deps = _check_dependencies()
        assert isinstance(deps, dict)
        assert "torch" in deps
        assert "transformers" in deps
        assert "diffusers" in deps
        assert "pillow" in deps

    def test_torch_is_available(self):
        """Torch should be installed in the test environment."""
        deps = _check_dependencies()
        assert deps["torch"] is True

    def test_transformers_is_available(self):
        """Transformers should be installed in the test environment."""
        deps = _check_dependencies()
        assert deps["transformers"] is True


# ============================================================
# 8. Import Tests
# ============================================================
class TestImports:
    """Verify all image_gen exports are importable."""

    def test_import_image_gen(self):
        import image_gen
        assert hasattr(image_gen, 'IMAGE_GEN_MODELS')
        assert hasattr(image_gen, 'TAG_SITE_FORMATS')
        assert hasattr(image_gen, 'normalize_tags')
        assert hasattr(image_gen, 'parse_tag_file')
        assert hasattr(image_gen, 'create_tag_dataset')
        assert hasattr(image_gen, 'list_image_models')
        assert hasattr(image_gen, 'get_image_model_by_number')
        assert hasattr(image_gen, 'get_image_model_by_name')
        assert hasattr(image_gen, 'generate_image')
        assert hasattr(image_gen, 'train_image_model')
        assert hasattr(image_gen, 'interactive_image_gen')

    def test_import_from_run(self):
        import run
        assert hasattr(run, 'cmd_image_gen')

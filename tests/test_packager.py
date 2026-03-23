"""Tests for utils/model_packager.py — export, import, list archives."""

import os
import json
import tarfile
import pytest
from utils.model_packager import export_model, import_model, list_archives


class TestExportModel:
    def test_export_creates_archive(self, tmp_path):
        # Create a fake checkpoint
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "model.pt").write_bytes(b"fake model data")

        output = str(tmp_path / "export.tar.gz")
        result = export_model(
            "test-model", output,
            checkpoint_dir=str(ckpt_dir),
            include_config=False,
        )
        assert os.path.exists(result)
        assert result.endswith('.tar.gz')

    def test_export_contains_manifest(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "model.pt").write_bytes(b"data")

        output = str(tmp_path / "export.tar.gz")
        export_model("test-model", output, checkpoint_dir=str(ckpt_dir),
                      include_config=False)

        with tarfile.open(output, 'r:gz') as tar:
            names = tar.getnames()
            assert "manifest.json" in names
            mf = tar.extractfile("manifest.json")
            manifest = json.loads(mf.read().decode('utf-8'))
            assert manifest["name"] == "test-model"

    def test_export_includes_checkpoint(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "best_model.pt").write_bytes(b"model weights")

        output = str(tmp_path / "export.tar.gz")
        export_model("test", output, checkpoint_dir=str(ckpt_dir),
                      include_config=False)

        with tarfile.open(output, 'r:gz') as tar:
            names = tar.getnames()
            assert any("best_model.pt" in n for n in names)

    def test_export_auto_extension(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        output = str(tmp_path / "export")
        result = export_model("test", output, checkpoint_dir=str(ckpt_dir),
                              include_config=False)
        assert result.endswith('.tar.gz')

    def test_export_with_notes(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "model.pt").write_bytes(b"data")

        output = str(tmp_path / "export.tar.gz")
        export_model("test", output, checkpoint_dir=str(ckpt_dir),
                      include_config=False, notes="Test notes")

        with tarfile.open(output, 'r:gz') as tar:
            mf = tar.extractfile("manifest.json")
            manifest = json.loads(mf.read().decode('utf-8'))
            assert manifest["notes"] == "Test notes"

    def test_export_missing_checkpoint_dir(self, tmp_path, capsys):
        output = str(tmp_path / "export.tar.gz")
        result = export_model("test", output,
                              checkpoint_dir=str(tmp_path / "nonexistent"),
                              include_config=False)
        assert os.path.exists(result)


class TestImportModel:
    def test_import_extracts_files(self, tmp_path):
        # Create archive first
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "model.pt").write_bytes(b"model data")

        archive = str(tmp_path / "model.tar.gz")
        export_model("import-test", archive, checkpoint_dir=str(ckpt_dir),
                      include_config=False)

        # Import to new location
        target = str(tmp_path / "imported")
        result = import_model(archive, target_dir=target, register=False)
        assert result == target
        assert os.path.exists(os.path.join(target, "model.pt"))

    def test_import_missing_archive(self, capsys):
        result = import_model("nonexistent.tar.gz", register=False)
        assert result is None

    def test_import_reads_manifest(self, tmp_path, capsys):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "model.pt").write_bytes(b"data")

        archive = str(tmp_path / "model.tar.gz")
        export_model("manifest-test", archive, checkpoint_dir=str(ckpt_dir),
                      include_config=False, notes="test notes")

        target = str(tmp_path / "imported")
        import_model(archive, target_dir=target, register=False)
        captured = capsys.readouterr()
        assert "manifest-test" in captured.out


class TestListArchives:
    def test_empty_directory(self, tmp_path):
        archives = list_archives(str(tmp_path))
        assert archives == []

    def test_lists_archives(self, tmp_path):
        # Create a fake archive
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "model.pt").write_bytes(b"data")

        exports_dir = tmp_path / "exports"
        exports_dir.mkdir()
        archive = str(exports_dir / "test.tar.gz")
        export_model("list-test", archive, checkpoint_dir=str(ckpt_dir),
                      include_config=False)

        archives = list_archives(str(exports_dir))
        assert len(archives) == 1
        assert archives[0]["file"] == "test.tar.gz"
        assert archives[0]["name"] == "list-test"

    def test_nonexistent_directory(self):
        archives = list_archives("nonexistent_dir")
        assert archives == []

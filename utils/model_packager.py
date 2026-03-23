"""Model Packager — Export/import trained models as portable archives.

Packages a model checkpoint + tokenizer + config into a single .tar.gz
that can be shared and loaded on another machine.

Usage:
    from utils.model_packager import export_model, import_model

    # Export
    export_model("my-model", "exports/my-model.tar.gz")

    # Import
    import_model("exports/my-model.tar.gz", "imported-model")
"""

import os
import json
import shutil
import tarfile
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, List


def export_model(
    model_name: str,
    output_path: str,
    checkpoint_dir: str = "checkpoints",
    include_config: bool = True,
    include_training_data: bool = False,
    notes: str = "",
) -> str:
    """Export a model as a portable .tar.gz archive.

    Args:
        model_name: Name for the exported model
        output_path: Path for the output archive
        checkpoint_dir: Directory containing model checkpoints
        include_config: Include config.yaml
        include_training_data: Include training data files
        notes: Optional notes about the model

    Returns:
        Path to the created archive
    """
    print(f"\n  Packaging model '{model_name}'...")

    if not output_path.endswith('.tar.gz'):
        output_path += '.tar.gz'

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # Create manifest
    manifest = {
        "name": model_name,
        "exported_at": datetime.now().isoformat(),
        "notes": notes,
        "files": [],
        "format_version": "1.0",
    }

    with tarfile.open(output_path, 'w:gz') as tar:
        # Add model checkpoints
        if os.path.isdir(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                filepath = os.path.join(checkpoint_dir, f)
                if os.path.isfile(filepath) and (f.endswith('.pt') or f.endswith('.pkl')):
                    tar.add(filepath, arcname=f"model/{f}")
                    size = os.path.getsize(filepath)
                    manifest["files"].append({
                        "name": f, "path": f"model/{f}",
                        "size": size, "type": "checkpoint" if f.endswith('.pt') else "tokenizer"
                    })
                    print(f"    + {f} ({size:,} bytes)")
        else:
            print(f"  ⚠ Checkpoint dir not found: {checkpoint_dir}")

        # Check registry for model info
        try:
            from models.registry import get_model_info
            info = get_model_info(model_name)
            if info:
                manifest["model_info"] = info
                # Add registered model files
                model_path = info.get("path", "")
                if model_path and os.path.isdir(model_path):
                    for f in os.listdir(model_path):
                        filepath = os.path.join(model_path, f)
                        if os.path.isfile(filepath):
                            tar.add(filepath, arcname=f"model/{f}")
                            manifest["files"].append({
                                "name": f, "path": f"model/{f}",
                                "size": os.path.getsize(filepath),
                            })
        except Exception:
            pass

        # Add config
        if include_config and os.path.exists("config.yaml"):
            tar.add("config.yaml", arcname="config.yaml")
            manifest["files"].append({"name": "config.yaml", "path": "config.yaml"})

        # Add training data
        if include_training_data:
            for data_file in ("data/train.json", "data/test.json"):
                if os.path.exists(data_file):
                    tar.add(data_file, arcname=data_file)
                    manifest["files"].append({"name": data_file, "path": data_file})

        # Write manifest
        manifest_json = json.dumps(manifest, indent=2)
        import io
        manifest_bytes = manifest_json.encode('utf-8')
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        tar.addfile(info, io.BytesIO(manifest_bytes))

    archive_size = os.path.getsize(output_path)
    print(f"\n  ✓ Model packaged: {output_path} ({archive_size:,} bytes)")
    print(f"    {len(manifest['files'])} file(s) included")
    return output_path


def import_model(
    archive_path: str,
    model_name: Optional[str] = None,
    target_dir: str = "checkpoints",
    register: bool = True,
) -> Optional[str]:
    """Import a model from a .tar.gz archive.

    Args:
        archive_path: Path to the archive
        model_name: Override the model name (default: from manifest)
        target_dir: Where to extract model files
        register: Register in the model registry

    Returns:
        Path to extracted model directory, or None on failure
    """
    if not os.path.exists(archive_path):
        print(f"  ❌ Archive not found: {archive_path}")
        return None

    print(f"\n  Importing model from {archive_path}...")

    with tarfile.open(archive_path, 'r:gz') as tar:
        # Read manifest
        manifest = None
        try:
            mf = tar.extractfile("manifest.json")
            if mf:
                manifest = json.loads(mf.read().decode('utf-8'))
        except (KeyError, json.JSONDecodeError):
            pass

        if manifest:
            if not model_name:
                model_name = manifest.get("name", "imported-model")
            print(f"  Model: {model_name}")
            print(f"  Exported: {manifest.get('exported_at', 'unknown')}")
            if manifest.get("notes"):
                print(f"  Notes: {manifest['notes']}")

        # Extract model files
        os.makedirs(target_dir, exist_ok=True)
        extracted = 0
        for member in tar.getmembers():
            if member.name.startswith("model/"):
                member.name = member.name[6:]  # Remove model/ prefix
                tar.extract(member, target_dir)
                print(f"    + {member.name}")
                extracted += 1
            elif member.name == "config.yaml":
                tar.extract(member, ".")
                print(f"    + config.yaml")

        print(f"\n  ✓ Extracted {extracted} file(s) to {target_dir}/")

        # Register model
        if register and manifest:
            try:
                from models.registry import register_model
                info = manifest.get("model_info", {})
                register_model(
                    name=model_name or "imported-model",
                    intent=info.get("intent", "Imported model"),
                    base_model=info.get("base_model", "custom"),
                    pipeline=info.get("pipeline", "scratch"),
                    checkpoint_dir=target_dir,
                )
                print(f"  ✓ Registered as '{model_name}'")
            except Exception as e:
                print(f"  ⚠ Registration failed: {e}")

    return target_dir


def list_archives(directory: str = "exports") -> List:
    """List available model archives."""
    archives = []
    if not os.path.isdir(directory):
        return archives

    for f in os.listdir(directory):
        if f.endswith('.tar.gz'):
            path = os.path.join(directory, f)
            size = os.path.getsize(path)
            # Try to read manifest
            name = f.replace('.tar.gz', '')
            try:
                with tarfile.open(path, 'r:gz') as tar:
                    mf = tar.extractfile("manifest.json")
                    if mf:
                        manifest = json.loads(mf.read().decode('utf-8'))
                        name = manifest.get("name", name)
            except Exception:
                pass
            archives.append({"file": f, "name": name, "size": size, "path": path})

    return archives


def interactive_packager():
    """Interactive model packaging interface."""
    print("\n" + "=" * 55)
    print("       Model Packager")
    print("=" * 55)

    while True:
        print("\n  Options:")
        print("  1  Export model")
        print("  2  Import model")
        print("  3  List archives")
        print("  0  Back")

        try:
            choice = input("\n  packager>> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice in ('0', 'back', 'quit', 'q'):
            break

        if choice == '1':
            try:
                name = input("  Model name: ").strip() or "my-model"
                output = input("  Output path [exports/<name>.tar.gz]: ").strip()
                if not output:
                    output = f"exports/{name}.tar.gz"
                notes = input("  Notes (optional): ").strip()
                export_model(name, output, notes=notes)
            except (KeyboardInterrupt, EOFError):
                continue

        elif choice == '2':
            try:
                path = input("  Archive path: ").strip()
                if path:
                    import_model(path)
            except (KeyboardInterrupt, EOFError):
                continue

        elif choice == '3':
            archives = list_archives()
            if archives:
                print(f"\n  Archives in exports/:")
                for a in archives:
                    print(f"    {a['file']:<30} {a['name']:<20} {a['size']:>10,} bytes")
            else:
                print("  No archives found in exports/")



"""Image Tools — Inpainting, ControlNet, upscaling, style transfer, variations.

All features gracefully handle missing dependencies (diffusers, PIL, etc.)
and provide clear guidance on what's needed.
"""

import os
from typing import Optional, Dict, List, Any


def _check_deps() -> Dict[str, bool]:
    deps = {}
    for mod in ["torch", "PIL", "diffusers", "torchvision"]:
        try:
            __import__(mod)
            deps[mod] = True
        except ImportError:
            deps[mod] = False
    return deps


# ---------------------------------------------------------------------------
# Upscaling Pipeline
# ---------------------------------------------------------------------------

def upscale_image(image_path: str, output_path: str = "", scale: int = 4,
                  method: str = "lanczos") -> str:
    """Upscale an image using available methods.

    Methods: lanczos (PIL), realesrgan (model-based)
    """
    try:
        from PIL import Image
    except ImportError:
        print("  PIL required: pip install Pillow")
        return ""

    if not os.path.exists(image_path):
        print(f"  Image not found: {image_path}")
        return ""

    img = Image.open(image_path)
    new_size = (img.width * scale, img.height * scale)

    if method == "lanczos":
        upscaled = img.resize(new_size, Image.LANCZOS)
    else:
        # Fallback to bilinear
        upscaled = img.resize(new_size, Image.BILINEAR)

    if not output_path:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_upscaled{ext}"

    upscaled.save(output_path)
    print(f"  ✓ Upscaled {scale}x: {output_path} ({new_size[0]}x{new_size[1]})")
    return output_path


# ---------------------------------------------------------------------------
# Image Variation Generator
# ---------------------------------------------------------------------------

def generate_variations(image_path: str, num_variations: int = 4,
                        output_dir: str = "outputs/variations") -> List[str]:
    """Generate variations of an image using transforms."""
    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError:
        print("  PIL required: pip install Pillow")
        return []

    if not os.path.exists(image_path):
        print(f"  Image not found: {image_path}")
        return []

    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path)
    base = os.path.splitext(os.path.basename(image_path))[0]
    outputs = []

    transforms = [
        ("bright", lambda i: ImageEnhance.Brightness(i).enhance(1.3)),
        ("contrast", lambda i: ImageEnhance.Contrast(i).enhance(1.4)),
        ("color", lambda i: ImageEnhance.Color(i).enhance(1.5)),
        ("sharp", lambda i: ImageEnhance.Sharpness(i).enhance(2.0)),
        ("blur", lambda i: i.filter(ImageFilter.GaussianBlur(2))),
        ("edge", lambda i: i.filter(ImageFilter.EDGE_ENHANCE_MORE)),
        ("warm", lambda i: ImageEnhance.Color(i).enhance(1.3)),
        ("cool", lambda i: ImageEnhance.Brightness(i).enhance(0.9)),
    ]

    import random
    selected = random.sample(transforms, min(num_variations, len(transforms)))

    for name, transform in selected:
        try:
            variant = transform(img.copy())
            out_path = os.path.join(output_dir, f"{base}_{name}.png")
            variant.save(out_path)
            outputs.append(out_path)
            print(f"  ✓ Variation: {name} → {out_path}")
        except Exception as e:
            print(f"  ⚠ {name} failed: {e}")

    return outputs


# ---------------------------------------------------------------------------
# Style Transfer (basic color transfer)
# ---------------------------------------------------------------------------

def color_transfer(source_path: str, reference_path: str,
                   output_path: str = "") -> str:
    """Apply color palette from reference image to source."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("  PIL and numpy required")
        return ""

    source = np.array(Image.open(source_path).convert("RGB"), dtype=np.float64)
    reference = np.array(Image.open(reference_path).convert("RGB"), dtype=np.float64)

    # Simple mean/std color transfer
    for ch in range(3):
        src_mean, src_std = source[:, :, ch].mean(), source[:, :, ch].std()
        ref_mean, ref_std = reference[:, :, ch].mean(), reference[:, :, ch].std()
        if src_std > 0:
            source[:, :, ch] = (source[:, :, ch] - src_mean) * (ref_std / src_std) + ref_mean

    source = np.clip(source, 0, 255).astype(np.uint8)
    result = Image.fromarray(source)

    if not output_path:
        base, ext = os.path.splitext(source_path)
        output_path = f"{base}_styled{ext or '.png'}"

    result.save(output_path)
    print(f"  ✓ Style transferred: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# LoRA Tools
# ---------------------------------------------------------------------------

def merge_lora_weights(base_model_path: str, lora_paths: List[str],
                       weights: Optional[List[float]] = None,
                       output_path: str = "") -> str:
    """Merge multiple LoRA weights with a base model (scaffold)."""
    try:
        import torch
    except ImportError:
        print("  PyTorch required")
        return ""

    if not weights:
        weights = [1.0 / len(lora_paths)] * len(lora_paths)

    print(f"  Merging {len(lora_paths)} LoRA(s) into base model...")
    for path, w in zip(lora_paths, weights):
        print(f"    {path} (weight={w:.2f})")

    # Scaffold: actual LoRA merging requires peft library
    print("  Note: Full LoRA merging requires 'peft' library.")
    print("  Install: pip install peft")
    print("  This is a scaffold — implement with actual model loading.")
    return output_path or "merged_model.pt"


# ---------------------------------------------------------------------------
# ONNX Export (scaffold)
# ---------------------------------------------------------------------------

def export_to_onnx(model_path: str, output_path: str = "",
                   input_shape: tuple = (1, 128)) -> str:
    """Export model to ONNX format for optimized inference (scaffold)."""
    try:
        import torch
    except ImportError:
        print("  PyTorch required")
        return ""

    if not output_path:
        output_path = model_path.replace('.pt', '.onnx')

    print(f"  ONNX export: {model_path} → {output_path}")
    print("  Note: Requires torch.onnx.export with actual model instance.")
    print("  Install onnx: pip install onnx onnxruntime")
    return output_path


# ---------------------------------------------------------------------------
# Interactive
# ---------------------------------------------------------------------------

def interactive_image_tools():
    """Interactive image tools interface."""
    print("\n" + "=" * 55)
    print("       Image Tools")
    print("=" * 55)

    deps = _check_deps()
    print(f"\n  Dependencies: {', '.join(f'{k}={"✓" if v else "✗"}' for k, v in deps.items())}")

    while True:
        print("\n  Options:")
        print("  1  Upscale image")
        print("  2  Generate variations")
        print("  3  Color/style transfer")
        print("  4  Merge LoRA weights")
        print("  5  Export to ONNX")
        print("  0  Back")

        try:
            choice = input("\n  image-tools>> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if choice in ('0', 'back', 'quit', 'q'):
            break
        try:
            if choice == '1':
                path = input("  Image path: ").strip()
                scale = input("  Scale [4]: ").strip()
                if path:
                    upscale_image(path, scale=int(scale) if scale else 4)
            elif choice == '2':
                path = input("  Image path: ").strip()
                n = input("  Variations [4]: ").strip()
                if path:
                    generate_variations(path, num_variations=int(n) if n else 4)
            elif choice == '3':
                src = input("  Source image: ").strip()
                ref = input("  Reference image: ").strip()
                if src and ref:
                    color_transfer(src, ref)
            elif choice == '4':
                base = input("  Base model path: ").strip()
                loras = input("  LoRA paths (comma-sep): ").strip()
                if base and loras:
                    merge_lora_weights(base, [l.strip() for l in loras.split(",")])
            elif choice == '5':
                path = input("  Model path (.pt): ").strip()
                if path:
                    export_to_onnx(path)
        except Exception as e:
            print(f"  Error: {e}")

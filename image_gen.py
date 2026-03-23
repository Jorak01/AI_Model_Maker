"""Image Generation Module - Generate images with diffusion models, train with tag-based data.

Supports:
  - Text-to-image generation using Stable Diffusion and similar models
  - Tag-based training data from art cataloging sites (Danbooru, DeviantArt, etc.)
  - Fine-tuning image models on custom tagged datasets
  - Interactive generation and training menus

Tag formats supported:
  - Danbooru-style: comma-separated tags (e.g., "1girl, blue_hair, school_uniform")
  - DeviantArt/general: natural language captions with optional tags
  - Custom JSON datasets with image paths + tag lists
"""

import os
import json
import re
import yaml
from typing import Optional, List, Dict, Tuple
from datetime import datetime


# ── Constants ────────────────────────────────────────────────────────────

IMAGE_GEN_MODELS = {
    "stable-diffusion-v1-5": {
        "hf_id": "runwayml/stable-diffusion-v1-5",
        "desc": "Stable Diffusion v1.5 — general-purpose text-to-image",
        "params": "860M",
        "resolution": 512,
        "family": "stable-diffusion",
    },
    "stable-diffusion-v2-1": {
        "hf_id": "stabilityai/stable-diffusion-2-1",
        "desc": "Stable Diffusion v2.1 — improved quality, 768px",
        "params": "865M",
        "resolution": 768,
        "family": "stable-diffusion",
    },
    "stable-diffusion-xl": {
        "hf_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "desc": "SDXL — high quality, 1024px generation",
        "params": "3.5B",
        "resolution": 1024,
        "family": "stable-diffusion-xl",
    },
    "openjourney": {
        "hf_id": "prompthero/openjourney",
        "desc": "Midjourney-style fine-tuned SD v1.5",
        "params": "860M",
        "resolution": 512,
        "family": "stable-diffusion",
    },
    "waifu-diffusion": {
        "hf_id": "hakurei/waifu-diffusion",
        "desc": "Anime/manga style fine-tuned model",
        "params": "860M",
        "resolution": 512,
        "family": "stable-diffusion",
    },
    "anything-v5": {
        "hf_id": "stablediffusionapi/anything-v5",
        "desc": "Anime illustration model v5",
        "params": "860M",
        "resolution": 512,
        "family": "stable-diffusion",
    },
}

# Tag normalization patterns for various art sites
TAG_SITE_FORMATS = {
    "danbooru": {
        "separator": ",",
        "underscore_to_space": True,
        "strip_parentheses": True,
        "desc": "Danbooru/Gelbooru style: comma-separated, underscores",
    },
    "e621": {
        "separator": " ",
        "underscore_to_space": False,
        "strip_parentheses": False,
        "desc": "e621 style: space-separated tags with underscores",
    },
    "deviantart": {
        "separator": ",",
        "underscore_to_space": True,
        "strip_parentheses": False,
        "desc": "DeviantArt style: comma-separated, natural language",
    },
    "custom": {
        "separator": ",",
        "underscore_to_space": True,
        "strip_parentheses": False,
        "desc": "Custom format: comma-separated tags",
    },
}

# Default output directory for generated images
DEFAULT_OUTPUT_DIR = "generated_images"
DEFAULT_DATASET_DIR = "data/image_training"


# ── Dependency checking ──────────────────────────────────────────────────

def _check_dependencies() -> Dict[str, bool]:
    """Check which optional dependencies are installed."""
    deps = {}
    try:
        import diffusers  # noqa: F401  # type: ignore[import-unresolved]
        deps["diffusers"] = bool(diffusers)
    except ImportError:
        deps["diffusers"] = False

    try:
        from PIL import Image  # noqa: F401
        deps["pillow"] = True
    except ImportError:
        deps["pillow"] = False

    try:
        import torch  # noqa: F401
        deps["torch"] = True
    except ImportError:
        deps["torch"] = False

    try:
        import transformers  # noqa: F401
        deps["transformers"] = True
    except ImportError:
        deps["transformers"] = False

    return deps


def _require_dependencies(needed: Optional[List[str]] = None) -> bool:
    """Check and report missing dependencies. Returns True if all present."""
    if needed is None:
        needed = ["diffusers", "pillow", "torch"]

    deps = _check_dependencies()
    missing = [d for d in needed if not deps.get(d, False)]

    if missing:
        print(f"\n  ✗ Missing required packages: {', '.join(missing)}")
        print("  Install them with:")
        install_names = {
            "diffusers": "diffusers[torch]",
            "pillow": "Pillow",
            "torch": "torch",
            "transformers": "transformers",
        }
        pkgs = " ".join(install_names.get(m, m) for m in missing)
        print(f"    pip install {pkgs}")
        return False
    return True


# ── Tag Processing ───────────────────────────────────────────────────────

def normalize_tags(tags_str: str, site_format: str = "danbooru") -> str:
    """Normalize a tag string from a specific art site format to a clean caption.

    Args:
        tags_str: Raw tag string (e.g., "1girl, blue_hair, school_uniform, (smile:1.2)")
        site_format: One of TAG_SITE_FORMATS keys.

    Returns:
        Normalized caption string suitable for model training.
    """
    fmt = TAG_SITE_FORMATS.get(site_format, TAG_SITE_FORMATS["custom"])
    separator = fmt["separator"]

    # Split tags
    if separator == " ":
        tags = tags_str.split()
    else:
        tags = [t.strip() for t in tags_str.split(separator)]

    # Filter empty
    tags = [t for t in tags if t]

    # Strip parentheses and weights (e.g., "(smile:1.2)" → "smile")
    if fmt.get("strip_parentheses"):
        cleaned = []
        for tag in tags:
            # Remove weight notation like (tag:1.2)
            tag = re.sub(r'\(([^:)]+)(?::[0-9.]+)?\)', r'\1', tag)
            # Remove standalone parens
            tag = tag.replace('(', '').replace(')', '')
            cleaned.append(tag.strip())
        tags = cleaned

    # Underscores to spaces
    if fmt.get("underscore_to_space"):
        tags = [t.replace('_', ' ') for t in tags]

    # Final cleanup
    tags = [t.strip() for t in tags if t.strip()]

    return ", ".join(tags)


def parse_tag_file(filepath: str, site_format: str = "danbooru") -> List[Dict]:
    """Parse a tag file into training entries.

    Supports formats:
      - JSON: List of {"image": "path", "tags": "tag1, tag2"} or {"image": "path", "tags": ["tag1", "tag2"]}
      - TXT: One entry per line, format: "image_path|tags" or just tags (paired with images by filename)
      - CSV: image_path,tags columns

    Args:
        filepath: Path to the tag file.
        site_format: Tag format for normalization.

    Returns:
        List of {"image_path": str, "caption": str, "raw_tags": str} dicts.
    """
    entries: List[Dict] = []

    if not os.path.exists(filepath):
        print(f"  ✗ File not found: {filepath}")
        return entries

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".json":
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    image_path = item.get("image", item.get("image_path", ""))
                    raw_tags = item.get("tags", item.get("caption", ""))
                    if isinstance(raw_tags, list):
                        raw_tags = ", ".join(raw_tags)
                    caption = normalize_tags(str(raw_tags), site_format)
                    entries.append({
                        "image_path": image_path,
                        "caption": caption,
                        "raw_tags": str(raw_tags),
                    })

    elif ext == ".txt":
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '|' in line:
                    parts = line.split('|', 1)
                    image_path = parts[0].strip()
                    raw_tags = parts[1].strip()
                else:
                    image_path = ""
                    raw_tags = line
                caption = normalize_tags(raw_tags, site_format)
                entries.append({
                    "image_path": image_path,
                    "caption": caption,
                    "raw_tags": raw_tags,
                })

    elif ext == ".csv":
        import csv
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    image_path = row[0].strip()
                    raw_tags = row[1].strip()
                    caption = normalize_tags(raw_tags, site_format)
                    entries.append({
                        "image_path": image_path,
                        "caption": caption,
                        "raw_tags": raw_tags,
                    })

    else:
        print(f"  ✗ Unsupported file format: {ext}")
        print("    Supported: .json, .txt, .csv")

    return entries


def create_tag_dataset(entries: List[Dict], output_path: str,
                       site_format: str = "danbooru") -> str:
    """Create a training-ready JSON dataset from tag entries.

    Args:
        entries: List of {"image_path": str, "caption": str} dicts.
        output_path: Where to save the dataset JSON.
        site_format: Tag format used.

    Returns:
        Path to the saved dataset.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    dataset = {
        "metadata": {
            "format": "image_caption_pairs",
            "site_format": site_format,
            "count": len(entries),
            "created": datetime.now().isoformat(),
        },
        "entries": entries,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Dataset saved: {output_path} ({len(entries)} entries)")
    return output_path


# ── Image Generation ─────────────────────────────────────────────────────

def list_image_models():
    """Print available image generation models."""
    print("\n" + "=" * 75)
    print("       Image Generation Models")
    print("=" * 75)
    print(f"\n  {'#':<4} {'Name':<28} {'Params':<8} {'Res':<6} {'Description'}")
    print("  " + "-" * 70)

    for i, (name, info) in enumerate(IMAGE_GEN_MODELS.items(), 1):
        res = f"{info['resolution']}px"
        print(f"  {i:<4} {name:<28} {info['params']:<8} {res:<6} {info['desc'][:35]}")

    print(f"\n  Total: {len(IMAGE_GEN_MODELS)} models")
    print()


def get_image_model_by_number(number: int) -> Optional[Tuple[str, Dict]]:
    """Get an image model by its list number (1-indexed)."""
    models = list(IMAGE_GEN_MODELS.items())
    if 1 <= number <= len(models):
        name, info = models[number - 1]
        return name, info
    return None


def get_image_model_by_name(name: str) -> Optional[Tuple[str, Dict]]:
    """Get an image model by name (case-insensitive)."""
    name_lower = name.lower().strip()
    for key, info in IMAGE_GEN_MODELS.items():
        if key.lower() == name_lower:
            return key, info
    return None


def generate_image(prompt: str, model_name: str = "stable-diffusion-v1-5",
                   output_dir: str = DEFAULT_OUTPUT_DIR,
                   num_images: int = 1,
                   steps: int = 50,
                   guidance_scale: float = 7.5,
                   negative_prompt: Optional[str] = None,
                   width: Optional[int] = None,
                   height: Optional[int] = None,
                   seed: Optional[int] = None,
                   device: str = "cpu") -> List[str]:
    """Generate images from a text prompt using a diffusion model.

    Args:
        prompt: Text description of the desired image.
        model_name: Key from IMAGE_GEN_MODELS.
        output_dir: Directory to save generated images.
        num_images: Number of images to generate.
        steps: Number of diffusion steps (more = higher quality, slower).
        guidance_scale: How closely to follow the prompt (7-12 typical).
        negative_prompt: Things to avoid in the image.
        width: Image width (default from model config).
        height: Image height (default from model config).
        seed: Random seed for reproducibility.
        device: Torch device string.

    Returns:
        List of paths to saved images.
    """
    if not _require_dependencies(["diffusers", "pillow", "torch"]):
        return []

    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler  # type: ignore[import-unresolved]

    model_info = IMAGE_GEN_MODELS.get(model_name)
    if model_info is None:
        print(f"  ✗ Unknown model: {model_name}")
        print(f"  Available: {', '.join(IMAGE_GEN_MODELS.keys())}")
        return []

    hf_id = model_info["hf_id"]
    default_res = model_info["resolution"]
    w = width or default_res
    h = height or default_res

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  ── Image Generation ──")
    print(f"  Model:    {model_name} ({hf_id})")
    print(f"  Prompt:   {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
    print(f"  Size:     {w}x{h}")
    print(f"  Steps:    {steps}")
    print(f"  Images:   {num_images}")
    if negative_prompt:
        print(f"  Negative: {negative_prompt[:40]}...")
    print()

    try:
        # Load pipeline
        print("  Loading model pipeline...")
        dtype = torch.float16 if device != "cpu" and torch.cuda.is_available() else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            hf_id, torch_dtype=dtype, safety_checker=None
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)

        # Enable memory optimizations
        if device != "cpu":
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)

        # Generate
        saved_paths: List[str] = []
        for i in range(num_images):
            print(f"  Generating image {i + 1}/{num_images}...")
            result = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=w,
                height=h,
                generator=generator,
            )
            image = result.images[0]

            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = re.sub(r'[^\w\s-]', '', prompt[:30]).strip().replace(' ', '_')
            filename = f"{safe_prompt}_{timestamp}_{i + 1}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            saved_paths.append(filepath)
            print(f"  ✓ Saved: {filepath}")

        print(f"\n  ✓ Generated {len(saved_paths)} image(s) in {output_dir}/")
        return saved_paths

    except Exception as e:
        print(f"  ✗ Generation error: {e}")
        return []


# ── Training ─────────────────────────────────────────────────────────────

def train_image_model(model_name: str = "stable-diffusion-v1-5",
                      dataset_path: str = "",
                      output_dir: str = "checkpoints/image_model",
                      epochs: int = 5,
                      learning_rate: float = 1e-5,
                      batch_size: int = 1,
                      resolution: Optional[int] = None,
                      device: str = "cpu",
                      use_lora: bool = True,
                      lora_rank: int = 4) -> Optional[str]:
    """Fine-tune an image generation model on a tagged dataset.

    Uses LoRA (Low-Rank Adaptation) by default for efficient training.

    Args:
        model_name: Key from IMAGE_GEN_MODELS to use as base.
        dataset_path: Path to training dataset JSON (from create_tag_dataset).
        output_dir: Where to save fine-tuned model.
        epochs: Number of training epochs.
        learning_rate: Training learning rate.
        batch_size: Training batch size.
        resolution: Image resolution for training.
        device: Torch device string.
        use_lora: Use LoRA for efficient fine-tuning.
        lora_rank: LoRA rank (lower = less params, higher = more capacity).

    Returns:
        Path to saved model, or None on failure.
    """
    if not _require_dependencies(["diffusers", "pillow", "torch", "transformers"]):
        return None

    model_info = IMAGE_GEN_MODELS.get(model_name)
    if model_info is None:
        print(f"  ✗ Unknown model: {model_name}")
        return None

    if not dataset_path or not os.path.exists(dataset_path):
        print(f"  ✗ Dataset not found: {dataset_path}")
        return None

    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    entries = dataset.get("entries", [])
    if not entries:
        print("  ✗ Dataset is empty")
        return None

    # Validate images exist
    valid_entries = []
    for entry in entries:
        img_path = entry.get("image_path", "")
        if img_path and os.path.exists(img_path):
            valid_entries.append(entry)

    if not valid_entries:
        print("  ✗ No valid image files found in dataset")
        print("    Ensure image_path fields point to existing files")
        return None

    hf_id = model_info["hf_id"]
    res = resolution or model_info["resolution"]

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  ── Image Model Training ──")
    print(f"  Base model:  {model_name} ({hf_id})")
    print(f"  Dataset:     {dataset_path} ({len(valid_entries)} images)")
    print(f"  Resolution:  {res}x{res}")
    print(f"  Epochs:      {epochs}")
    print(f"  LR:          {learning_rate}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LoRA:        {'rank=' + str(lora_rank) if use_lora else 'disabled'}")
    print(f"  Output:      {output_dir}")
    print()

    try:
        import torch
        from PIL import Image
        from diffusers import StableDiffusionPipeline, DDPMScheduler  # type: ignore[import-unresolved]
        from transformers import CLIPTokenizer
        from torch.utils.data import Dataset, DataLoader
        import torch.nn.functional as F

        print("  Loading base model...")
        dtype = torch.float32  # Training needs float32
        pipe = StableDiffusionPipeline.from_pretrained(hf_id, torch_dtype=dtype)
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder.to(device)
        vae = pipe.vae.to(device)
        unet = pipe.unet.to(device)
        noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        # Freeze VAE and text encoder
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        if use_lora:
            print(f"  Applying LoRA (rank={lora_rank})...")
            try:
                from peft import LoraConfig, get_peft_model  # type: ignore[import-unresolved]
                lora_config = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_rank,
                    target_modules=["to_q", "to_v", "to_k", "to_out.0"],
                    lora_dropout=0.1,
                )
                unet = get_peft_model(unet, lora_config)
                trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
                total = sum(p.numel() for p in unet.parameters())
                print(f"  LoRA params: {trainable:,} / {total:,} total ({100*trainable/total:.1f}%)")
            except ImportError:
                print("  ⚠ peft not installed, training all UNet parameters")
                print("    Install with: pip install peft")

        # Create dataset
        class ImageCaptionDataset(Dataset):
            def __init__(self, data_entries, tok, img_res):
                self.entries = data_entries
                self.tokenizer = tok
                self.resolution = img_res

            def __len__(self):
                return len(self.entries)

            def __getitem__(self, idx):
                entry = self.entries[idx]
                # Load and preprocess image
                img = Image.open(entry["image_path"]).convert("RGB")
                # Use Resampling enum (Pillow 9.1+) with fallback for older versions
                try:
                    resample = Image.Resampling.LANCZOS
                except AttributeError:
                    resample = getattr(Image, 'LANCZOS', 1)  # 1 = LANCZOS in older Pillow
                img = img.resize((self.resolution, self.resolution), resample)
                # Normalize to [-1, 1] (manual conversion, no torchvision needed)
                import numpy as np
                img_array = np.array(img).astype(np.float32) / 255.0  # [0, 1]
                img_array = (img_array - 0.5) / 0.5                   # [-1, 1]
                # HWC -> CHW
                pixel_values = torch.from_numpy(img_array.transpose(2, 0, 1))

                # Tokenize caption
                tokens = self.tokenizer(
                    entry["caption"],
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                return pixel_values, tokens.input_ids.squeeze(0)

        dataset_obj = ImageCaptionDataset(valid_entries, tokenizer, res)
        dataloader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in unet.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Training loop
        print("  Starting training...\n")
        vae.eval()
        text_encoder.eval()
        unet.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            steps = 0
            for pixel_values, input_ids in dataloader:
                pixel_values = pixel_values.to(device)
                input_ids = input_ids.to(device)

                # Encode images to latent space
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Loss
                loss = F.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                steps += 1

            avg_loss = epoch_loss / max(steps, 1)
            print(f"  Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")

        # Save
        print(f"\n  Saving model to {output_dir}...")
        unet_path = os.path.join(output_dir, "unet")
        unet.save_pretrained(unet_path)

        # Save training metadata
        meta = {
            "base_model": model_name,
            "hf_id": hf_id,
            "dataset": dataset_path,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "resolution": res,
            "use_lora": use_lora,
            "lora_rank": lora_rank if use_lora else None,
            "num_images": len(valid_entries),
            "trained_at": datetime.now().isoformat(),
        }
        with open(os.path.join(output_dir, "training_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"  ✓ Training complete! Model saved to {output_dir}")
        return output_dir

    except Exception as e:
        print(f"  ✗ Training error: {e}")
        return None


# ── Interactive Menu ─────────────────────────────────────────────────────

def interactive_image_gen():
    """Interactive image generation menu — generate, train, or browse models.

    Main entry point for the 'image-gen' command in run.py.
    """
    while True:
        print("\n" + "=" * 55)
        print("       Image Generation")
        print("=" * 55)

        # Check dependencies
        deps = _check_dependencies()
        if not deps.get("diffusers"):
            print("\n  ⚠ diffusers package not installed")
            print("    Generation and training require: pip install diffusers[torch] Pillow")
            print("    You can still browse models and prepare datasets.\n")

        print("  " + "-" * 50)
        print("  1   generate     Generate images from a text prompt")
        print("  2   train        Fine-tune a model on tagged images")
        print("  3   models       List available image generation models")
        print("  4   tags         Tag processing tools & dataset prep")
        print("  5   status       Check dependencies and model status")
        print("  0   back         Return to main menu")
        print()

        try:
            choice = input("  image>> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Returning to menu...")
            break

        if choice in ('0', 'back', 'quit', 'exit', 'q'):
            break

        if choice in ('1', 'generate'):
            _interactive_generate()
        elif choice in ('2', 'train'):
            _interactive_train()
        elif choice in ('3', 'models'):
            list_image_models()
        elif choice in ('4', 'tags'):
            _interactive_tags()
        elif choice in ('5', 'status'):
            _show_status()
        else:
            print(f"  Unknown option: '{choice}'")


def _interactive_generate():
    """Interactive image generation sub-menu."""
    if not _require_dependencies():
        return

    list_image_models()

    try:
        print("  Type 'back' or '0' to cancel.")
        sel = input("  Select model (number or name) [1]: ").strip()
        if not sel:
            sel = "1"
        if sel.lower() in ('back', '0', 'quit', 'exit', 'q'):
            return

        # Find model
        try:
            number = int(sel)
            result = get_image_model_by_number(number)
        except ValueError:
            result = get_image_model_by_name(sel)

        if result is None:
            print(f"  ✗ Model not found: {sel}")
            return

        model_name, model_info = result

        prompt = input("  Prompt: ").strip()
        if not prompt or prompt.lower() in ('back', '0', 'quit', 'exit', 'q'):
            return

        neg = input("  Negative prompt (optional): ").strip()
        negative_prompt = neg if neg and neg.lower() not in ('back', '0') else None

        num_str = input("  Number of images [1]: ").strip()
        num_images = int(num_str) if num_str and num_str.isdigit() else 1

        steps_str = input("  Inference steps [50]: ").strip()
        steps = int(steps_str) if steps_str and steps_str.isdigit() else 50

        seed_str = input("  Seed (optional, for reproducibility): ").strip()
        seed = int(seed_str) if seed_str and seed_str.isdigit() else None

        # Detect device
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        generate_image(
            prompt=prompt,
            model_name=model_name,
            num_images=num_images,
            steps=steps,
            negative_prompt=negative_prompt,
            seed=seed,
            device=device,
        )

    except (KeyboardInterrupt, EOFError):
        print("\n  Generation cancelled.")
    except ValueError as e:
        print(f"  Invalid input: {e}")


def _interactive_train():
    """Interactive image model training sub-menu."""
    if not _require_dependencies(["diffusers", "pillow", "torch", "transformers"]):
        return

    list_image_models()

    try:
        print("  Type 'back' or '0' to cancel at any prompt.")
        sel = input("  Base model (number or name) [1]: ").strip()
        if not sel:
            sel = "1"
        if sel.lower() in ('back', '0', 'quit', 'exit', 'q'):
            return

        try:
            number = int(sel)
            result = get_image_model_by_number(number)
        except ValueError:
            result = get_image_model_by_name(sel)

        if result is None:
            print(f"  ✗ Model not found: {sel}")
            return

        model_name, model_info = result

        dataset = input(f"  Dataset JSON path: ").strip()
        if not dataset or dataset.lower() in ('back', '0', 'quit', 'exit', 'q'):
            return

        epochs_str = input("  Epochs [5]: ").strip()
        if epochs_str.lower() in ('back', '0', 'quit', 'exit', 'q'):
            return
        epochs = int(epochs_str) if epochs_str and epochs_str.isdigit() else 5

        lora_str = input("  Use LoRA fine-tuning? [Y/n]: ").strip().lower()
        use_lora = lora_str != 'n'

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        train_image_model(
            model_name=model_name,
            dataset_path=dataset,
            epochs=epochs,
            device=device,
            use_lora=use_lora,
        )

    except (KeyboardInterrupt, EOFError):
        print("\n  Training cancelled.")
    except ValueError as e:
        print(f"  Invalid input: {e}")


def _interactive_tags():
    """Interactive tag processing and dataset preparation."""
    while True:
        print("\n  ── Tag Processing Tools ──")
        print("  " + "-" * 45)
        print("  1   parse       Parse a tag file into a dataset")
        print("  2   normalize   Normalize a tag string (preview)")
        print("  3   formats     Show supported tag formats")
        print("  0   back        Return to image gen menu")
        print()

        try:
            choice = input("  tags>> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            break

        if choice in ('0', 'back', 'quit', 'exit', 'q'):
            break

        if choice in ('1', 'parse'):
            _interactive_parse_tags()
        elif choice in ('2', 'normalize'):
            _interactive_normalize_tags()
        elif choice in ('3', 'formats'):
            _show_tag_formats()
        else:
            print(f"  Unknown option: '{choice}'")


def _interactive_parse_tags():
    """Parse a tag file and create a dataset."""
    try:
        filepath = input("  Tag file path (.json, .txt, .csv): ").strip()
        if not filepath or filepath.lower() in ('back', '0'):
            return

        print("\n  Tag format options:")
        for i, (name, fmt) in enumerate(TAG_SITE_FORMATS.items(), 1):
            print(f"    {i}. {name:<12} — {fmt['desc']}")
        fmt_sel = input("  Format [1=danbooru]: ").strip()
        fmt_names = list(TAG_SITE_FORMATS.keys())
        try:
            idx = int(fmt_sel) - 1 if fmt_sel else 0
            site_format = fmt_names[idx]
        except (ValueError, IndexError):
            site_format = "danbooru"

        entries = parse_tag_file(filepath, site_format)
        if not entries:
            print("  No entries parsed.")
            return

        print(f"\n  Parsed {len(entries)} entries. Preview:")
        for e in entries[:3]:
            print(f"    Image: {e['image_path'][:40] or '(none)'}")
            print(f"    Tags:  {e['caption'][:60]}")
            print()

        output = input(f"  Save dataset to [{DEFAULT_DATASET_DIR}/dataset.json]: ").strip()
        if not output:
            output = os.path.join(DEFAULT_DATASET_DIR, "dataset.json")

        create_tag_dataset(entries, output, site_format)

    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")


def _interactive_normalize_tags():
    """Preview tag normalization."""
    try:
        tags = input("  Enter tags: ").strip()
        if not tags:
            return

        print("\n  Normalized by format:")
        for name in TAG_SITE_FORMATS:
            result = normalize_tags(tags, name)
            print(f"    {name:<12} → {result}")

    except (KeyboardInterrupt, EOFError):
        pass


def _show_tag_formats():
    """Show supported tag site formats."""
    print("\n  ── Supported Tag Formats ──")
    print("  " + "-" * 55)
    for name, fmt in TAG_SITE_FORMATS.items():
        print(f"\n  {name}:")
        print(f"    {fmt['desc']}")
        print(f"    Separator: '{fmt['separator']}'")
        print(f"    Underscore→space: {fmt['underscore_to_space']}")
        print(f"    Strip parens: {fmt.get('strip_parentheses', False)}")
    print()


def _show_status():
    """Show image generation status and dependencies."""
    deps = _check_dependencies()

    print("\n  ── Image Generation Status ──")
    print("  " + "-" * 45)

    for pkg, installed in deps.items():
        status = "✓ installed" if installed else "✗ not installed"
        print(f"  {pkg:<15} {status}")

    # Check output directory
    if os.path.exists(DEFAULT_OUTPUT_DIR):
        images = [f for f in os.listdir(DEFAULT_OUTPUT_DIR) if f.endswith('.png')]
        print(f"\n  Generated images: {len(images)} in {DEFAULT_OUTPUT_DIR}/")
    else:
        print(f"\n  Output directory: {DEFAULT_OUTPUT_DIR}/ (not created yet)")

    # Check datasets
    if os.path.exists(DEFAULT_DATASET_DIR):
        datasets = [f for f in os.listdir(DEFAULT_DATASET_DIR) if f.endswith('.json')]
        print(f"  Training datasets: {len(datasets)} in {DEFAULT_DATASET_DIR}/")
    else:
        print(f"  Dataset directory: {DEFAULT_DATASET_DIR}/ (not created yet)")

    # Check for trained image models
    img_ckpt = "checkpoints/image_model"
    if os.path.exists(img_ckpt):
        print(f"  Trained image model: found in {img_ckpt}/")
    else:
        print(f"  Trained image model: none yet")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_mem
            print(f"  GPU Memory: {mem / 1e9:.1f} GB")
            print("  ✓ GPU available for image generation")
        else:
            print(f"\n  GPU: Not available (CPU only)")
            print("  ⚠ Image generation on CPU is very slow")
    except Exception:
        print(f"\n  GPU: Check failed")

    print()

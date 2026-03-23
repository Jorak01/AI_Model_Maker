"""Image Auto Trainer — Automated tag-based training data collection for image generation models.

Searches public sources for image tags, prompt collections, and art style
descriptions, then builds tag-based training datasets automatically.

Sources:
  - Danbooru tag wiki (public API, no key needed)
  - Web search for prompt engineering collections
  - Wikipedia art styles, techniques, movements
  - Public prompt repositories and galleries

Usage:
    # Interactive
    python image_auto_trainer.py

    # Programmatic
    from image_auto_trainer import auto_collect_tags
    dataset_path = auto_collect_tags(
        styles=["anime", "landscape", "portrait"],
        model_name="my-art-model",
    )

    # CLI
    python image_auto_trainer.py --styles "anime" "cyberpunk" --name my-model
"""

import os
import re
import json
import time
import random
from typing import Optional, List, Dict, Tuple
from datetime import datetime

from utils.web_collector import (
    _safe_get, _clean_text, search_wikipedia, fetch_wikipedia_article,
    search_duckduckgo, fetch_url_text, WebCollector, save_collected_data,
)
from image_gen import (
    normalize_tags, create_tag_dataset, TAG_SITE_FORMATS, IMAGE_GEN_MODELS,
    DEFAULT_DATASET_DIR, train_image_model,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DANBOORU_API = "https://danbooru.donmai.us"

# Common quality/meta tags used across diffusion model prompts
QUALITY_TAGS = [
    "masterpiece", "best quality", "high resolution", "detailed",
    "sharp focus", "professional", "8k", "uhd", "photorealistic",
    "highly detailed", "intricate details", "award winning",
]

NEGATIVE_QUALITY_TAGS = [
    "low quality", "worst quality", "blurry", "bad anatomy",
    "deformed", "disfigured", "watermark", "signature", "text",
    "cropped", "out of frame", "duplicate", "ugly",
]

# Style categories with associated search terms and tag seeds
STYLE_CATEGORIES: Dict[str, Dict] = {
    "anime": {
        "search_terms": ["anime art style tags", "anime illustration prompts"],
        "wiki_topics": ["Anime", "Manga", "Anime art style"],
        "seed_tags": [
            "anime style", "cel shading", "vibrant colors", "large eyes",
            "dynamic pose", "school uniform", "cherry blossom", "sakura",
            "detailed hair", "sparkle effects", "pastel colors",
        ],
        "danbooru_tags": ["anime", "anime_style", "1girl", "1boy"],
    },
    "landscape": {
        "search_terms": ["landscape photography prompts", "landscape art tags"],
        "wiki_topics": ["Landscape painting", "Landscape photography"],
        "seed_tags": [
            "mountain landscape", "sunset", "golden hour", "dramatic sky",
            "rolling hills", "misty forest", "ocean waves", "reflection",
            "seasonal colors", "panoramic view", "natural lighting",
        ],
        "danbooru_tags": ["scenery", "landscape", "nature", "sky"],
    },
    "portrait": {
        "search_terms": ["portrait photography prompts", "portrait art styles"],
        "wiki_topics": ["Portrait painting", "Portrait photography"],
        "seed_tags": [
            "portrait", "close-up face", "studio lighting", "rembrandt lighting",
            "soft bokeh background", "natural expression", "dramatic shadow",
            "eye contact", "detailed skin texture", "professional headshot",
        ],
        "danbooru_tags": ["portrait", "face", "looking_at_viewer"],
    },
    "cyberpunk": {
        "search_terms": ["cyberpunk art prompts", "cyberpunk aesthetic tags"],
        "wiki_topics": ["Cyberpunk", "Cyberpunk derivatives"],
        "seed_tags": [
            "cyberpunk", "neon lights", "rain-soaked streets", "holographic",
            "futuristic city", "augmented reality", "chrome implants",
            "dark alley", "night city", "glowing signs", "blade runner style",
        ],
        "danbooru_tags": ["cyberpunk", "neon_lights", "futuristic", "sci-fi"],
    },
    "fantasy": {
        "search_terms": ["fantasy art prompts", "fantasy illustration tags"],
        "wiki_topics": ["Fantasy art", "Fantasy illustration"],
        "seed_tags": [
            "fantasy", "magical forest", "dragon", "medieval castle",
            "enchanted", "ethereal glow", "wizard", "ancient ruins",
            "mythical creature", "sword and sorcery", "epic scene",
        ],
        "danbooru_tags": ["fantasy", "magic", "dragon", "castle"],
    },
    "realistic": {
        "search_terms": ["photorealistic AI art prompts", "realistic rendering tags"],
        "wiki_topics": ["Photorealism", "Hyperrealism (visual arts)"],
        "seed_tags": [
            "photorealistic", "ultra detailed", "natural lighting",
            "depth of field", "film grain", "DSLR quality", "sharp focus",
            "cinematic composition", "35mm photograph", "raw photo",
        ],
        "danbooru_tags": ["realistic", "photo_(medium)", "photorealistic"],
    },
    "abstract": {
        "search_terms": ["abstract art prompts", "abstract painting styles"],
        "wiki_topics": ["Abstract art", "Abstract expressionism"],
        "seed_tags": [
            "abstract", "geometric shapes", "color field", "fluid art",
            "fractal patterns", "minimalist", "bold colors",
            "texture heavy", "non-representational", "modern art",
        ],
        "danbooru_tags": ["abstract", "no_humans", "colorful"],
    },
    "watercolor": {
        "search_terms": ["watercolor painting prompts", "watercolor art style"],
        "wiki_topics": ["Watercolor painting"],
        "seed_tags": [
            "watercolor painting", "soft edges", "transparent washes",
            "wet on wet", "bleeding colors", "paper texture",
            "delicate brushstrokes", "pastel palette", "floral subject",
        ],
        "danbooru_tags": ["watercolor_(medium)", "traditional_media"],
    },
    "oil-painting": {
        "search_terms": ["oil painting prompts", "oil painting techniques"],
        "wiki_topics": ["Oil painting", "Impasto"],
        "seed_tags": [
            "oil painting", "thick impasto", "rich colors", "canvas texture",
            "classical style", "chiaroscuro", "glazing technique",
            "visible brushstrokes", "old master style", "baroque lighting",
        ],
        "danbooru_tags": ["oil_painting_(medium)", "traditional_media"],
    },
    "pixel-art": {
        "search_terms": ["pixel art prompts", "pixel art style guide"],
        "wiki_topics": ["Pixel art"],
        "seed_tags": [
            "pixel art", "16-bit style", "retro game", "sprite art",
            "limited palette", "dithering", "low resolution aesthetic",
            "8-bit colors", "tile-based", "nostalgic gaming",
        ],
        "danbooru_tags": ["pixel_art", "lowres", "retro_artstyle"],
    },
}


# ---------------------------------------------------------------------------
# Danbooru Public API (no key needed for read-only tag info)
# ---------------------------------------------------------------------------

def fetch_danbooru_tag_info(tag: str) -> Dict:
    """Fetch tag information from Danbooru's public API.

    Returns dict with tag name, count, category, and related tags.
    """
    url = f"{DANBOORU_API}/tags.json"
    params = {"search[name_matches]": tag, "limit": "1"}
    raw = _safe_get(url, params, timeout=10)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
    except (json.JSONDecodeError, IndexError):
        pass
    return {}


def fetch_danbooru_related_tags(tag: str, limit: int = 20) -> List[str]:
    """Fetch tags related to a given tag from Danbooru."""
    url = f"{DANBOORU_API}/related_tag.json"
    params = {"query": tag, "limit": str(limit)}
    raw = _safe_get(url, params, timeout=10)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        related = data.get("related_tags", [])
        return [t.get("tag", {}).get("name", "") for t in related
                if t.get("tag", {}).get("name")][:limit]
    except (json.JSONDecodeError, KeyError):
        return []


def fetch_danbooru_popular_tags(category: int = 0,
                                limit: int = 50) -> List[str]:
    """Fetch popular tags from Danbooru.

    Categories: 0=general, 1=artist, 3=copyright, 4=character, 5=meta
    """
    url = f"{DANBOORU_API}/tags.json"
    params = {
        "search[category]": str(category),
        "search[order]": "count",
        "limit": str(limit),
    }
    raw = _safe_get(url, params, timeout=10)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return [t.get("name", "") for t in data if t.get("name")]
    except (json.JSONDecodeError, KeyError):
        return []


# ---------------------------------------------------------------------------
# Tag Collection from Public Sources
# ---------------------------------------------------------------------------

def collect_tags_from_danbooru(style_tags: List[str],
                               max_tags: int = 100,
                               verbose: bool = True) -> List[str]:
    """Collect related tags from Danbooru for given style tags."""
    all_tags: set = set()
    for tag in style_tags:
        if verbose:
            print(f"  [Danbooru] Fetching related tags for: {tag}")
        related = fetch_danbooru_related_tags(tag, limit=30)
        for r in related:
            clean = r.replace("_", " ").strip()
            if clean and len(clean) > 1:
                all_tags.add(clean)
        if len(all_tags) >= max_tags:
            break
        time.sleep(0.3)  # polite delay
    return list(all_tags)[:max_tags]


def collect_tags_from_web(search_terms: List[str],
                          max_tags: int = 100,
                          verbose: bool = True) -> List[str]:
    """Search the web for prompt/tag collections and extract tags."""
    all_tags: set = set()

    for term in search_terms:
        if verbose:
            print(f"  [Web] Searching: {term}")
        results = search_duckduckgo(term, max_results=5)

        for r in results[:3]:
            snippet = r.get("snippet", "")
            if snippet:
                # Extract comma-separated items that look like tags
                potential_tags = [t.strip() for t in snippet.split(",")]
                for tag in potential_tags:
                    clean = tag.strip().lower()
                    # Filter: only keep short, tag-like phrases
                    if 2 < len(clean) < 50 and not clean.startswith("http"):
                        all_tags.add(clean)

            # Fetch page content for richer tag extraction
            url = r.get("url", "")
            if url and url.startswith("http"):
                text = fetch_url_text(url, max_chars=5000)
                if text:
                    extracted = _extract_tags_from_text(text)
                    all_tags.update(extracted)

        if len(all_tags) >= max_tags:
            break

    return list(all_tags)[:max_tags]


def collect_tags_from_wikipedia(wiki_topics: List[str],
                                max_tags: int = 50,
                                verbose: bool = True) -> List[str]:
    """Extract art-related tags from Wikipedia articles."""
    all_tags: set = set()

    for topic in wiki_topics:
        if verbose:
            print(f"  [Wikipedia] Fetching: {topic}")
        text = fetch_wikipedia_article(topic, max_chars=8000)
        if text:
            extracted = _extract_art_terms(text)
            all_tags.update(extracted)

    return list(all_tags)[:max_tags]


def _extract_tags_from_text(text: str) -> set:
    """Extract tag-like phrases from general text."""
    tags = set()
    # Look for comma-separated lists
    lines = text.split("\n")
    for line in lines:
        if "," in line and len(line) < 500:
            parts = line.split(",")
            if len(parts) >= 3:  # Likely a tag list
                for p in parts:
                    clean = p.strip().lower()
                    clean = re.sub(r"[^\w\s-]", "", clean).strip()
                    if 2 < len(clean) < 40:
                        tags.add(clean)
    return tags


def _extract_art_terms(text: str) -> set:
    """Extract art-related terms from text (Wikipedia articles, etc.)."""
    terms = set()
    # Common art-related patterns
    art_patterns = [
        r"(?:style|technique|method|approach) (?:called|known as|named) [\"']?(\w[\w\s-]{2,30})[\"']?",
        r"(\w[\w\s-]{2,25}) (?:style|technique|movement|school|period)",
        r"characterized by ([\w\s,]{5,60})",
        r"features? (?:such as|including) ([\w\s,]{5,80})",
    ]
    for pattern in art_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Split compound matches on commas
            for part in match.split(","):
                clean = part.strip().lower()
                if 2 < len(clean) < 40:
                    terms.add(clean)
    return terms


# ---------------------------------------------------------------------------
# Tag → Prompt/Caption Generation
# ---------------------------------------------------------------------------

def build_tag_captions(collected_tags: List[str],
                       style_name: str,
                       seed_tags: List[str],
                       num_captions: int = 100,
                       tags_per_caption: int = 8,
                       include_quality: bool = True) -> List[Dict]:
    """Build diverse image captions from collected tags.

    Creates varied combinations of tags suitable for training image models.
    Each caption is a comma-separated tag string.

    Args:
        collected_tags: All tags collected from various sources
        style_name: Style category name
        seed_tags: Core tags that define the style
        num_captions: How many caption entries to generate
        tags_per_caption: Average tags per caption
        include_quality: Whether to prepend quality tags

    Returns:
        List of {"image_path": "", "caption": str, "raw_tags": str} dicts
    """
    entries: List[Dict] = []
    all_tags = list(set(collected_tags + seed_tags))

    if not all_tags:
        return entries

    # Remove duplicates and very short tags
    all_tags = [t for t in all_tags if len(t.strip()) > 2]

    for i in range(num_captions):
        # Select a random subset of tags
        n_tags = max(3, min(len(all_tags), tags_per_caption + random.randint(-2, 3)))
        selected = random.sample(all_tags, min(n_tags, len(all_tags)))

        # Always include at least one seed tag for style consistency
        if seed_tags and not any(s in selected for s in seed_tags):
            selected[0] = random.choice(seed_tags)

        # Optionally prepend quality tags
        if include_quality:
            n_quality = random.randint(1, 3)
            quality = random.sample(QUALITY_TAGS, min(n_quality, len(QUALITY_TAGS)))
            selected = quality + selected

        caption = ", ".join(selected)
        raw = caption

        entries.append({
            "image_path": "",  # No actual images — caption-only dataset
            "caption": caption,
            "raw_tags": raw,
        })

    return entries


def build_negative_prompts(num_entries: int = 20) -> List[Dict]:
    """Build negative prompt training entries."""
    entries = []
    for _ in range(num_entries):
        n = random.randint(3, 7)
        selected = random.sample(NEGATIVE_QUALITY_TAGS,
                                 min(n, len(NEGATIVE_QUALITY_TAGS)))
        caption = ", ".join(selected)
        entries.append({
            "image_path": "",
            "caption": f"[negative] {caption}",
            "raw_tags": caption,
        })
    return entries


# ---------------------------------------------------------------------------
# High-level auto collection
# ---------------------------------------------------------------------------

def auto_collect_tags(
    styles: List[str],
    model_name: str = "auto-image",
    max_tags_per_style: int = 100,
    num_captions: int = 150,
    sources: Optional[List[str]] = None,
    include_quality: bool = True,
    include_negatives: bool = True,
    output_dir: str = DEFAULT_DATASET_DIR,
    verbose: bool = True,
) -> str:
    """Collect tags from public sources and build an image training dataset.

    Args:
        styles: List of style categories (see STYLE_CATEGORIES keys)
                or custom style names to search for
        model_name: Name for the output dataset
        max_tags_per_style: Max tags to collect per style
        num_captions: Number of caption entries to generate per style
        sources: Which sources to use ["danbooru", "web", "wikipedia"]
                 Default: all available
        include_quality: Add quality booster tags to captions
        include_negatives: Include negative prompt examples
        output_dir: Where to save the dataset
        verbose: Print progress

    Returns:
        Path to the saved dataset JSON file
    """
    if sources is None:
        sources = ["danbooru", "web", "wikipedia"]

    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in model_name.lower().strip()
    )

    print("\n" + "=" * 60)
    print("       Image Auto Trainer — Tag Collection")
    print("=" * 60)
    print(f"\n  Model name : {model_name}")
    print(f"  Styles     : {styles}")
    print(f"  Sources    : {sources}")
    print(f"  Captions   : {num_captions} per style")
    print()

    all_entries: List[Dict] = []

    for style in styles:
        if verbose:
            print(f"\n  {'='*50}")
            print(f"  Collecting tags for style: {style}")
            print(f"  {'='*50}")

        # Get style config (or build one for custom styles)
        style_cfg = STYLE_CATEGORIES.get(style.lower(), {
            "search_terms": [f"{style} art prompts", f"{style} image generation tags"],
            "wiki_topics": [style.title()],
            "seed_tags": [style.lower()],
            "danbooru_tags": [style.lower().replace(" ", "_")],
        })

        collected_tags: List[str] = list(style_cfg.get("seed_tags", []))

        # Collect from Danbooru
        if "danbooru" in sources:
            danbooru_seeds = style_cfg.get("danbooru_tags", [])
            if danbooru_seeds:
                tags = collect_tags_from_danbooru(
                    danbooru_seeds, max_tags=max_tags_per_style, verbose=verbose
                )
                collected_tags.extend(tags)
                if verbose:
                    print(f"    Danbooru: {len(tags)} tags")

        # Collect from web search
        if "web" in sources:
            search_terms = style_cfg.get("search_terms", [])
            if search_terms:
                tags = collect_tags_from_web(
                    search_terms, max_tags=max_tags_per_style, verbose=verbose
                )
                collected_tags.extend(tags)
                if verbose:
                    print(f"    Web: {len(tags)} tags")

        # Collect from Wikipedia
        if "wikipedia" in sources:
            wiki_topics = style_cfg.get("wiki_topics", [])
            if wiki_topics:
                tags = collect_tags_from_wikipedia(
                    wiki_topics, max_tags=max_tags_per_style // 2, verbose=verbose
                )
                collected_tags.extend(tags)
                if verbose:
                    print(f"    Wikipedia: {len(tags)} tags")

        # Deduplicate
        collected_tags = list(set(collected_tags))
        if verbose:
            print(f"\n    Total unique tags for '{style}': {len(collected_tags)}")
            if collected_tags:
                preview = ", ".join(collected_tags[:10])
                print(f"    Preview: {preview}...")

        # Build captions
        entries = build_tag_captions(
            collected_tags=collected_tags,
            style_name=style,
            seed_tags=style_cfg.get("seed_tags", []),
            num_captions=num_captions,
            include_quality=include_quality,
        )
        all_entries.extend(entries)
        if verbose:
            print(f"    Generated {len(entries)} caption entries")

    # Add negative prompts
    if include_negatives:
        neg_entries = build_negative_prompts(num_entries=min(30, len(all_entries) // 5))
        all_entries.extend(neg_entries)
        if verbose:
            print(f"\n  Added {len(neg_entries)} negative prompt examples")

    if not all_entries:
        print("\n  ❌ No tag data collected.")
        return ""

    # Save dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{safe_name}_tags.json")
    create_tag_dataset(all_entries, output_path, site_format="custom")

    # Also save a simple tag list for reference
    tag_list_path = os.path.join(output_dir, f"{safe_name}_taglist.txt")
    unique_tags = set()
    for entry in all_entries:
        for tag in entry["caption"].split(","):
            clean = tag.strip()
            if clean and not clean.startswith("[negative]"):
                unique_tags.add(clean)
    with open(tag_list_path, "w", encoding="utf-8") as f:
        for tag in sorted(unique_tags):
            f.write(tag + "\n")

    print(f"\n  ✓ Dataset: {output_path} ({len(all_entries)} entries)")
    print(f"  ✓ Tag list: {tag_list_path} ({len(unique_tags)} unique tags)")

    return output_path


# ---------------------------------------------------------------------------
# Full auto train: collect + train
# ---------------------------------------------------------------------------

def auto_train_image(
    styles: List[str],
    model_name: str = "auto-image",
    base_model: str = "stable-diffusion-v1-5",
    max_tags_per_style: int = 100,
    num_captions: int = 150,
    sources: Optional[List[str]] = None,
    epochs: int = 5,
    learning_rate: float = 1e-5,
    use_lora: bool = True,
    device: str = "cpu",
    verbose: bool = True,
) -> Optional[str]:
    """Collect tags, build dataset, and train an image model end-to-end.

    Note: Actual image model training requires images paired with captions.
    This function collects the caption/tag dataset. For full training,
    you'll need to pair captions with actual images.

    Args:
        styles: Style categories to train on
        model_name: Name for the trained model
        base_model: Base diffusion model to fine-tune
        All other args match auto_collect_tags and train_image_model

    Returns:
        Path to the dataset (training requires paired images)
    """
    # Step 1: Collect tags and build dataset
    dataset_path = auto_collect_tags(
        styles=styles,
        model_name=model_name,
        max_tags_per_style=max_tags_per_style,
        num_captions=num_captions,
        sources=sources,
        verbose=verbose,
    )

    if not dataset_path:
        return None

    print("\n" + "=" * 60)
    print("  Tag Dataset Created Successfully!")
    print("=" * 60)
    print(f"\n  Dataset: {dataset_path}")
    print(f"\n  To train an image model, you need to:")
    print(f"  1. Pair these captions with actual training images")
    print(f"  2. Update the image_path fields in the dataset JSON")
    print(f"  3. Use 'image-gen > train' to fine-tune a model")
    print(f"\n  The captions are ready for diffusion model training")
    print(f"  using {base_model} or any compatible model.\n")

    return dataset_path


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def auto_image_train_interactive():
    """Interactive auto image training flow."""
    print("\n" + "=" * 60)
    print("       Image Auto Trainer — Public Tag Collection")
    print("=" * 60)
    print("\n  Automatically collect image tags and prompts from public")
    print("  sources to build training datasets for image generation models.")

    # 1. Show available styles
    print("\n  Available style presets:")
    style_names = list(STYLE_CATEGORIES.keys())
    for i, name in enumerate(style_names, 1):
        desc = STYLE_CATEGORIES[name].get("seed_tags", [""])[0]
        print(f"    {i:>2}. {name:<16} (e.g., {desc})")
    print(f"    Or enter custom style names separated by commas.")

    # 2. Select styles
    try:
        print(f"\n  Enter styles (numbers, names, or custom — comma-separated):")
        raw = input("  Styles: ").strip()
        if not raw:
            print("  Cancelled.")
            return
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        return

    styles: List[str] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part) - 1
            if 0 <= idx < len(style_names):
                styles.append(style_names[idx])
            else:
                print(f"  ⚠ Invalid number: {part}")
        except ValueError:
            styles.append(part.lower())

    if not styles:
        print("  No styles selected. Cancelled.")
        return

    print(f"\n  Selected styles: {', '.join(styles)}")

    # 3. Model name
    try:
        default_name = f"{'_'.join(styles[:3])}-model"
        print(f"\n  Dataset name? (default: {default_name})")
        model_name = input("  Name: ").strip() or default_name
    except (KeyboardInterrupt, EOFError):
        model_name = f"{'_'.join(styles[:3])}-model"

    # 4. Sources
    try:
        print("\n  Data sources:")
        print("  1. All sources (Danbooru + Web + Wikipedia) [default]")
        print("  2. Danbooru only (tag-focused)")
        print("  3. Web search only")
        print("  4. Wikipedia only (art theory)")
        src_choice = input("  [1/2/3/4]: ").strip()
        source_map = {
            "1": ["danbooru", "web", "wikipedia"],
            "2": ["danbooru"],
            "3": ["web"],
            "4": ["wikipedia"],
        }
        sources = source_map.get(src_choice, ["danbooru", "web", "wikipedia"])
    except (KeyboardInterrupt, EOFError):
        sources = ["danbooru", "web", "wikipedia"]

    # 5. Number of captions
    try:
        print(f"\n  Captions per style? (default=150)")
        nc = input("  Captions: ").strip()
        num_captions = int(nc) if nc else 150
    except (KeyboardInterrupt, EOFError, ValueError):
        num_captions = 150

    # 6. Mode selection
    try:
        print("\n  Mode:")
        print("  1. Collect tags & build dataset (default)")
        print("  2. Collect tags & train image model")
        mode = input("  [1/2]: ").strip()
    except (KeyboardInterrupt, EOFError):
        mode = "1"

    if mode == "2":
        # Select base model
        try:
            print("\n  Base image model:")
            models = list(IMAGE_GEN_MODELS.keys())
            for i, m in enumerate(models, 1):
                print(f"    {i}. {m}")
            sel = input(f"  [1-{len(models)}, default=1]: ").strip()
            idx = int(sel) - 1 if sel else 0
            base_model = models[idx] if 0 <= idx < len(models) else models[0]
        except (KeyboardInterrupt, EOFError, ValueError, IndexError):
            base_model = "stable-diffusion-v1-5"

        auto_train_image(
            styles=styles,
            model_name=model_name,
            base_model=base_model,
            num_captions=num_captions,
            sources=sources,
        )
    else:
        auto_collect_tags(
            styles=styles,
            model_name=model_name,
            num_captions=num_captions,
            sources=sources,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Image Auto Trainer — Collect tags for image model training"
    )
    parser.add_argument(
        "--styles", nargs="+",
        help=f"Styles to collect. Presets: {', '.join(STYLE_CATEGORIES.keys())}",
    )
    parser.add_argument("--name", default="auto-image", help="Dataset name")
    parser.add_argument(
        "--sources", nargs="+", default=None,
        help="Data sources: danbooru, web, wikipedia",
    )
    parser.add_argument("--captions", type=int, default=150, help="Captions per style")
    parser.add_argument("--max-tags", type=int, default=100, help="Max tags per style")
    parser.add_argument("--list-styles", action="store_true", help="List available styles")

    args = parser.parse_args()

    if args.list_styles:
        print("\nAvailable style presets:")
        for name, cfg in STYLE_CATEGORIES.items():
            seeds = ", ".join(cfg["seed_tags"][:3])
            print(f"  {name:<16} → {seeds}...")
        print()
    elif args.styles:
        auto_collect_tags(
            styles=args.styles,
            model_name=args.name,
            max_tags_per_style=args.max_tags,
            num_captions=args.captions,
            sources=args.sources,
        )
    else:
        auto_image_train_interactive()

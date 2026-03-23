"""Tag Manager — Frequency analysis, hierarchy/ontology, negative prompt tuning.

Features:
  - Tag frequency analysis and visualization
  - Tag hierarchy/ontology tree
  - Negative prompt tuning discovery
"""

import os
import json
import re
from typing import List, Dict, Set, Optional, Tuple
from collections import Counter


# ---------------------------------------------------------------------------
# Tag Frequency Analysis
# ---------------------------------------------------------------------------

def analyze_tag_frequency(dataset_path: str, tag_field: str = "caption") -> Dict[str, int]:
    """Analyze tag frequency from a dataset file."""
    if not os.path.exists(dataset_path):
        return {}
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    entries = data.get("entries", data) if isinstance(data, dict) else data
    counter: Counter = Counter()
    for entry in entries:
        if isinstance(entry, dict):
            text = entry.get(tag_field, entry.get("tags", ""))
        else:
            text = str(entry)
        if isinstance(text, str):
            tags = [t.strip().lower() for t in text.split(",") if t.strip()]
            counter.update(tags)
        elif isinstance(text, list):
            counter.update([t.strip().lower() for t in text if t.strip()])
    return dict(counter.most_common())


def display_frequency(freq: Dict[str, int], top_n: int = 30):
    """Display tag frequency distribution."""
    if not freq:
        print("  No tags found.")
        return
    items = sorted(freq.items(), key=lambda x: -x[1])[:top_n]
    max_count = items[0][1] if items else 1
    print(f"\n  Top {min(top_n, len(items))} Tags:")
    print("  " + "-" * 55)
    for tag, count in items:
        bar_len = int(count * 30 / max_count)
        bar = "█" * bar_len
        print(f"  {tag:<25} {bar} {count}")


def find_rare_tags(freq: Dict[str, int], threshold: int = 2) -> List[str]:
    """Find tags that appear fewer than threshold times."""
    return [tag for tag, count in freq.items() if count < threshold]


def find_overrepresented(freq: Dict[str, int], ratio: float = 0.3) -> List[str]:
    """Find tags that appear in more than ratio of all entries."""
    total = sum(freq.values())
    if total == 0:
        return []
    return [tag for tag, count in freq.items() if count / total > ratio]


# ---------------------------------------------------------------------------
# Tag Hierarchy / Ontology
# ---------------------------------------------------------------------------

TAG_HIERARCHY = {
    "color": {
        "warm": ["red", "orange", "yellow", "gold", "amber", "warm colors"],
        "cool": ["blue", "green", "purple", "teal", "cyan", "cool colors"],
        "neutral": ["black", "white", "gray", "grey", "brown", "beige"],
        "effect": ["pastel", "neon", "vibrant", "muted", "monochrome", "sepia"],
    },
    "lighting": {
        "natural": ["sunlight", "golden hour", "moonlight", "natural lighting", "daylight"],
        "studio": ["studio lighting", "rembrandt lighting", "rim lighting", "backlighting"],
        "dramatic": ["chiaroscuro", "dramatic lighting", "volumetric lighting", "god rays"],
        "soft": ["soft lighting", "ambient light", "diffused light", "overcast"],
    },
    "composition": {
        "framing": ["close-up", "wide shot", "medium shot", "portrait", "full body"],
        "angle": ["low angle", "high angle", "bird eye view", "dutch angle", "isometric"],
        "technique": ["rule of thirds", "symmetry", "leading lines", "depth of field", "bokeh"],
    },
    "style": {
        "traditional": ["oil painting", "watercolor", "pencil sketch", "charcoal", "ink"],
        "digital": ["digital art", "3d render", "cgi", "vector art", "pixel art"],
        "photographic": ["photograph", "dslr", "film grain", "35mm", "raw photo"],
        "artistic": ["impressionist", "surreal", "abstract", "minimalist", "art nouveau"],
    },
    "subject": {
        "person": ["portrait", "face", "figure", "pose", "expression"],
        "nature": ["landscape", "seascape", "mountain", "forest", "flower"],
        "urban": ["cityscape", "architecture", "street", "building", "skyline"],
        "fantasy": ["dragon", "magic", "wizard", "enchanted", "mythical"],
    },
    "quality": {
        "positive": ["masterpiece", "best quality", "detailed", "sharp focus", "professional"],
        "negative": ["low quality", "blurry", "bad anatomy", "deformed", "watermark"],
    },
}


def get_tag_category(tag: str) -> Optional[Tuple[str, str]]:
    """Find the category and subcategory for a tag."""
    tag_lower = tag.lower().strip()
    for category, subcats in TAG_HIERARCHY.items():
        for subcat, tags in subcats.items():
            if tag_lower in [t.lower() for t in tags]:
                return (category, subcat)
    return None


def categorize_tags(tags: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """Categorize a list of tags according to the hierarchy."""
    result: Dict[str, Dict[str, List[str]]] = {}
    uncategorized = []
    for tag in tags:
        cat = get_tag_category(tag)
        if cat:
            category, subcat = cat
            result.setdefault(category, {}).setdefault(subcat, []).append(tag)
        else:
            uncategorized.append(tag)
    if uncategorized:
        result["uncategorized"] = {"other": uncategorized}
    return result


def display_hierarchy():
    """Display the full tag hierarchy."""
    print("\n  Tag Hierarchy / Ontology:")
    print("  " + "=" * 50)
    for category, subcats in TAG_HIERARCHY.items():
        print(f"\n  📂 {category.upper()}")
        for subcat, tags in subcats.items():
            print(f"    📁 {subcat}: {', '.join(tags[:5])}{'...' if len(tags) > 5 else ''}")


# ---------------------------------------------------------------------------
# Negative Prompt Tuning
# ---------------------------------------------------------------------------

NEGATIVE_PROMPT_SETS = {
    "general": [
        "low quality", "worst quality", "blurry", "bad anatomy", "deformed",
        "disfigured", "watermark", "signature", "text", "cropped", "out of frame",
    ],
    "portrait": [
        "bad face", "ugly", "extra fingers", "mutated hands", "poorly drawn face",
        "mutation", "extra limbs", "malformed limbs", "bad proportions",
    ],
    "landscape": [
        "oversaturated", "overexposed", "underexposed", "flat", "boring composition",
        "unrealistic sky", "bad perspective",
    ],
    "anime": [
        "bad anatomy", "wrong proportions", "extra digits", "fewer digits",
        "bad hands", "error", "missing fingers",
    ],
}


def suggest_negatives(style: str, existing_tags: Optional[List[str]] = None) -> List[str]:
    """Suggest negative prompt tags for a given style."""
    negatives = list(NEGATIVE_PROMPT_SETS.get("general", []))
    style_negatives = NEGATIVE_PROMPT_SETS.get(style.lower(), [])
    negatives.extend(style_negatives)
    if existing_tags:
        negatives = [n for n in negatives if n not in existing_tags]
    return list(set(negatives))


# ---------------------------------------------------------------------------
# Interactive
# ---------------------------------------------------------------------------

def interactive_tag_manager():
    """Interactive tag management interface."""
    print("\n" + "=" * 55)
    print("       Tag Manager")
    print("=" * 55)

    while True:
        print("\n  Options:")
        print("  1  Analyze tag frequency")
        print("  2  Display tag hierarchy")
        print("  3  Categorize tags from file")
        print("  4  Suggest negative prompts")
        print("  5  Find rare/overrepresented tags")
        print("  0  Back")

        try:
            choice = input("\n  tags>> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if choice in ('0', 'back', 'quit', 'q'):
            break
        try:
            if choice == '1':
                path = input("  Dataset path: ").strip()
                if path:
                    freq = analyze_tag_frequency(path)
                    display_frequency(freq)
            elif choice == '2':
                display_hierarchy()
            elif choice == '3':
                path = input("  Dataset path: ").strip()
                if path:
                    freq = analyze_tag_frequency(path)
                    tags = list(freq.keys())[:50]
                    cats = categorize_tags(tags)
                    for cat, subcats in cats.items():
                        print(f"\n  {cat}:")
                        for sub, t in subcats.items():
                            print(f"    {sub}: {', '.join(t)}")
            elif choice == '4':
                style = input("  Style [general]: ").strip() or "general"
                negs = suggest_negatives(style)
                print(f"\n  Suggested negatives for '{style}':")
                print(f"  {', '.join(negs)}")
            elif choice == '5':
                path = input("  Dataset path: ").strip()
                if path:
                    freq = analyze_tag_frequency(path)
                    rare = find_rare_tags(freq)
                    over = find_overrepresented(freq)
                    print(f"  Rare tags (<2): {len(rare)}")
                    if rare[:10]:
                        print(f"    {', '.join(rare[:10])}")
                    print(f"  Overrepresented (>30%%): {len(over)}")
                    if over:
                        print(f"    {', '.join(over)}")
        except Exception as e:
            print(f"  Error: {e}")

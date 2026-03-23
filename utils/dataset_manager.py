"""Dataset Manager — Versioning, quality scoring, augmentation, deduplication.

Features:
  - Dataset snapshots with version tracking
  - Quality scoring for training pairs (coherence, diversity, noise)
  - Text augmentation (synonym swap, paraphrase, back-translate patterns)
  - Global deduplication index across datasets
"""

import os
import json
import hashlib
import random
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter


# ---------------------------------------------------------------------------
# Dataset Versioning
# ---------------------------------------------------------------------------

VERSION_DIR = "data/versions"
VERSION_INDEX = os.path.join(VERSION_DIR, "index.json")


def _load_version_index() -> Dict:
    if os.path.exists(VERSION_INDEX):
        with open(VERSION_INDEX, 'r') as f:
            return json.load(f)
    return {"versions": [], "current": None}


def _save_version_index(index: Dict):
    os.makedirs(VERSION_DIR, exist_ok=True)
    with open(VERSION_INDEX, 'w') as f:
        json.dump(index, f, indent=2)


def snapshot_dataset(data_path: str, tag: str = "", notes: str = "") -> str:
    """Create a versioned snapshot of a dataset file.

    Returns version ID.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    # Generate version ID
    content_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_id = f"v_{timestamp}_{content_hash}"

    if tag:
        version_id = f"v_{tag}_{content_hash}"

    # Save snapshot
    os.makedirs(VERSION_DIR, exist_ok=True)
    snapshot_path = os.path.join(VERSION_DIR, f"{version_id}.json")
    with open(snapshot_path, 'w') as f:
        json.dump(data, f, indent=2)

    # Update index
    index = _load_version_index()
    index["versions"].append({
        "id": version_id,
        "timestamp": datetime.now().isoformat(),
        "source": data_path,
        "entries": len(data),
        "tag": tag,
        "notes": notes,
        "hash": content_hash,
    })
    index["current"] = version_id
    _save_version_index(index)

    print(f"  ✓ Snapshot: {version_id} ({len(data)} entries)")
    return version_id


def list_versions() -> List[Dict]:
    """List all dataset versions."""
    index = _load_version_index()
    return index.get("versions", [])


def restore_version(version_id: str, target_path: str = "data/train.json") -> bool:
    """Restore a dataset from a version snapshot."""
    snapshot_path = os.path.join(VERSION_DIR, f"{version_id}.json")
    if not os.path.exists(snapshot_path):
        print(f"  ❌ Version not found: {version_id}")
        return False

    with open(snapshot_path, 'r') as f:
        data = json.load(f)

    with open(target_path, 'w') as f:
        json.dump(data, f, indent=2)

    index = _load_version_index()
    index["current"] = version_id
    _save_version_index(index)

    print(f"  ✓ Restored {version_id} → {target_path} ({len(data)} entries)")
    return True


# ---------------------------------------------------------------------------
# Quality Scoring
# ---------------------------------------------------------------------------

def score_pair(prompt: str, response: str) -> Dict[str, float]:
    """Score a single prompt-response pair for quality.

    Returns dict of scores (0-1): coherence, length, diversity, noise.
    """
    scores = {}

    # Length score — penalize very short or very long
    prompt_len = len(prompt.split())
    response_len = len(response.split())
    len_score = min(1.0, prompt_len / 5) * min(1.0, response_len / 10)
    len_score = max(0.1, len_score)
    if response_len > 500:
        len_score *= 0.7
    scores["length"] = round(len_score, 3)

    # Diversity score — unique word ratio
    words = (prompt + " " + response).lower().split()
    if words:
        unique_ratio = len(set(words)) / len(words)
        scores["diversity"] = round(unique_ratio, 3)
    else:
        scores["diversity"] = 0.0

    # Noise score — penalize excessive punctuation, URLs, special chars
    noise_patterns = [
        r'http[s]?://\S+', r'[^\w\s]{3,}', r'\b(\w)\1{3,}\b',
        r'[A-Z]{10,}', r'[\U0001F600-\U0001F64F]',
    ]
    noise_count = sum(len(re.findall(p, prompt + " " + response)) for p in noise_patterns)
    noise_score = max(0.0, 1.0 - noise_count * 0.2)
    scores["noise"] = round(noise_score, 3)

    # Coherence — basic check: response shouldn't be identical to prompt
    if prompt.strip().lower() == response.strip().lower():
        scores["coherence"] = 0.1
    elif any(response.lower().startswith(w) for w in ["i", "the", "yes", "no", "it", "that"]):
        scores["coherence"] = 0.9
    else:
        scores["coherence"] = 0.6

    # Overall score
    scores["overall"] = round(
        scores["length"] * 0.2 + scores["diversity"] * 0.3 +
        scores["noise"] * 0.2 + scores["coherence"] * 0.3, 3
    )

    return scores


def score_dataset(data: List[Dict], verbose: bool = True) -> Dict:
    """Score an entire dataset. Returns summary statistics."""
    if not data:
        return {"count": 0, "avg_score": 0.0}

    all_scores = []
    low_quality = []
    for i, item in enumerate(data):
        prompt = item.get("prompt", "")
        response = item.get("response", "")
        scores = score_pair(prompt, response)
        all_scores.append(scores)
        if scores["overall"] < 0.4:
            low_quality.append((i, scores["overall"]))

    avg_scores = {}
    for key in all_scores[0]:
        avg_scores[key] = round(sum(s[key] for s in all_scores) / len(all_scores), 3)

    result = {
        "count": len(data),
        "avg_score": avg_scores["overall"],
        "avg_scores": avg_scores,
        "low_quality_count": len(low_quality),
        "low_quality_indices": [i for i, _ in low_quality[:20]],
    }

    if verbose:
        print(f"\n  Dataset Quality Report ({len(data)} entries):")
        print("  " + "-" * 40)
        for key, val in avg_scores.items():
            bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
            print(f"  {key:<12} {bar} {val:.3f}")
        if low_quality:
            print(f"\n  ⚠ {len(low_quality)} low-quality entries (score < 0.4)")

    return result


def filter_low_quality(data: List[Dict], threshold: float = 0.4) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into good and low-quality entries."""
    good, bad = [], []
    for item in data:
        scores = score_pair(item.get("prompt", ""), item.get("response", ""))
        if scores["overall"] >= threshold:
            good.append(item)
        else:
            bad.append(item)
    return good, bad


# ---------------------------------------------------------------------------
# Text Augmentation
# ---------------------------------------------------------------------------

# Simple synonym map for augmentation
SYNONYMS = {
    "good": ["great", "excellent", "fine", "nice", "wonderful"],
    "bad": ["poor", "terrible", "awful", "horrible", "dreadful"],
    "big": ["large", "huge", "enormous", "massive", "vast"],
    "small": ["tiny", "little", "minor", "compact", "miniature"],
    "fast": ["quick", "rapid", "swift", "speedy", "prompt"],
    "slow": ["sluggish", "gradual", "unhurried", "leisurely"],
    "important": ["crucial", "vital", "essential", "significant", "key"],
    "easy": ["simple", "straightforward", "effortless", "basic"],
    "hard": ["difficult", "challenging", "tough", "complex"],
    "help": ["assist", "aid", "support", "guide"],
    "make": ["create", "build", "produce", "construct", "generate"],
    "use": ["utilize", "employ", "apply", "leverage"],
    "show": ["display", "present", "demonstrate", "reveal"],
    "think": ["believe", "consider", "suppose", "reckon"],
    "want": ["desire", "wish", "need", "require"],
    "like": ["enjoy", "prefer", "appreciate", "favor"],
    "say": ["state", "mention", "express", "declare"],
    "get": ["obtain", "acquire", "receive", "gain"],
}


def augment_synonym_swap(text: str, swap_prob: float = 0.15) -> str:
    """Replace random words with synonyms."""
    words = text.split()
    result = []
    for w in words:
        lower = w.lower().strip(".,!?;:")
        if lower in SYNONYMS and random.random() < swap_prob:
            replacement = random.choice(SYNONYMS[lower])
            if w[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement)
        else:
            result.append(w)
    return " ".join(result)


def augment_word_dropout(text: str, drop_prob: float = 0.1) -> str:
    """Randomly drop words from text."""
    words = text.split()
    if len(words) <= 3:
        return text
    result = [w for w in words if random.random() > drop_prob]
    return " ".join(result) if result else text


def augment_word_swap(text: str, swap_prob: float = 0.1) -> str:
    """Randomly swap adjacent words."""
    words = text.split()
    for i in range(len(words) - 1):
        if random.random() < swap_prob:
            words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


def augment_case_change(text: str) -> str:
    """Randomly change case of first word."""
    if not text:
        return text
    if random.random() < 0.5:
        return text[0].lower() + text[1:]
    return text


def augment_dataset(data: List[Dict], multiplier: int = 2,
                    methods: Optional[List[str]] = None) -> List[Dict]:
    """Augment a dataset by creating variations of existing entries.

    Args:
        data: Original dataset
        multiplier: How many augmented copies per original
        methods: Which augmentation methods to use
                 ["synonym", "dropout", "swap", "case"]

    Returns:
        Augmented dataset (original + new entries)
    """
    if methods is None:
        methods = ["synonym", "dropout", "swap", "case"]

    augmenters = {
        "synonym": augment_synonym_swap,
        "dropout": augment_word_dropout,
        "swap": augment_word_swap,
        "case": augment_case_change,
    }

    augmented = list(data)  # Keep originals
    for item in data:
        for _ in range(multiplier):
            method = random.choice(methods)
            func = augmenters.get(method, augment_synonym_swap)
            new_item = {
                "prompt": func(item["prompt"]),
                "response": item["response"],  # Keep response intact
                "augmented": True,
                "method": method,
            }
            augmented.append(new_item)

    print(f"  ✓ Augmented: {len(data)} → {len(augmented)} entries")
    return augmented


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _hash_pair(prompt: str, response: str) -> str:
    """Create a normalized hash for a prompt-response pair."""
    normalized = (prompt.strip().lower() + "|||" + response.strip().lower())
    return hashlib.md5(normalized.encode()).hexdigest()


DEDUP_INDEX_PATH = "data/dedup_index.json"


def load_dedup_index() -> Set[str]:
    """Load the global deduplication index."""
    if os.path.exists(DEDUP_INDEX_PATH):
        with open(DEDUP_INDEX_PATH, 'r') as f:
            return set(json.load(f))
    return set()


def save_dedup_index(index: Set[str]):
    """Save the deduplication index."""
    os.makedirs(os.path.dirname(DEDUP_INDEX_PATH), exist_ok=True)
    with open(DEDUP_INDEX_PATH, 'w') as f:
        json.dump(sorted(index), f)


def deduplicate(data: List[Dict], use_global_index: bool = True) -> List[Dict]:
    """Remove duplicate entries from a dataset.

    Args:
        data: Dataset to deduplicate
        use_global_index: Check against global dedup index

    Returns:
        Deduplicated dataset
    """
    if use_global_index:
        seen = load_dedup_index()
    else:
        seen = set()

    unique = []
    dupes = 0
    for item in data:
        h = _hash_pair(item.get("prompt", ""), item.get("response", ""))
        if h not in seen:
            seen.add(h)
            unique.append(item)
        else:
            dupes += 1

    if use_global_index:
        save_dedup_index(seen)

    if dupes:
        print(f"  ✓ Deduplicated: {len(data)} → {len(unique)} ({dupes} duplicates removed)")
    return unique


# ---------------------------------------------------------------------------
# Interactive Dataset Manager
# ---------------------------------------------------------------------------

def interactive_dataset_manager():
    """Interactive dataset management interface."""
    print("\n" + "=" * 55)
    print("       Dataset Manager")
    print("=" * 55)

    while True:
        print("\n  Options:")
        print("  1  Score dataset quality")
        print("  2  Augment dataset")
        print("  3  Deduplicate dataset")
        print("  4  Filter low-quality entries")
        print("  5  Create version snapshot")
        print("  6  List versions")
        print("  7  Restore version")
        print("  0  Back")

        try:
            choice = input("\n  dataset>> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice in ('0', 'back', 'quit', 'q'):
            break

        try:
            if choice == '1':
                path = input("  Dataset path [data/train.json]: ").strip() or "data/train.json"
                with open(path, 'r') as f:
                    data = json.load(f)
                score_dataset(data)

            elif choice == '2':
                path = input("  Dataset path [data/train.json]: ").strip() or "data/train.json"
                mult = input("  Multiplier [2]: ").strip()
                mult = int(mult) if mult else 2
                with open(path, 'r') as f:
                    data = json.load(f)
                augmented = augment_dataset(data, multiplier=mult)
                out = input(f"  Save to [{path}]: ").strip() or path
                with open(out, 'w') as f:
                    json.dump(augmented, f, indent=2)
                print(f"  ✓ Saved to {out}")

            elif choice == '3':
                path = input("  Dataset path [data/train.json]: ").strip() or "data/train.json"
                with open(path, 'r') as f:
                    data = json.load(f)
                unique = deduplicate(data)
                with open(path, 'w') as f:
                    json.dump(unique, f, indent=2)

            elif choice == '4':
                path = input("  Dataset path [data/train.json]: ").strip() or "data/train.json"
                threshold = input("  Quality threshold [0.4]: ").strip()
                threshold = float(threshold) if threshold else 0.4
                with open(path, 'r') as f:
                    data = json.load(f)
                good, bad = filter_low_quality(data, threshold)
                print(f"  Kept: {len(good)} | Removed: {len(bad)}")
                save = input("  Save filtered? [y/N]: ").strip().lower()
                if save == 'y':
                    with open(path, 'w') as f:
                        json.dump(good, f, indent=2)
                    print(f"  ✓ Saved {len(good)} entries")

            elif choice == '5':
                path = input("  Dataset path [data/train.json]: ").strip() or "data/train.json"
                tag = input("  Tag (optional): ").strip()
                notes = input("  Notes (optional): ").strip()
                snapshot_dataset(path, tag=tag, notes=notes)

            elif choice == '6':
                versions = list_versions()
                if versions:
                    print(f"\n  Dataset Versions ({len(versions)}):")
                    print("  " + "-" * 60)
                    for v in versions:
                        print(f"  {v['id']:<35} {v['entries']:>5} entries  {v.get('tag', '')}")
                else:
                    print("  No versions found.")

            elif choice == '7':
                versions = list_versions()
                if not versions:
                    print("  No versions available.")
                    continue
                for v in versions[-5:]:
                    print(f"  {v['id']}")
                vid = input("  Version ID: ").strip()
                if vid:
                    restore_version(vid)

        except (KeyboardInterrupt, EOFError):
            continue
        except Exception as e:
            print(f"  Error: {e}")

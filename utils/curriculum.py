"""Curriculum Learning — Progressive training scheduler and domain mixing.

Features:
  - Progressive difficulty: start with simple examples, increase complexity
  - Domain mixing ratios: control topic distribution per epoch
  - Automatic difficulty estimation based on text length and vocabulary
"""

import json
import random
import math
from typing import List, Dict, Tuple, Optional
from collections import Counter


# ---------------------------------------------------------------------------
# Difficulty Estimation
# ---------------------------------------------------------------------------

def estimate_difficulty(prompt: str, response: str) -> float:
    """Estimate the difficulty of a training pair (0.0 = easy, 1.0 = hard).

    Factors:
      - Combined text length (longer = harder)
      - Vocabulary complexity (unique word ratio)
      - Average word length
      - Sentence count
    """
    text = prompt + " " + response
    words = text.lower().split()
    if not words:
        return 0.0

    # Length factor (normalized to 0-1, caps at 200 words)
    length_score = min(1.0, len(words) / 200)

    # Vocabulary complexity
    unique_ratio = len(set(words)) / len(words) if words else 0
    vocab_score = unique_ratio

    # Average word length (longer words = harder)
    avg_word_len = sum(len(w) for w in words) / len(words)
    word_len_score = min(1.0, avg_word_len / 10)

    # Sentence complexity
    sentences = text.count('.') + text.count('!') + text.count('?')
    sentence_score = min(1.0, sentences / 5)

    difficulty = (
        length_score * 0.3 +
        vocab_score * 0.3 +
        word_len_score * 0.2 +
        sentence_score * 0.2
    )
    return round(min(1.0, max(0.0, difficulty)), 3)


def sort_by_difficulty(data: List[Dict], reverse: bool = False) -> List[Dict]:
    """Sort dataset by difficulty (easy first by default)."""
    scored = []
    for item in data:
        diff = estimate_difficulty(item.get("prompt", ""), item.get("response", ""))
        scored.append((diff, item))
    scored.sort(key=lambda x: x[0], reverse=reverse)
    return [item for _, item in scored]


def assign_difficulty_levels(data: List[Dict], num_levels: int = 5) -> List[Dict]:
    """Assign difficulty levels to each item in the dataset."""
    scored = []
    for item in data:
        diff = estimate_difficulty(item.get("prompt", ""), item.get("response", ""))
        scored.append((diff, item))
    scored.sort(key=lambda x: x[0])

    items_per_level = max(1, len(scored) // num_levels)
    result = []
    for i, (diff, item) in enumerate(scored):
        level = min(num_levels - 1, i // items_per_level)
        item_copy = dict(item)
        item_copy["_difficulty"] = diff
        item_copy["_level"] = level
        result.append(item_copy)
    return result


# ---------------------------------------------------------------------------
# Progressive Training Scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """Progressive training scheduler — introduces harder examples over time.

    Usage:
        scheduler = CurriculumScheduler(data, num_epochs=20)
        for epoch in range(20):
            epoch_data = scheduler.get_epoch_data(epoch)
            # train on epoch_data
    """

    def __init__(self, data: List[Dict], num_epochs: int = 10,
                 warmup_epochs: int = 3, strategy: str = "linear"):
        """
        Args:
            data: Full training dataset
            num_epochs: Total training epochs
            warmup_epochs: Epochs before all data is included
            strategy: "linear", "exponential", or "step"
        """
        self.num_epochs = num_epochs
        self.warmup_epochs = max(1, warmup_epochs)
        self.strategy = strategy

        # Sort by difficulty
        self.sorted_data = sort_by_difficulty(data)
        self.total = len(self.sorted_data)

    def _get_fraction(self, epoch: int) -> float:
        """Get the fraction of data to use at this epoch."""
        if epoch >= self.warmup_epochs:
            return 1.0

        progress = epoch / self.warmup_epochs

        if self.strategy == "linear":
            return 0.3 + 0.7 * progress
        elif self.strategy == "exponential":
            return 0.3 + 0.7 * (1 - math.exp(-3 * progress))
        elif self.strategy == "step":
            steps = [0.3, 0.5, 0.7, 0.85, 1.0]
            idx = min(int(progress * len(steps)), len(steps) - 1)
            return steps[idx]
        else:
            return 1.0

    def get_epoch_data(self, epoch: int) -> List[Dict]:
        """Get training data for a specific epoch."""
        fraction = self._get_fraction(epoch)
        n_items = max(1, int(self.total * fraction))
        epoch_data = self.sorted_data[:n_items]
        random.shuffle(epoch_data)
        return epoch_data

    def get_schedule_info(self) -> List[Dict]:
        """Preview the training schedule."""
        info = []
        for epoch in range(self.num_epochs):
            fraction = self._get_fraction(epoch)
            n_items = max(1, int(self.total * fraction))
            info.append({
                "epoch": epoch + 1,
                "fraction": round(fraction, 2),
                "items": n_items,
            })
        return info


# ---------------------------------------------------------------------------
# Domain Mixing
# ---------------------------------------------------------------------------

def detect_domain(text: str) -> str:
    """Simple domain detection based on keywords."""
    text_lower = text.lower()

    domain_keywords = {
        "code": ["code", "function", "variable", "python", "javascript", "class", "def ",
                 "import ", "return", "algorithm", "programming", "debug", "compile"],
        "math": ["calculate", "equation", "formula", "number", "sum", "multiply",
                 "algebra", "geometry", "calculus", "theorem", "proof"],
        "science": ["experiment", "hypothesis", "molecule", "atom", "cell", "dna",
                    "evolution", "physics", "chemistry", "biology", "quantum"],
        "history": ["century", "war", "empire", "king", "queen", "revolution",
                    "ancient", "medieval", "civilization", "dynasty"],
        "creative": ["story", "poem", "write", "creative", "imagine", "fiction",
                     "character", "plot", "narrative", "metaphor"],
        "chat": ["hello", "how are you", "thanks", "please", "help me",
                "can you", "what do you think", "tell me about"],
    }

    scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[domain] = score

    if scores:
        return max(scores, key=scores.get)  # type: ignore
    return "general"


def analyze_domain_distribution(data: List[Dict]) -> Dict[str, int]:
    """Analyze the domain distribution of a dataset."""
    domains: Counter = Counter()
    for item in data:
        text = item.get("prompt", "") + " " + item.get("response", "")
        domain = detect_domain(text)
        domains[domain] += 1
    return dict(domains)


def mix_by_domain(data: List[Dict], ratios: Optional[Dict[str, float]] = None,
                  target_size: Optional[int] = None) -> List[Dict]:
    """Mix dataset entries according to domain ratios.

    Args:
        data: Full dataset
        ratios: Desired domain ratios (e.g., {"code": 0.4, "chat": 0.3, "general": 0.3})
                If None, uses equal distribution
        target_size: Target dataset size (default: same as input)

    Returns:
        Mixed dataset
    """
    # Group by domain
    by_domain: Dict[str, List[Dict]] = {}
    for item in data:
        text = item.get("prompt", "") + " " + item.get("response", "")
        domain = detect_domain(text)
        by_domain.setdefault(domain, []).append(item)

    if ratios is None:
        # Equal distribution
        domains = list(by_domain.keys())
        ratios = {d: 1.0 / len(domains) for d in domains}

    if target_size is None:
        target_size = len(data)

    # Normalize ratios
    total_ratio = sum(ratios.values())
    ratios = {d: r / total_ratio for d, r in ratios.items()}

    # Sample according to ratios
    mixed = []
    for domain, ratio in ratios.items():
        n_target = int(target_size * ratio)
        domain_items = by_domain.get(domain, [])
        if domain_items:
            # Sample with replacement if needed
            if len(domain_items) >= n_target:
                selected = random.sample(domain_items, n_target)
            else:
                selected = domain_items * (n_target // len(domain_items) + 1)
                selected = selected[:n_target]
            mixed.extend(selected)

    random.shuffle(mixed)
    return mixed[:target_size]


# ---------------------------------------------------------------------------
# Interactive Curriculum Manager
# ---------------------------------------------------------------------------

def interactive_curriculum():
    """Interactive curriculum learning interface."""
    print("\n" + "=" * 55)
    print("       Curriculum Learning")
    print("=" * 55)

    while True:
        print("\n  Options:")
        print("  1  Analyze difficulty distribution")
        print("  2  Analyze domain distribution")
        print("  3  Preview training schedule")
        print("  4  Sort dataset by difficulty")
        print("  5  Mix dataset by domain ratios")
        print("  0  Back")

        try:
            choice = input("\n  curriculum>> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice in ('0', 'back', 'quit', 'q'):
            break

        try:
            if choice == '1':
                path = input("  Dataset path [data/train.json]: ").strip() or "data/train.json"
                with open(path, 'r') as f:
                    data = json.load(f)
                leveled = assign_difficulty_levels(data)
                level_counts = Counter(item["_level"] for item in leveled)
                print(f"\n  Difficulty Distribution ({len(data)} entries):")
                labels = ["Easy", "Medium-Easy", "Medium", "Medium-Hard", "Hard"]
                for level in sorted(level_counts):
                    label = labels[level] if level < len(labels) else f"Level {level}"
                    count = level_counts[level]
                    bar = "█" * (count * 30 // len(data))
                    print(f"  {label:<14} {bar} {count}")

            elif choice == '2':
                path = input("  Dataset path [data/train.json]: ").strip() or "data/train.json"
                with open(path, 'r') as f:
                    data = json.load(f)
                dist = analyze_domain_distribution(data)
                print(f"\n  Domain Distribution ({len(data)} entries):")
                for domain, count in sorted(dist.items(), key=lambda x: -x[1]):
                    pct = count * 100 / len(data)
                    bar = "█" * int(pct / 2)
                    print(f"  {domain:<12} {bar} {count} ({pct:.1f}%)")

            elif choice == '3':
                path = input("  Dataset path [data/train.json]: ").strip() or "data/train.json"
                epochs = input("  Epochs [10]: ").strip()
                epochs = int(epochs) if epochs else 10
                with open(path, 'r') as f:
                    data = json.load(f)
                scheduler = CurriculumScheduler(data, num_epochs=epochs)
                schedule = scheduler.get_schedule_info()
                print(f"\n  Training Schedule:")
                for s in schedule:
                    bar = "█" * int(s["fraction"] * 30)
                    print(f"  Epoch {s['epoch']:>2}: {bar} {s['items']:>5} items ({s['fraction']*100:.0f}%)")

            elif choice == '4':
                path = input("  Dataset path [data/train.json]: ").strip() or "data/train.json"
                with open(path, 'r') as f:
                    data = json.load(f)
                sorted_data = sort_by_difficulty(data)
                with open(path, 'w') as f:
                    json.dump(sorted_data, f, indent=2)
                print(f"  ✓ Dataset sorted by difficulty (easy → hard)")

            elif choice == '5':
                path = input("  Dataset path [data/train.json]: ").strip() or "data/train.json"
                with open(path, 'r') as f:
                    data = json.load(f)
                dist = analyze_domain_distribution(data)
                print(f"  Current: {dist}")
                print("  Enter ratios (domain:ratio, e.g., 'code:0.4,chat:0.3,general:0.3'):")
                raw = input("  Ratios: ").strip()
                if raw:
                    ratios = {}
                    for part in raw.split(","):
                        d, r = part.strip().split(":")
                        ratios[d.strip()] = float(r.strip())
                    mixed = mix_by_domain(data, ratios)
                    with open(path, 'w') as f:
                        json.dump(mixed, f, indent=2)
                    new_dist = analyze_domain_distribution(mixed)
                    print(f"  ✓ New distribution: {new_dist}")

        except (KeyboardInterrupt, EOFError):
            continue
        except Exception as e:
            print(f"  Error: {e}")

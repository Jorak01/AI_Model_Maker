"""Data loading utilities — optimized with pre-tokenization and parallel workers."""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


class ConversationDataset(Dataset):
    """Pre-tokenizes all data on init for faster __getitem__ during training."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        print(f"Loaded {len(raw_data)} pairs from {data_path}")

        # Pre-tokenize everything upfront (avoids per-batch tokenization overhead)
        self.input_ids: List[torch.Tensor] = []
        self.target_ids: List[torch.Tensor] = []
        for item in raw_data:
            inp, tgt = tokenizer.encode_conversation(
                item['prompt'], item['response'], max_length
            )
            self.input_ids.append(torch.tensor(inp, dtype=torch.long))
            self.target_ids.append(torch.tensor(tgt, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_data_loaders(train_path: str, test_path: str, tokenizer,
                        batch_size: int = 32, max_length: int = 128,
                        num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders with parallel workers and memory pinning.

    Args:
        num_workers: Number of parallel data loading processes.
                     0 = main process only (safe default).
                     2-4 recommended on multi-core CPUs with large datasets.
    """
    train_ds = ConversationDataset(train_path, tokenizer, max_length)
    test_ds = ConversationDataset(test_path, tokenizer, max_length)

    # Detect CUDA for pin_memory (speeds up host→device transfer)
    pin = torch.cuda.is_available()

    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs['persistent_workers'] = True   # Keep workers alive between epochs
        worker_kwargs['prefetch_factor'] = 2          # Prefetch 2 batches per worker

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, **worker_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, **worker_kwargs
    )
    return train_loader, test_loader


def load_and_prepare_data(data_path: str) -> List[str]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = []
    for item in data:
        texts.append(item['prompt'])
        texts.append(item['response'])
    return texts

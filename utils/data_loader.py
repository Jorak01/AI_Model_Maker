"""Data loading utilities for training."""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple


class ConversationDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} pairs from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        input_ids, target_ids = self.tokenizer.encode_conversation(
            item['prompt'], item['response'], self.max_length
        )
        return torch.tensor(input_ids), torch.tensor(target_ids)


def create_data_loaders(train_path: str, test_path: str, tokenizer,
                        batch_size: int = 32, max_length: int = 128) -> Tuple[DataLoader, DataLoader]:
    train_ds = ConversationDataset(train_path, tokenizer, max_length)
    test_ds = ConversationDataset(test_path, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


def load_and_prepare_data(data_path: str) -> List[str]:
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = []
    for item in data:
        texts.append(item['prompt'])
        texts.append(item['response'])
    return texts

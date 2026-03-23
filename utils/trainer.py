"""Training utilities for the conversational AI model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import time


class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 test_loader: DataLoader, learning_rate: float = 0.0001,
                 device: str = 'cpu', checkpoint_dir: str = 'checkpoints',
                 gradient_clip: float = 1.0, warmup_steps: int = 500,
                 weight_decay: float = 0.01, pad_token_id: int = 0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.gradient_clip = gradient_clip
        self.warmup_steps = warmup_steps
        self.base_lr = learning_rate
        self.global_step = 0
        self.best_loss = float('inf')

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, len(train_loader) * 10)
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            logits = self.model(input_ids)
            logits_flat = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            targets_flat = target_ids[:, 1:].contiguous().view(-1)
            loss = self.criterion(logits_flat, targets_flat)

            self.optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.gradient_clip
                )

            if self.global_step < self.warmup_steps:
                scale = self.global_step / max(1, self.warmup_steps)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.base_lr * scale

            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            self.global_step += 1

            pbar.set_postfix(loss=f'{loss.item():.4f}',
                             avg=f'{total_loss / (batch_idx + 1):.4f}')

        return total_loss / max(1, len(self.train_loader))

    def evaluate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0
        pbar = tqdm(self.test_loader, desc=f"Epoch {epoch} [Eval]")

        with torch.no_grad():
            for input_ids, target_ids in pbar:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                logits = self.model(input_ids)
                logits_flat = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                targets_flat = target_ids[:, 1:].contiguous().view(-1)
                loss = self.criterion(logits_flat, targets_flat)
                total_loss += loss.item()
                pbar.set_postfix(loss=f'{loss.item():.4f}')

        return total_loss / max(1, len(self.test_loader))

    def _save_model(self, path: str):
        """Save the model to the given path, using its save() method if available."""
        if hasattr(self.model, 'save') and callable(getattr(self.model, 'save')):
            self.model.save(path)  # type: ignore[operator]
        else:
            torch.save(self.model.state_dict(), path)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pt')
        self._save_model(path)
        print(f"Saved: {path}")
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            self._save_model(best_path)
            print(f"Best model saved: {best_path}")

    def train(self, num_epochs: int, save_every: int = 1, keep_last: int = 5):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"\nTraining {num_epochs} epochs on {self.device}")
        print(f"Parameters: {trainable:,} trainable / {total:,} total")
        print(f"Batches: {len(self.train_loader)} train, {len(self.test_loader)} test\n")
        start = time.time()

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            test_loss = self.evaluate(epoch)

            is_best = test_loss < self.best_loss
            if is_best:
                self.best_loss = test_loss

            print(f"\nEpoch {epoch}/{num_epochs}: train={train_loss:.4f} test={test_loss:.4f} "
                  f"time={time.time() - t0:.1f}s lr={self.optimizer.param_groups[0]['lr']:.6f}\n")

            if epoch % save_every == 0:
                self.save_checkpoint(epoch, is_best)

            if keep_last > 0:
                ckpts = sorted([f for f in os.listdir(self.checkpoint_dir)
                                if f.startswith('model_epoch_') and f.endswith('.pt')])
                while len(ckpts) > keep_last:
                    os.remove(os.path.join(self.checkpoint_dir, ckpts.pop(0)))

        print(f"\nDone in {time.time() - start:.1f}s | Best loss: {self.best_loss:.4f}")

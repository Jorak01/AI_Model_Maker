"""Tests for the Trainer."""

import os
import sys
import json
import tempfile
import shutil
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import ConversationalModel
from models.tokenizer import Tokenizer
from utils.data_loader import ConversationDataset, create_data_loaders
from utils.trainer import Trainer


@pytest.fixture
def sample_data_file():
    data = [
        {"prompt": "Hello", "response": "Hi there"},
        {"prompt": "How are you", "response": "I am fine"},
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(data, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def setup_training(sample_data_file):
    tok = Tokenizer(vocab_size=500, method="word")
    tok.build_vocab(["hello hi there how are you i am fine"])

    train_loader, test_loader = create_data_loaders(
        sample_data_file, sample_data_file, tok, batch_size=2, max_length=16
    )

    model = ConversationalModel(
        vocab_size=len(tok), embedding_dim=16, hidden_dim=32,
        num_layers=1, num_heads=2, max_seq_length=16, pad_token_id=0
    )

    ckpt_dir = tempfile.mkdtemp()
    return model, train_loader, test_loader, tok, ckpt_dir


class TestTrainer:
    def test_trainer_creation(self, setup_training):
        model, train_loader, test_loader, tok, ckpt_dir = setup_training
        try:
            trainer = Trainer(model=model, train_loader=train_loader,
                              test_loader=test_loader, device='cpu',
                              checkpoint_dir=ckpt_dir, pad_token_id=tok.pad_token_id)
            assert trainer.device == 'cpu'
            assert trainer.global_step == 0
            assert trainer.best_loss == float('inf')
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def test_train_one_epoch(self, setup_training):
        model, train_loader, test_loader, tok, ckpt_dir = setup_training
        try:
            trainer = Trainer(model=model, train_loader=train_loader,
                              test_loader=test_loader, device='cpu',
                              checkpoint_dir=ckpt_dir, pad_token_id=tok.pad_token_id)
            loss = trainer.train_epoch(1)
            assert isinstance(loss, float)
            assert loss > 0
            assert trainer.global_step > 0
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def test_evaluate(self, setup_training):
        model, train_loader, test_loader, tok, ckpt_dir = setup_training
        try:
            trainer = Trainer(model=model, train_loader=train_loader,
                              test_loader=test_loader, device='cpu',
                              checkpoint_dir=ckpt_dir, pad_token_id=tok.pad_token_id)
            loss = trainer.evaluate(1)
            assert isinstance(loss, float)
            assert loss > 0
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def test_save_checkpoint(self, setup_training):
        model, train_loader, test_loader, tok, ckpt_dir = setup_training
        try:
            trainer = Trainer(model=model, train_loader=train_loader,
                              test_loader=test_loader, device='cpu',
                              checkpoint_dir=ckpt_dir, pad_token_id=tok.pad_token_id)
            trainer.save_checkpoint(1, is_best=True)
            assert os.path.exists(os.path.join(ckpt_dir, 'model_epoch_1.pt'))
            assert os.path.exists(os.path.join(ckpt_dir, 'best_model.pt'))
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def test_full_train_loop(self, setup_training):
        model, train_loader, test_loader, tok, ckpt_dir = setup_training
        try:
            trainer = Trainer(model=model, train_loader=train_loader,
                              test_loader=test_loader, device='cpu',
                              checkpoint_dir=ckpt_dir, pad_token_id=tok.pad_token_id,
                              warmup_steps=2)
            trainer.train(num_epochs=2, save_every=1, keep_last=2)
            assert trainer.best_loss < float('inf')
            # Check checkpoints exist
            files = os.listdir(ckpt_dir)
            assert any('model_epoch_' in f for f in files)
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def test_only_trainable_params_optimized(self, setup_training):
        model, train_loader, test_loader, tok, ckpt_dir = setup_training
        # Freeze some params
        for i, p in enumerate(model.parameters()):
            if i == 0:
                p.requires_grad = False
        try:
            trainer = Trainer(model=model, train_loader=train_loader,
                              test_loader=test_loader, device='cpu',
                              checkpoint_dir=ckpt_dir, pad_token_id=tok.pad_token_id)
            # Optimizer should only have trainable params
            opt_params = sum(len(pg['params']) for pg in trainer.optimizer.param_groups)
            trainable = sum(1 for p in model.parameters() if p.requires_grad)
            assert opt_params == trainable
        finally:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

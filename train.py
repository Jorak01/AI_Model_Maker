"""Train the conversational AI model — with performance optimizations."""

import os
import yaml
import torch
from typing import Optional
from models.tokenizer import Tokenizer
from models.model_factory import create_model, list_models, list_pipelines
from utils.data_loader import create_data_loaders, load_and_prepare_data
from utils.trainer import Trainer


def load_config(path: str = 'config.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device(config: dict) -> str:
    if config['device']['use_cuda'] and torch.cuda.is_available():
        dev = f"cuda:{config['device']['cuda_device']}"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return dev
    print("Using CPU")
    return 'cpu'


def main(config_path: str = 'config.yaml', checkpoint: Optional[str] = None, rebuild_vocab: bool = False):
    config = load_config(config_path)
    device = get_device(config)
    ckpt_dir = config['checkpoint']['save_dir']
    tok_path = os.path.join(ckpt_dir, 'tokenizer.pkl')
    base_model = config['model']['base_model']
    pipeline = config['training']['pipeline']
    perf = config.get('performance', {})

    print(f"\nModel: {base_model} | Pipeline: {pipeline}")

    # Tokenizer
    if os.path.exists(tok_path) and not rebuild_vocab:
        print(f"Loading tokenizer from {tok_path}")
        tokenizer = Tokenizer.load(tok_path)
    else:
        print("Building vocabulary...")
        tokenizer = Tokenizer(config['model']['vocab_size'], method='word')
        texts = load_and_prepare_data(config['data']['train_path'])
        if os.path.exists(config['data']['test_path']):
            texts += load_and_prepare_data(config['data']['test_path'])
        tokenizer.build_vocab(texts)
        os.makedirs(ckpt_dir, exist_ok=True)
        tokenizer.save(tok_path)

    # Data (with configurable num_workers for parallel loading)
    num_workers = perf.get('num_workers', 0)
    print(f"Creating data loaders (workers={num_workers})...")
    train_loader, test_loader = create_data_loaders(
        config['data']['train_path'], config['data']['test_path'],
        tokenizer, config['training']['batch_size'],
        config['model']['max_seq_length'], num_workers=num_workers,
    )

    # Model (via factory)
    print("Initializing model...")
    if base_model == "custom":
        config['model']['vocab_size'] = len(tokenizer)
    model = create_model(config)

    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt['state_dict'])

    # Train (with AMP and torch.compile from performance config)
    trainer = Trainer(
        model=model, train_loader=train_loader, test_loader=test_loader,
        learning_rate=config['training']['learning_rate'], device=device,
        checkpoint_dir=ckpt_dir, gradient_clip=config['training']['gradient_clip'],
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        pad_token_id=tokenizer.pad_token_id,
        use_amp=perf.get('mixed_precision', True),
        compile_model=perf.get('compile_model', True),
    )
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_every=config['checkpoint']['save_every'],
        keep_last=config['checkpoint']['keep_last']
    )
    print("\nTraining complete!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train conversational AI')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--rebuild-vocab', action='store_true')
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.rebuild_vocab)

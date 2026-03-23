"""Interactive chat with the trained AI model — optimized with inference_mode and torch.compile."""

import os
import yaml
import torch
from typing import Optional
from models.tokenizer import Tokenizer
from models.model_factory import load_model


def load_config(path: str = 'config.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device(config: dict) -> str:
    if config['device']['use_cuda'] and torch.cuda.is_available():
        return f"cuda:{config['device']['cuda_device']}"
    return 'cpu'


@torch.inference_mode()
def generate_response(model, tokenizer: Tokenizer, prompt: str, config: dict, device: str) -> str:
    """Generate a response — uses inference_mode for fastest inference."""
    model.eval()
    prompt_ids = tokenizer.encode(prompt, add_special=False)
    input_ids = [tokenizer.bos_token_id] + prompt_ids + [tokenizer.sep_token_id]
    input_tensor = torch.tensor([input_ids], device=device)

    gen_cfg = config['generation']
    generated = model.generate(
        input_tensor, max_length=gen_cfg['max_length'],
        temperature=gen_cfg['temperature'], top_k=gen_cfg['top_k'],
        top_p=gen_cfg['top_p'], repetition_penalty=gen_cfg['repetition_penalty'],
        eos_token_id=tokenizer.eos_token_id
    )

    gen_ids = generated[0].tolist()
    try:
        sep_idx = gen_ids.index(tokenizer.sep_token_id)
        response_ids = gen_ids[sep_idx + 1:]
    except ValueError:
        response_ids = gen_ids[len(input_ids):]

    return tokenizer.decode(response_ids, skip_special=True)


def load_model_and_tokenizer(config: dict, device: str, checkpoint: Optional[str] = None):
    """Load model and tokenizer from checkpoints, optionally with torch.compile."""
    ckpt_dir = config['checkpoint']['save_dir']
    tok_path = os.path.join(ckpt_dir, 'tokenizer.pkl')

    if not os.path.exists(tok_path):
        print(f"Error: Tokenizer not found at {tok_path}")
        print("Train the model first: python run.py train")
        return None, None

    tokenizer = Tokenizer.load(tok_path)

    if not checkpoint:
        best = os.path.join(ckpt_dir, 'best_model.pt')
        if os.path.exists(best):
            checkpoint = best
        else:
            ckpts = sorted([f for f in os.listdir(ckpt_dir)
                            if f.startswith('model_epoch_') and f.endswith('.pt')])
            if not ckpts:
                print(f"Error: No checkpoints in {ckpt_dir}")
                print("Train the model first: python run.py train")
                return None, None
            checkpoint = os.path.join(ckpt_dir, ckpts[-1])

    if checkpoint is None:
        print(f"Error: No checkpoint available in {ckpt_dir}")
        print("Train the model first: python run.py train")
        return None, None

    print(f"Loading model from {checkpoint}")
    model = load_model(checkpoint, device=device)

    # torch.compile for faster inference (PyTorch 2.0+)
    perf = config.get('performance', {})
    if perf.get('compile_model', True) and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            print("  ⚡ torch.compile() enabled for inference")
        except Exception:
            pass

    return model, tokenizer


def interactive_chat(model, tokenizer, config, device):
    print("\n" + "=" * 50)
    print("  AI Chat  |  'quit' to exit  |  'clear' to reset")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nReturning to menu...")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'stop'):
            print("Returning to menu...")
            break
        if user_input.lower() == 'clear':
            print("[Cleared]\n")
            continue

        try:
            response = generate_response(model, tokenizer, user_input, config, device)
            print(f"AI: {response}\n")
        except Exception as e:
            print(f"[Error: {e}]\n")


def main(config_path: str = 'config.yaml', checkpoint: Optional[str] = None):
    config = load_config(config_path)
    device = get_device(config)
    model, tokenizer = load_model_and_tokenizer(config, device, checkpoint)
    if model is None:
        return
    interactive_chat(model, tokenizer, config, device)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Chat with AI model')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    main(args.config, args.checkpoint)

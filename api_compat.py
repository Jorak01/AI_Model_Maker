"""
OpenAI-Compatible API Endpoints.

Exposes /v1/chat/completions and /v1/images/generations so external tools
(Cursor, Continue, Open WebUI, etc.) can use local models as a drop-in
OpenAI replacement.

Run: python api_compat.py
     python api_compat.py --port 8001
"""

import json
import time
import uuid
import os
import sys
import argparse

try:
    from flask import Flask, request, jsonify, Response
    from flask_cors import CORS
except ImportError:
    Flask = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_id():
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def _timestamp():
    return int(time.time())


def _load_local_model():
    """Try to load the local trained model and tokenizer."""
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        ckpt_dir = config['checkpoint']['save_dir']
        best_path = os.path.join(ckpt_dir, 'best_model.pt')
        tok_path = os.path.join(ckpt_dir, 'tokenizer.pkl')

        if not os.path.exists(best_path) or not os.path.exists(tok_path):
            return None, None, config

        import torch
        from models.model_factory import load_model
        from models.tokenizer import Tokenizer

        device = 'cpu'
        if config['device']['use_cuda'] and torch.cuda.is_available():
            device = f"cuda:{config['device']['cuda_device']}"

        model = load_model(best_path, device=device)
        tokenizer = Tokenizer.load(tok_path)
        return model, tokenizer, config
    except Exception:
        return None, None, {}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app():
    """Create the OpenAI-compatible Flask app."""
    if Flask is None:
        raise ImportError("Flask is required: pip install flask flask-cors")

    app = Flask(__name__)
    CORS(app)

    # Lazy-load model on first request
    _state = {'model': None, 'tokenizer': None, 'config': {}, 'loaded': False}

    def _ensure_model():
        if not _state['loaded']:
            m, t, c = _load_local_model()
            _state['model'] = m
            _state['tokenizer'] = t
            _state['config'] = c
            _state['loaded'] = True
        return _state['model'], _state['tokenizer'], _state['config']

    # ------------------------------------------------------------------
    # /v1/models
    # ------------------------------------------------------------------
    @app.route('/v1/models', methods=['GET'])
    def list_models():
        models = [
            {
                "id": "local-model",
                "object": "model",
                "created": _timestamp(),
                "owned_by": "local",
            }
        ]
        # Add registered models
        try:
            from model_registry import _load_registry
            registry = _load_registry()
            for name, info in registry.get('models', {}).items():
                models.append({
                    "id": name,
                    "object": "model",
                    "created": _timestamp(),
                    "owned_by": "local",
                })
        except Exception:
            pass

        return jsonify({"object": "list", "data": models})

    # ------------------------------------------------------------------
    # /v1/chat/completions
    # ------------------------------------------------------------------
    @app.route('/v1/chat/completions', methods=['POST'])
    def chat_completions():
        data = request.get_json(force=True)
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.8)
        max_tokens = data.get('max_tokens', 256)
        model_name = data.get('model', 'local-model')

        # Build prompt from messages
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)

        # Try local model first
        model, tokenizer, config = _ensure_model()
        response_text = ""

        if model is not None and tokenizer is not None:
            try:
                import torch
                device = next(model.parameters()).device
                gen_config = config.get('generation', {})

                encoded = tokenizer.encode(prompt)
                input_ids = torch.tensor([encoded], device=device)

                with torch.no_grad():
                    # Simple greedy/sampling generation
                    generated = list(encoded)
                    for _ in range(min(max_tokens, gen_config.get('max_length', 256))):
                        inp = torch.tensor([generated], device=device)
                        output = model(inp)
                        logits = output[:, -1, :] / max(temperature, 0.01)

                        # Top-k sampling
                        top_k = gen_config.get('top_k', 50)
                        if top_k > 0:
                            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                            logits[indices_to_remove] = float('-inf')

                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                        generated.append(next_token)

                        if hasattr(tokenizer, 'special_tokens'):
                            if next_token == tokenizer.special_tokens.get('<eos>', -1):
                                break

                    response_text = tokenizer.decode(generated[len(encoded):])
            except Exception as e:
                response_text = f"[Generation error: {e}]"
        else:
            # Fallback: try external API
            try:
                from external_api import chat_with_provider, find_available_provider
                provider = find_available_provider()
                if provider:
                    user_msg = messages[-1].get('content', '') if messages else ''
                    response_text = chat_with_provider(provider, user_msg)
                else:
                    response_text = "[No model available. Train a model first or configure an external API provider.]"
            except Exception:
                response_text = "[No model available. Train a model or configure an external API.]"

        # Format as OpenAI response
        result = {
            "id": _generate_id(),
            "object": "chat.completion",
            "created": _timestamp(),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text.strip()
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(prompt.split()) + len(response_text.split())
            }
        }
        return jsonify(result)

    # ------------------------------------------------------------------
    # /v1/images/generations
    # ------------------------------------------------------------------
    @app.route('/v1/images/generations', methods=['POST'])
    def image_generations():
        data = request.get_json(force=True)
        prompt = data.get('prompt', '')
        n = data.get('n', 1)
        size = data.get('size', '512x512')

        images = []
        try:
            from image_gen import generate_image
            for i in range(n):
                w, h = [int(x) for x in size.split('x')]
                result = generate_image(prompt, width=w, height=h)
                if result and os.path.exists(result):
                    import base64
                    with open(result, 'rb') as f:
                        b64 = base64.b64encode(f.read()).decode()
                    images.append({"b64_json": b64})
                else:
                    images.append({"b64_json": "", "error": "Generation failed"})
        except Exception as e:
            for _ in range(n):
                images.append({"b64_json": "", "error": str(e)})

        return jsonify({
            "created": _timestamp(),
            "data": images
        })

    # ------------------------------------------------------------------
    # /v1/embeddings (stub)
    # ------------------------------------------------------------------
    @app.route('/v1/embeddings', methods=['POST'])
    def embeddings():
        data = request.get_json(force=True)
        input_text = data.get('input', '')
        if isinstance(input_text, str):
            input_text = [input_text]

        # Simple TF-IDF-like embedding stub
        embeddings_list = []
        for i, text in enumerate(input_text):
            words = text.lower().split()
            # Create a simple hash-based embedding
            import hashlib
            emb = []
            for dim in range(128):
                h = hashlib.md5(f"{text}_{dim}".encode()).hexdigest()
                emb.append((int(h[:8], 16) / 0xFFFFFFFF) * 2 - 1)
            embeddings_list.append({
                "object": "embedding",
                "index": i,
                "embedding": emb
            })

        return jsonify({
            "object": "list",
            "data": embeddings_list,
            "model": data.get('model', 'local-model'),
            "usage": {"prompt_tokens": sum(len(t.split()) for t in input_text), "total_tokens": sum(len(t.split()) for t in input_text)}
        })

    # ------------------------------------------------------------------
    # Health & info
    # ------------------------------------------------------------------
    @app.route('/health', methods=['GET'])
    @app.route('/v1/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok", "service": "ai-model-compat-api"})

    return app


def main():
    parser = argparse.ArgumentParser(description='OpenAI-Compatible API Server')
    parser.add_argument('--port', type=int, default=8001, help='Port (default: 8001)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host (default: 0.0.0.0)')
    args = parser.parse_args()

    app = create_app()
    print(f"\n  OpenAI-Compatible API running on http://{args.host}:{args.port}")
    print(f"  Endpoints:")
    print(f"    POST /v1/chat/completions")
    print(f"    POST /v1/images/generations")
    print(f"    POST /v1/embeddings")
    print(f"    GET  /v1/models")
    print(f"\n  Use as OpenAI base_url: http://localhost:{args.port}/v1")
    print()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()

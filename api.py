"""REST API for the conversational AI model."""

from typing import Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from models.model_factory import load_model
from models.tokenizer import Tokenizer
from chat import load_config, get_device, load_model_and_tokenizer, generate_response
import torch.nn as nn

app = Flask(__name__)
CORS(app)

# Global state
_model: Optional[nn.Module] = None
_tokenizer: Optional[Tokenizer] = None
_config: Optional[dict] = None
_device: Optional[str] = None


def init_model(config_path: str = 'config.yaml', checkpoint: Optional[str] = None):
    """Initialize model for API serving."""
    global _model, _tokenizer, _config, _device
    _config = load_config(config_path)
    _device = get_device(_config)
    _model, _tokenizer = load_model_and_tokenizer(_config, _device, checkpoint)
    if _model is None:
        raise RuntimeError("Failed to load model. Train first: python run.py train")
    print("Model loaded and ready for API requests.")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model_loaded': _model is not None})


@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint. Body: {"message": "your message"}"""
    if _model is None or _tokenizer is None or _config is None or _device is None:
        return jsonify({'error': 'Model not loaded', 'status': 'error'}), 503

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field', 'status': 'error'}), 400

    message = data['message'].strip()
    if not message:
        return jsonify({'error': 'Empty message', 'status': 'error'}), 400

    try:
        response = generate_response(_model, _tokenizer, message, _config, _device)
        return jsonify({'response': response, 'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/config', methods=['GET'])
def get_model_config():
    """Get current model/generation configuration."""
    if _config is None:
        return jsonify({'error': 'Not initialized', 'status': 'error'}), 503
    return jsonify({
        'model': _config['model'],
        'generation': _config['generation'],
        'status': 'ok'
    })


@app.route('/generate', methods=['POST'])
def generate():
    """Advanced generation with custom params.
    Body: {"message": "...", "temperature": 0.8, "top_k": 50, "top_p": 0.9, "max_length": 100}
    """
    if _model is None or _tokenizer is None or _config is None or _device is None:
        return jsonify({'error': 'Model not loaded', 'status': 'error'}), 503

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing "message" field', 'status': 'error'}), 400

    gen_config = dict(_config)
    gen_config['generation'] = dict(_config['generation'])
    for key in ('temperature', 'top_k', 'top_p', 'max_length', 'repetition_penalty'):
        if key in data:
            gen_config['generation'][key] = data[key]

    try:
        response = generate_response(_model, _tokenizer, data['message'].strip(), gen_config, _device)
        return jsonify({'response': response, 'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


def main(config_path: str = 'config.yaml', checkpoint: Optional[str] = None):
    config = load_config(config_path)
    init_model(config_path, checkpoint)
    host = config.get('api', {}).get('host', '0.0.0.0')
    port = config.get('api', {}).get('port', 8000)
    debug = config.get('api', {}).get('debug', False)
    print(f"\nAPI running at http://{host}:{port}")
    print(f"  POST /chat     - Send a chat message")
    print(f"  POST /generate - Generate with custom params")
    print(f"  GET  /health   - Health check")
    print(f"  GET  /config   - View configuration\n")
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run AI model API server')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    main(args.config, args.checkpoint)

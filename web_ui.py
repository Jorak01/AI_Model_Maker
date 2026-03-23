"""
Web UI — Simple browser-based interface for the AI Model Suite.

Uses a built-in Flask server with HTML/CSS/JS (no Gradio dependency required).
Provides chat, image generation, model status, and training controls.

Run: python web_ui.py
     python web_ui.py --port 7860
"""

import os
import sys
import json
import argparse

try:
    from flask import Flask, request, jsonify, send_from_directory, Response
    from flask_cors import CORS
except ImportError:
    Flask = None

# ---------------------------------------------------------------------------
# HTML Template (single-page app)
# ---------------------------------------------------------------------------

WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Model Suite</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f0f; color: #e0e0e0; }
.header { background: #1a1a2e; padding: 16px 24px; border-bottom: 1px solid #333; display: flex; align-items: center; gap: 16px; }
.header h1 { font-size: 20px; color: #7c8cf8; }
.header .nav { display: flex; gap: 8px; margin-left: auto; }
.header .nav button { background: #2a2a3e; border: 1px solid #444; color: #ccc; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.header .nav button.active { background: #7c8cf8; color: #fff; border-color: #7c8cf8; }
.header .nav button:hover { background: #3a3a4e; }
.main { max-width: 900px; margin: 0 auto; padding: 24px; }
.panel { display: none; }
.panel.active { display: block; }

/* Chat */
.chat-box { background: #1a1a2e; border: 1px solid #333; border-radius: 8px; height: 450px; overflow-y: auto; padding: 16px; margin-bottom: 12px; }
.chat-msg { margin-bottom: 12px; padding: 10px 14px; border-radius: 8px; max-width: 80%; }
.chat-msg.user { background: #2a3a5e; margin-left: auto; text-align: right; }
.chat-msg.assistant { background: #2a2a3e; }
.chat-msg .role { font-size: 11px; color: #888; margin-bottom: 4px; }
.chat-input { display: flex; gap: 8px; }
.chat-input input { flex: 1; background: #1a1a2e; border: 1px solid #444; color: #e0e0e0; padding: 12px; border-radius: 6px; font-size: 14px; }
.chat-input button { background: #7c8cf8; color: #fff; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-size: 14px; }

/* Status */
.status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.status-card { background: #1a1a2e; border: 1px solid #333; border-radius: 8px; padding: 20px; }
.status-card h3 { color: #7c8cf8; margin-bottom: 12px; font-size: 14px; text-transform: uppercase; }
.status-card .value { font-size: 24px; font-weight: bold; }
.status-card .detail { color: #888; font-size: 13px; margin-top: 4px; }

/* Settings */
.settings-form label { display: block; color: #aaa; font-size: 13px; margin-bottom: 4px; margin-top: 12px; }
.settings-form input, .settings-form select { background: #1a1a2e; border: 1px solid #444; color: #e0e0e0; padding: 8px 12px; border-radius: 4px; width: 100%; max-width: 300px; }
.settings-form .row { display: flex; gap: 16px; flex-wrap: wrap; }

/* Models */
.model-list { list-style: none; }
.model-list li { background: #1a1a2e; border: 1px solid #333; border-radius: 6px; padding: 12px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
.model-list li .name { font-weight: bold; }
.model-list li .meta { color: #888; font-size: 12px; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
.badge.ok { background: #1a3a2e; color: #4caf50; }
.badge.warn { background: #3a3a1e; color: #ffb74d; }
</style>
</head>
<body>
<div class="header">
    <h1>🤖 AI Model Suite</h1>
    <div class="nav">
        <button class="active" onclick="showPanel('chat')">Chat</button>
        <button onclick="showPanel('status')">Status</button>
        <button onclick="showPanel('models')">Models</button>
        <button onclick="showPanel('settings')">Settings</button>
    </div>
</div>
<div class="main">
    <!-- Chat Panel -->
    <div id="panel-chat" class="panel active">
        <div class="chat-box" id="chatBox"></div>
        <div class="chat-input">
            <input type="text" id="chatInput" placeholder="Type a message..." onkeydown="if(event.key==='Enter')sendMsg()">
            <button onclick="sendMsg()">Send</button>
        </div>
    </div>

    <!-- Status Panel -->
    <div id="panel-status" class="panel">
        <h2 style="margin-bottom:16px;">System Status</h2>
        <div class="status-grid" id="statusGrid">
            <div class="status-card"><h3>Status</h3><div class="value" id="srvStatus">Loading...</div></div>
            <div class="status-card"><h3>Models</h3><div class="value" id="srvModels">-</div></div>
            <div class="status-card"><h3>Device</h3><div class="value" id="srvDevice">-</div></div>
            <div class="status-card"><h3>Uptime</h3><div class="value" id="srvUptime">-</div></div>
        </div>
    </div>

    <!-- Models Panel -->
    <div id="panel-models" class="panel">
        <h2 style="margin-bottom:16px;">Registered Models</h2>
        <ul class="model-list" id="modelList"><li>Loading...</li></ul>
    </div>

    <!-- Settings Panel -->
    <div id="panel-settings" class="panel">
        <h2 style="margin-bottom:16px;">Generation Settings</h2>
        <div class="settings-form">
            <div class="row">
                <div>
                    <label>Temperature</label>
                    <input type="range" id="setTemp" min="0.1" max="2.0" step="0.1" value="0.8"
                           oninput="document.getElementById('setTempVal').textContent=this.value">
                    <span id="setTempVal">0.8</span>
                </div>
                <div>
                    <label>Max Tokens</label>
                    <input type="number" id="setMaxTok" value="256" min="16" max="2048">
                </div>
            </div>
            <div class="row">
                <div>
                    <label>Top-K</label>
                    <input type="number" id="setTopK" value="50" min="1" max="200">
                </div>
                <div>
                    <label>Top-P</label>
                    <input type="range" id="setTopP" min="0.1" max="1.0" step="0.05" value="0.9"
                           oninput="document.getElementById('setTopPVal').textContent=this.value">
                    <span id="setTopPVal">0.9</span>
                </div>
            </div>
        </div>
    </div>
</div>
<script>
const chatBox = document.getElementById('chatBox');
const chatInput = document.getElementById('chatInput');
let history = [];

function showPanel(name) {
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
    document.getElementById('panel-' + name).classList.add('active');
    event.target.classList.add('active');
    if (name === 'status') fetchStatus();
    if (name === 'models') fetchModels();
}

function addMsg(role, text) {
    const div = document.createElement('div');
    div.className = 'chat-msg ' + role;
    div.innerHTML = '<div class="role">' + role + '</div>' + text.replace(/\\n/g, '<br>');
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMsg() {
    const text = chatInput.value.trim();
    if (!text) return;
    chatInput.value = '';
    addMsg('user', text);
    history.push({role: 'user', content: text});

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                messages: history,
                temperature: parseFloat(document.getElementById('setTemp').value),
                max_tokens: parseInt(document.getElementById('setMaxTok').value)
            })
        });
        const data = await resp.json();
        const reply = data.response || data.error || 'No response';
        addMsg('assistant', reply);
        history.push({role: 'assistant', content: reply});
    } catch(e) {
        addMsg('assistant', 'Error: ' + e.message);
    }
}

async function fetchStatus() {
    try {
        const resp = await fetch('/api/status');
        const data = await resp.json();
        document.getElementById('srvStatus').textContent = data.status || 'Online';
        document.getElementById('srvModels').textContent = data.model_count || '0';
        document.getElementById('srvDevice').textContent = data.device || 'CPU';
        document.getElementById('srvUptime').textContent = data.uptime || '-';
    } catch(e) {
        document.getElementById('srvStatus').textContent = 'Error';
    }
}

async function fetchModels() {
    try {
        const resp = await fetch('/api/models');
        const data = await resp.json();
        const list = document.getElementById('modelList');
        list.innerHTML = '';
        if (data.models && data.models.length > 0) {
            data.models.forEach(m => {
                const li = document.createElement('li');
                li.innerHTML = '<div><span class="name">' + m.name + '</span><div class="meta">' +
                    (m.pipeline || '') + ' • ' + (m.base_model || '') + '</div></div>' +
                    '<span class="badge ok">ready</span>';
                list.appendChild(li);
            });
        } else {
            list.innerHTML = '<li>No models registered yet. Train a model first!</li>';
        }
    } catch(e) {
        document.getElementById('modelList').innerHTML = '<li>Error loading models</li>';
    }
}
</script>
</body>
</html>"""


def create_web_app():
    """Create the Web UI Flask app."""
    if Flask is None:
        raise ImportError("Flask is required: pip install flask flask-cors")

    app = Flask(__name__)
    CORS(app)

    _start_time = __import__('time').time()

    @app.route('/')
    def index():
        return Response(WEB_UI_HTML, mimetype='text/html')

    @app.route('/api/chat', methods=['POST'])
    def api_chat():
        data = request.get_json(force=True)
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.8)
        max_tokens = data.get('max_tokens', 256)

        user_msg = messages[-1]['content'] if messages else ''
        response_text = ""

        # Try local model
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)

            ckpt_dir = config['checkpoint']['save_dir']
            best_path = os.path.join(ckpt_dir, 'best_model.pt')
            tok_path = os.path.join(ckpt_dir, 'tokenizer.pkl')

            if os.path.exists(best_path) and os.path.exists(tok_path):
                import torch
                from models.model_factory import load_model
                from models.tokenizer import Tokenizer

                device = 'cpu'
                model = load_model(best_path, device=device)
                tokenizer = Tokenizer.load(tok_path)

                encoded = tokenizer.encode(user_msg)
                input_ids = torch.tensor([encoded], device=device)

                with torch.no_grad():
                    generated = list(encoded)
                    for _ in range(min(max_tokens, 256)):
                        inp = torch.tensor([generated], device=device)
                        output = model(inp)
                        logits = output[:, -1, :] / max(temperature, 0.01)
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, 1).item()
                        generated.append(next_token)
                        if hasattr(tokenizer, 'special_tokens'):
                            if next_token == tokenizer.special_tokens.get('<eos>', -1):
                                break

                response_text = tokenizer.decode(generated[len(encoded):])
        except Exception:
            pass

        # Fallback to external API
        if not response_text:
            try:
                from external_api import chat_with_provider, find_available_provider
                provider = find_available_provider()
                if provider:
                    response_text = chat_with_provider(provider, user_msg)
            except Exception:
                pass

        if not response_text:
            response_text = "No model available. Train a model with 'train' or configure an external API provider."

        return jsonify({"response": response_text})

    @app.route('/api/status', methods=['GET'])
    def api_status():
        import time as _time
        uptime_sec = int(_time.time() - _start_time)
        h, m = divmod(uptime_sec // 60, 60)

        model_count = 0
        try:
            from model_registry import _load_registry
            registry = _load_registry()
            model_count = len(registry.get('models', {}))
        except Exception:
            pass

        device = 'CPU'
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.get_device_name(0)
        except Exception:
            pass

        return jsonify({
            "status": "Online",
            "model_count": model_count,
            "device": device,
            "uptime": f"{h}h {m}m"
        })

    @app.route('/api/models', methods=['GET'])
    def api_models():
        models = []
        try:
            from model_registry import _load_registry
            registry = _load_registry()
            for name, info in registry.get('models', {}).items():
                models.append({
                    "name": name,
                    "pipeline": info.get("pipeline", ""),
                    "base_model": info.get("base_model", ""),
                    "intent": info.get("intent", "")
                })
        except Exception:
            pass
        return jsonify({"models": models})

    return app


def interactive_web_ui():
    """Launch the web UI interactively."""
    print("\n" + "=" * 55)
    print("       Web UI")
    print("=" * 55)
    print("\n  Starting web interface...")
    print("  Open http://localhost:7860 in your browser")
    print("  Press Ctrl+C to stop\n")

    app = create_web_app()
    try:
        app.run(host='0.0.0.0', port=7860, debug=False)
    except KeyboardInterrupt:
        print("\n  Web UI stopped.")


def main():
    parser = argparse.ArgumentParser(description='AI Model Suite Web UI')
    parser.add_argument('--port', type=int, default=7860, help='Port (default: 7860)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host')
    args = parser.parse_args()

    app = create_web_app()
    print(f"\n  Web UI running at http://localhost:{args.port}")
    print("  Press Ctrl+C to stop\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()

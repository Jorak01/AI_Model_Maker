"""Training Dashboard — Live metrics visualization via local web UI.

Features:
  - Real-time loss curve tracking
  - Learning rate schedule visualization
  - Training metrics storage and display
  - Simple HTML dashboard served via Flask
"""

import os
import json
import time
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any


# ---------------------------------------------------------------------------
# Metrics Storage
# ---------------------------------------------------------------------------

METRICS_PATH = "data/training_metrics.json"


class MetricsTracker:
    """Track and store training metrics over time."""

    def __init__(self, metrics_path: str = METRICS_PATH):
        self.metrics_path = metrics_path
        self.runs: List[Dict] = []
        self.current_run: Optional[Dict] = None
        self._load()

    def _load(self):
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, 'r') as f:
                    data = json.load(f)
                self.runs = data.get("runs", [])
            except (json.JSONDecodeError, IOError):
                self.runs = []

    def _save(self):
        os.makedirs(os.path.dirname(self.metrics_path) or '.', exist_ok=True)
        with open(self.metrics_path, 'w') as f:
            json.dump({"runs": self.runs}, f, indent=2)

    def start_run(self, name: str = "", config: Optional[Dict] = None):
        """Start a new training run."""
        self.current_run = {
            "name": name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "started": datetime.now().isoformat(),
            "config": config or {},
            "epochs": [],
            "status": "running",
        }

    def log_epoch(self, epoch: int, train_loss: float,
                  eval_loss: Optional[float] = None,
                  learning_rate: Optional[float] = None,
                  extra: Optional[Dict] = None):
        """Log metrics for one epoch."""
        if self.current_run is None:
            return

        entry = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "train_loss": round(train_loss, 6),
        }
        if eval_loss is not None:
            entry["eval_loss"] = round(eval_loss, 6)
        if learning_rate is not None:
            entry["learning_rate"] = learning_rate
        if extra:
            entry.update(extra)

        self.current_run["epochs"].append(entry)
        self._save_current()

    def _save_current(self):
        """Save current run progress."""
        if self.current_run is None:
            return
        # Update or append current run
        for i, run in enumerate(self.runs):
            if run.get("name") == self.current_run.get("name"):
                self.runs[i] = self.current_run
                self._save()
                return
        self.runs.append(self.current_run)
        self._save()

    def end_run(self, status: str = "completed"):
        """Mark current run as complete."""
        if self.current_run:
            self.current_run["status"] = status
            self.current_run["ended"] = datetime.now().isoformat()
            self._save_current()
            self.current_run = None

    def get_latest_run(self) -> Optional[Dict]:
        """Get the most recent training run."""
        if self.runs:
            return self.runs[-1]
        return None

    def get_all_runs(self) -> List[Dict]:
        """Get all training runs."""
        return self.runs

    def clear_history(self):
        """Clear all training history."""
        self.runs = []
        self.current_run = None
        self._save()


# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>AI Model Training Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
        h1 { color: #e94560; margin-bottom: 20px; }
        h2 { color: #0f3460; background: #e94560; padding: 8px 16px; border-radius: 4px; margin: 16px 0 8px; }
        .card { background: #16213e; border-radius: 8px; padding: 16px; margin: 8px 0; border: 1px solid #0f3460; }
        .metric { display: inline-block; margin: 8px 16px; text-align: center; }
        .metric .value { font-size: 2em; color: #e94560; font-weight: bold; }
        .metric .label { font-size: 0.9em; color: #888; }
        canvas { background: #0f3460; border-radius: 4px; margin: 8px 0; }
        table { width: 100%%; border-collapse: collapse; margin: 8px 0; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #0f3460; }
        th { color: #e94560; }
        .status-running { color: #4ecca3; }
        .status-completed { color: #7ec8e3; }
        .refresh-btn { background: #e94560; color: white; border: none; padding: 8px 16px;
                       border-radius: 4px; cursor: pointer; margin: 8px 0; }
        .refresh-btn:hover { background: #c73e54; }
    </style>
</head>
<body>
    <h1>🧠 AI Model Training Dashboard</h1>
    <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
    <button class="refresh-btn" onclick="autoRefresh()">⏱️ Auto-Refresh (5s)</button>

    <div id="summary" class="card"></div>
    <div id="chart-container" class="card">
        <canvas id="lossChart" width="800" height="300"></canvas>
    </div>
    <div id="runs" class="card"></div>
    <div id="epochs" class="card"></div>

    <script>
        let autoInterval = null;
        function autoRefresh() {
            if (autoInterval) { clearInterval(autoInterval); autoInterval = null; return; }
            autoInterval = setInterval(() => location.reload(), 5000);
        }

        fetch('/api/metrics').then(r => r.json()).then(data => {
            const runs = data.runs || [];
            const latest = runs[runs.length - 1];

            // Summary
            let summaryHtml = '<h2>Current Status</h2>';
            if (latest) {
                const epochs = latest.epochs || [];
                const lastEpoch = epochs[epochs.length - 1];
                summaryHtml += `
                    <div class="metric"><div class="value">${latest.name}</div><div class="label">Run Name</div></div>
                    <div class="metric"><div class="value">${epochs.length}</div><div class="label">Epochs</div></div>
                    <div class="metric"><div class="value">${lastEpoch ? lastEpoch.train_loss.toFixed(4) : 'N/A'}</div><div class="label">Latest Loss</div></div>
                    <div class="metric"><div class="value" class="status-${latest.status}">${latest.status}</div><div class="label">Status</div></div>
                `;
            } else {
                summaryHtml += '<p>No training runs yet.</p>';
            }
            document.getElementById('summary').innerHTML = summaryHtml;

            // Loss chart (simple canvas drawing)
            if (latest && latest.epochs.length > 0) {
                const canvas = document.getElementById('lossChart');
                const ctx = canvas.getContext('2d');
                const epochs = latest.epochs;
                const losses = epochs.map(e => e.train_loss);
                const evalLosses = epochs.map(e => e.eval_loss || null);

                const maxLoss = Math.max(...losses.filter(l => isFinite(l))) * 1.1;
                const minLoss = Math.min(...losses.filter(l => isFinite(l))) * 0.9;

                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // Grid
                ctx.strokeStyle = '#1a1a2e';
                ctx.lineWidth = 1;
                for (let i = 0; i < 5; i++) {
                    const y = 20 + (i / 4) * 260;
                    ctx.beginPath(); ctx.moveTo(60, y); ctx.lineTo(780, y); ctx.stroke();
                    const val = maxLoss - (i / 4) * (maxLoss - minLoss);
                    ctx.fillStyle = '#888'; ctx.font = '11px monospace';
                    ctx.fillText(val.toFixed(3), 5, y + 4);
                }

                // Train loss line
                ctx.strokeStyle = '#e94560';
                ctx.lineWidth = 2;
                ctx.beginPath();
                losses.forEach((loss, i) => {
                    const x = 60 + (i / Math.max(losses.length - 1, 1)) * 720;
                    const y = 20 + ((maxLoss - loss) / (maxLoss - minLoss)) * 260;
                    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                });
                ctx.stroke();

                // Eval loss line
                const validEval = evalLosses.filter(e => e !== null);
                if (validEval.length > 0) {
                    ctx.strokeStyle = '#4ecca3';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    let started = false;
                    evalLosses.forEach((loss, i) => {
                        if (loss === null) return;
                        const x = 60 + (i / Math.max(losses.length - 1, 1)) * 720;
                        const y = 20 + ((maxLoss - loss) / (maxLoss - minLoss)) * 260;
                        !started ? (ctx.moveTo(x, y), started = true) : ctx.lineTo(x, y);
                    });
                    ctx.stroke();
                }

                // Legend
                ctx.fillStyle = '#e94560'; ctx.fillRect(70, 10, 12, 3);
                ctx.fillStyle = '#eee'; ctx.fillText('Train Loss', 88, 14);
                ctx.fillStyle = '#4ecca3'; ctx.fillRect(180, 10, 12, 3);
                ctx.fillStyle = '#eee'; ctx.fillText('Eval Loss', 198, 14);
            }

            // Runs table
            let runsHtml = '<h2>Training Runs</h2><table><tr><th>Name</th><th>Status</th><th>Epochs</th><th>Started</th></tr>';
            runs.forEach(run => {
                runsHtml += `<tr>
                    <td>${run.name}</td>
                    <td class="status-${run.status}">${run.status}</td>
                    <td>${(run.epochs || []).length}</td>
                    <td>${run.started ? run.started.substring(0, 19) : ''}</td>
                </tr>`;
            });
            runsHtml += '</table>';
            document.getElementById('runs').innerHTML = runsHtml;

            // Epoch details for latest run
            if (latest && latest.epochs.length > 0) {
                let epochHtml = '<h2>Epoch Details (Latest Run)</h2><table><tr><th>Epoch</th><th>Train Loss</th><th>Eval Loss</th><th>LR</th></tr>';
                latest.epochs.slice(-20).forEach(e => {
                    epochHtml += `<tr>
                        <td>${e.epoch}</td>
                        <td>${e.train_loss.toFixed(6)}</td>
                        <td>${e.eval_loss ? e.eval_loss.toFixed(6) : '-'}</td>
                        <td>${e.learning_rate ? e.learning_rate.toExponential(2) : '-'}</td>
                    </tr>`;
                });
                epochHtml += '</table>';
                document.getElementById('epochs').innerHTML = epochHtml;
            }
        });
    </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Dashboard Server
# ---------------------------------------------------------------------------

def start_dashboard(port: int = 8501, open_browser: bool = True):
    """Start the training dashboard web server."""
    try:
        from flask import Flask, jsonify, send_from_directory
    except ImportError:
        print("  Flask required for dashboard. Install: pip install flask")
        return

    app = Flask(__name__)

    @app.route('/')
    def index():
        return DASHBOARD_HTML

    @app.route('/api/metrics')
    def metrics():
        tracker = MetricsTracker()
        return jsonify({"runs": tracker.get_all_runs()})

    print(f"\n  📊 Training Dashboard: http://localhost:{port}")
    print("  Press Ctrl+C to stop.\n")

    if open_browser:
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


def interactive_dashboard():
    """Interactive dashboard launcher."""
    print("\n" + "=" * 55)
    print("       Training Dashboard")
    print("=" * 55)

    tracker = MetricsTracker()
    runs = tracker.get_all_runs()

    if runs:
        print(f"\n  {len(runs)} training run(s) recorded.")
        latest = runs[-1]
        epochs = latest.get("epochs", [])
        if epochs:
            print(f"  Latest: {latest['name']} ({len(epochs)} epochs, "
                  f"loss={epochs[-1]['train_loss']:.4f})")
    else:
        print("\n  No training runs recorded yet.")
        print("  Metrics are logged automatically during training.")

    print("\n  Options:")
    print("  1  Launch web dashboard")
    print("  2  Show metrics summary")
    print("  3  Clear history")
    print("  0  Back")

    try:
        choice = input("\n  dashboard>> ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    if choice == '1':
        try:
            port = input("  Port [8501]: ").strip()
            port = int(port) if port else 8501
            start_dashboard(port=port)
        except (KeyboardInterrupt, EOFError):
            print("\n  Dashboard stopped.")

    elif choice == '2':
        if runs:
            for run in runs[-5:]:
                epochs = run.get("epochs", [])
                print(f"\n  {run['name']} [{run.get('status', '?')}]")
                if epochs:
                    losses = [e["train_loss"] for e in epochs]
                    print(f"    Epochs: {len(epochs)}")
                    print(f"    Start loss: {losses[0]:.4f}")
                    print(f"    End loss:   {losses[-1]:.4f}")
                    print(f"    Best loss:  {min(losses):.4f}")

    elif choice == '3':
        confirm = input("  Clear all training history? [y/N]: ").strip().lower()
        if confirm == 'y':
            tracker.clear_history()
            print("  ✓ History cleared.")

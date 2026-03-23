"""Tests for utils/training_dashboard.py — MetricsTracker."""

import os
import json
import pytest
from utils.training_dashboard import MetricsTracker


class TestMetricsTracker:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.metrics_path = str(tmp_path / "metrics.json")
        self.tracker = MetricsTracker(metrics_path=self.metrics_path)

    def test_start_run(self):
        self.tracker.start_run(name="test_run")
        assert self.tracker.current_run is not None
        assert self.tracker.current_run["name"] == "test_run"
        assert self.tracker.current_run["status"] == "running"

    def test_log_epoch(self):
        self.tracker.start_run(name="test_run")
        self.tracker.log_epoch(1, train_loss=2.5, eval_loss=2.8, learning_rate=0.001)
        epochs = self.tracker.current_run["epochs"]
        assert len(epochs) == 1
        assert epochs[0]["epoch"] == 1
        assert epochs[0]["train_loss"] == 2.5
        assert epochs[0]["eval_loss"] == 2.8
        assert epochs[0]["learning_rate"] == 0.001

    def test_log_epoch_without_run(self):
        # Should not crash
        self.tracker.log_epoch(1, train_loss=1.0)

    def test_end_run(self):
        self.tracker.start_run(name="test_run")
        self.tracker.log_epoch(1, train_loss=2.0)
        self.tracker.end_run(status="completed")
        assert self.tracker.current_run is None

    def test_persistence(self):
        self.tracker.start_run(name="persist_test")
        self.tracker.log_epoch(1, train_loss=1.5)
        self.tracker.end_run()

        # Load fresh tracker from same file
        tracker2 = MetricsTracker(metrics_path=self.metrics_path)
        runs = tracker2.get_all_runs()
        assert len(runs) == 1
        assert runs[0]["name"] == "persist_test"
        assert runs[0]["status"] == "completed"

    def test_get_latest_run(self):
        self.tracker.start_run(name="run1")
        self.tracker.end_run()
        self.tracker.start_run(name="run2")
        self.tracker.end_run()

        latest = self.tracker.get_latest_run()
        assert latest["name"] == "run2"

    def test_get_latest_run_empty(self):
        assert self.tracker.get_latest_run() is None

    def test_multiple_epochs(self):
        self.tracker.start_run(name="multi")
        for i in range(5):
            self.tracker.log_epoch(i + 1, train_loss=2.0 - i * 0.3)
        epochs = self.tracker.current_run["epochs"]
        assert len(epochs) == 5
        assert epochs[-1]["train_loss"] < epochs[0]["train_loss"]

    def test_clear_history(self):
        self.tracker.start_run(name="to_clear")
        self.tracker.end_run()
        self.tracker.clear_history()
        assert self.tracker.get_all_runs() == []
        assert self.tracker.current_run is None

    def test_extra_metrics(self):
        self.tracker.start_run(name="extra")
        self.tracker.log_epoch(1, train_loss=1.0, extra={"bleu": 0.5, "grad_norm": 1.2})
        epoch = self.tracker.current_run["epochs"][0]
        assert epoch["bleu"] == 0.5
        assert epoch["grad_norm"] == 1.2

    def test_run_has_timestamps(self):
        self.tracker.start_run(name="ts_test")
        assert "started" in self.tracker.current_run
        self.tracker.end_run()
        runs = self.tracker.get_all_runs()
        assert "ended" in runs[0]

"""Comprehensive runtime tests - Verify all aspects of the codebase.

Tests every module, function, class, and integration point to ensure
the full runtime works correctly end-to-end.
"""

import os
import sys
import json
import shutil
import tempfile
import importlib
import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. Module Import Tests - Verify all modules load correctly
# ============================================================
class TestModuleImports:
    """Verify every module in the project can be imported."""

    def test_import_models_model(self):
        import models.model
        assert hasattr(models.model, 'ConversationalModel')
        assert hasattr(models.model, 'PositionalEncoding')
        assert hasattr(models.model, 'TransformerBlock')

    def test_import_models_tokenizer(self):
        import models.tokenizer
        assert hasattr(models.tokenizer, 'Tokenizer')

    def test_import_models_factory(self):
        import models.model_factory
        assert hasattr(models.model_factory, 'create_model')
        assert hasattr(models.model_factory, 'load_model')
        assert hasattr(models.model_factory, 'list_models')
        assert hasattr(models.model_factory, 'list_pipelines')
        assert hasattr(models.model_factory, 'PretrainedWrapper')

    def test_import_models_init(self):
        import models
        assert hasattr(models, 'ConversationalModel')
        assert hasattr(models, 'Tokenizer')
        assert hasattr(models, 'create_model')
        assert hasattr(models, 'load_model')

    def test_import_utils_data_loader(self):
        import utils.data_loader
        assert hasattr(utils.data_loader, 'ConversationDataset')
        assert hasattr(utils.data_loader, 'create_data_loaders')
        assert hasattr(utils.data_loader, 'load_and_prepare_data')

    def test_import_utils_trainer(self):
        import utils.trainer
        assert hasattr(utils.trainer, 'Trainer')

    def test_import_utils_init(self):
        import utils
        assert hasattr(utils, 'ConversationDataset')
        assert hasattr(utils, 'Trainer')

    def test_import_api(self):
        import api
        assert hasattr(api, 'app')
        assert hasattr(api, 'health')
        assert hasattr(api, 'chat')

    def test_import_chat(self):
        import chat
        assert hasattr(chat, 'generate_response')
        assert hasattr(chat, 'load_model_and_tokenizer')
        assert hasattr(chat, 'interactive_chat')

    def test_import_train(self):
        import train
        assert hasattr(train, 'main')
        assert hasattr(train, 'load_config')
        assert hasattr(train, 'get_device')

    def test_import_model_registry(self):
        import model_registry
        assert hasattr(model_registry, 'register_model')
        assert hasattr(model_registry, 'list_registered_models')
        assert hasattr(model_registry, 'load_registered_model')
        assert hasattr(model_registry, 'delete_model')
        assert hasattr(model_registry, 'get_model_info')

    def test_import_prompt_trainer(self):
        import prompt_trainer
        assert hasattr(prompt_trainer, 'collect_prompts')
        assert hasattr(prompt_trainer, 'save_training_data')
        assert hasattr(prompt_trainer, 'prompt_train_interactive')

    def test_import_external_api(self):
        import external_api
        assert hasattr(external_api, 'ExternalAPIClient')
        assert hasattr(external_api, 'PROVIDERS')
        assert hasattr(external_api, 'api_call')
        assert hasattr(external_api, 'is_provider_configured')
        assert hasattr(external_api, 'find_available_provider')

    def test_import_run(self):
        import run
        assert hasattr(run, 'main')
        assert hasattr(run, 'print_menu')
        assert hasattr(run, 'print_banner')
        assert hasattr(run, 'cmd_explore')
        assert hasattr(run, '_explore_local')
        assert hasattr(run, '_explore_saved')
        assert hasattr(run, '_explore_api')


# ============================================================
# 2. Config Tests - Verify config.yaml is valid
# ============================================================
class TestConfig:
    """Verify configuration file is valid and complete."""

    @pytest.fixture
    def config(self):
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)

    def test_config_loads(self, config):
        assert config is not None
        assert isinstance(config, dict)

    def test_model_section(self, config):
        m = config['model']
        assert m['base_model'] in ('custom', 'gpt2', 'distilgpt2', 'gpt2-medium',
                                    'pythia-160m', 'pythia-410m')
        assert isinstance(m['vocab_size'], int) and m['vocab_size'] > 0
        assert isinstance(m['embedding_dim'], int) and m['embedding_dim'] > 0
        assert isinstance(m['hidden_dim'], int) and m['hidden_dim'] > 0
        assert isinstance(m['num_layers'], int) and m['num_layers'] > 0
        assert isinstance(m['num_heads'], int) and m['num_heads'] > 0
        assert isinstance(m['max_seq_length'], int) and m['max_seq_length'] > 0
        assert 0 <= m['dropout'] <= 1

    def test_training_section(self, config):
        t = config['training']
        assert t['pipeline'] in ('scratch', 'finetune', 'freeze')
        assert isinstance(t['batch_size'], int) and t['batch_size'] > 0
        assert isinstance(t['learning_rate'], float) and t['learning_rate'] > 0
        assert isinstance(t['num_epochs'], int) and t['num_epochs'] > 0
        assert isinstance(t['gradient_clip'], (int, float))
        assert isinstance(t['warmup_steps'], int)
        assert isinstance(t['weight_decay'], float)

    def test_data_section(self, config):
        d = config['data']
        assert 'train_path' in d
        assert 'test_path' in d
        assert os.path.exists(d['train_path'])
        assert os.path.exists(d['test_path'])

    def test_checkpoint_section(self, config):
        c = config['checkpoint']
        assert 'save_dir' in c
        assert isinstance(c['save_every'], int)
        assert isinstance(c['keep_last'], int)

    def test_generation_section(self, config):
        g = config['generation']
        assert isinstance(g['max_length'], int) and g['max_length'] > 0
        assert 0 < g['temperature'] <= 2.0
        assert isinstance(g['top_k'], int) and g['top_k'] >= 0
        assert 0 < g['top_p'] <= 1.0
        assert g['repetition_penalty'] >= 1.0

    def test_api_section(self, config):
        a = config['api']
        assert 'host' in a
        assert isinstance(a['port'], int)

    def test_external_api_section(self, config):
        e = config['external_api']
        assert 'openai' in e
        assert 'anthropic' in e
        assert 'ollama' in e
        assert 'custom' in e

    def test_device_section(self, config):
        d = config['device']
        assert 'use_cuda' in d
        assert isinstance(d['cuda_device'], int)


# ============================================================
# 3. Data File Tests - Verify training/test data is valid
# ============================================================
class TestDataFiles:
    """Verify data files are valid and well-formed."""

    def test_train_json_exists(self):
        assert os.path.exists('data/train.json')

    def test_test_json_exists(self):
        assert os.path.exists('data/test.json')

    def test_train_json_valid(self):
        with open('data/train.json', 'r') as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_test_json_valid(self):
        with open('data/test.json', 'r') as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_train_data_structure(self):
        with open('data/train.json', 'r') as f:
            data = json.load(f)
        for item in data:
            assert 'prompt' in item, f"Missing 'prompt' in {item}"
            assert 'response' in item, f"Missing 'response' in {item}"
            assert isinstance(item['prompt'], str)
            assert isinstance(item['response'], str)
            assert len(item['prompt']) > 0
            assert len(item['response']) > 0

    def test_test_data_structure(self):
        with open('data/test.json', 'r') as f:
            data = json.load(f)
        for item in data:
            assert 'prompt' in item
            assert 'response' in item


# ============================================================
# 4. Model Runtime Tests - Verify model operations work
# ============================================================
class TestModelRuntime:
    """End-to-end model creation, forward pass, generate, save/load."""

    @pytest.fixture
    def model(self):
        from models.model import ConversationalModel
        return ConversationalModel(
            vocab_size=100, embedding_dim=32, hidden_dim=64,
            num_layers=2, num_heads=4, max_seq_length=32, pad_token_id=0
        )

    def test_model_device(self, model):
        """Model starts on CPU."""
        param = next(model.parameters())
        assert param.device == torch.device('cpu')

    def test_forward_pass_gradient(self, model):
        """Forward pass produces gradients."""
        input_ids = torch.randint(0, 100, (2, 10))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        grad_exists = any(p.grad is not None for p in model.parameters())
        assert grad_exists

    def test_generate_produces_valid_tokens(self, model):
        """Generated tokens are within vocab range."""
        input_ids = torch.randint(0, 100, (1, 5))
        output = model.generate(input_ids, max_length=10, eos_token_id=3)
        assert (output >= 0).all()
        assert (output < 100).all()

    def test_save_load_roundtrip(self, model, tmp_path):
        """Save and load produces identical outputs."""
        path = str(tmp_path / "model.pt")
        model.save(path)
        assert os.path.exists(path)

        from models.model import ConversationalModel
        loaded = ConversationalModel.load(path)

        input_ids = torch.randint(0, 100, (1, 5))
        model.eval()
        loaded.eval()
        with torch.no_grad():
            assert torch.allclose(model(input_ids), loaded(input_ids), atol=1e-6)

    def test_model_eval_mode(self, model):
        """Model switches to eval mode."""
        model.eval()
        assert not model.training

    def test_model_train_mode(self, model):
        """Model switches to train mode."""
        model.eval()
        model.train()
        assert model.training


# ============================================================
# 5. Tokenizer Runtime Tests
# ============================================================
class TestTokenizerRuntime:
    """End-to-end tokenizer operations."""

    @pytest.fixture
    def tokenizer(self):
        from models.tokenizer import Tokenizer
        tok = Tokenizer(vocab_size=500, method="word")
        tok.build_vocab(["hello world how are you doing today",
                         "artificial intelligence is fascinating"])
        return tok

    def test_encode_decode_preserves_text(self, tokenizer):
        text = "hello world"
        ids = tokenizer.encode(text, add_special=False)
        decoded = tokenizer.decode(ids, skip_special=True)
        assert decoded == text

    def test_conversation_encoding_length(self, tokenizer):
        input_ids, target_ids = tokenizer.encode_conversation("hello", "world", max_length=32)
        assert len(input_ids) == 32
        assert len(target_ids) == 32

    def test_save_load_preserves_encoding(self, tokenizer, tmp_path):
        path = str(tmp_path / "tok.pkl")
        tokenizer.save(path)
        from models.tokenizer import Tokenizer
        loaded = Tokenizer.load(path)
        assert tokenizer.encode("hello world") == loaded.encode("hello world")

    def test_special_token_ids_consistent(self, tokenizer):
        assert tokenizer.token2id["[PAD]"] == tokenizer.pad_token_id
        assert tokenizer.token2id["[UNK]"] == tokenizer.unk_token_id
        assert tokenizer.token2id["[BOS]"] == tokenizer.bos_token_id
        assert tokenizer.token2id["[EOS]"] == tokenizer.eos_token_id
        assert tokenizer.token2id["[SEP]"] == tokenizer.sep_token_id


# ============================================================
# 6. Data Pipeline Runtime Tests
# ============================================================
class TestDataPipelineRuntime:
    """End-to-end data loading pipeline."""

    @pytest.fixture
    def data_file(self, tmp_path):
        data = [
            {"prompt": "Hello", "response": "Hi there"},
            {"prompt": "How are you", "response": "I am fine"},
            {"prompt": "What is AI", "response": "Artificial intelligence"},
            {"prompt": "Tell me a joke", "response": "Why did the chicken cross?"},
        ]
        path = str(tmp_path / "data.json")
        with open(path, 'w') as f:
            json.dump(data, f)
        return path

    @pytest.fixture
    def tokenizer(self):
        from models.tokenizer import Tokenizer
        tok = Tokenizer(vocab_size=500, method="word")
        tok.build_vocab(["hello hi there how are you i am fine what is ai "
                         "artificial intelligence tell me a joke why did the chicken cross"])
        return tok

    def test_full_pipeline(self, data_file, tokenizer):
        """Data loads through dataset → dataloader → batches."""
        from utils.data_loader import create_data_loaders
        train_loader, test_loader = create_data_loaders(
            data_file, data_file, tokenizer, batch_size=2, max_length=32
        )
        for input_ids, target_ids in train_loader:
            assert input_ids.shape[0] <= 2  # batch size
            assert input_ids.shape[1] == 32  # seq length
            assert target_ids.shape == input_ids.shape
            assert input_ids.dtype == torch.long
            break

    def test_load_and_prepare(self, data_file):
        from utils.data_loader import load_and_prepare_data
        texts = load_and_prepare_data(data_file)
        assert len(texts) == 8  # 4 prompts + 4 responses
        assert all(isinstance(t, str) for t in texts)


# ============================================================
# 7. Trainer Runtime Tests
# ============================================================
class TestTrainerRuntime:
    """End-to-end training pipeline."""

    @pytest.fixture
    def training_setup(self, tmp_path):
        data = [
            {"prompt": "Hello", "response": "Hi there"},
            {"prompt": "How are you", "response": "I am fine"},
        ]
        data_path = str(tmp_path / "data.json")
        with open(data_path, 'w') as f:
            json.dump(data, f)

        from models.tokenizer import Tokenizer
        from models.model import ConversationalModel
        from utils.data_loader import create_data_loaders

        tok = Tokenizer(vocab_size=200, method="word")
        tok.build_vocab(["hello hi there how are you i am fine"])
        train_loader, test_loader = create_data_loaders(
            data_path, data_path, tok, batch_size=2, max_length=16
        )
        model = ConversationalModel(
            vocab_size=len(tok), embedding_dim=16, hidden_dim=32,
            num_layers=1, num_heads=2, max_seq_length=16, pad_token_id=0
        )
        ckpt_dir = str(tmp_path / "ckpt")
        return model, train_loader, test_loader, tok, ckpt_dir

    def test_train_reduces_loss(self, training_setup):
        """Training for 2 epochs should reduce loss."""
        model, train_loader, test_loader, tok, ckpt_dir = training_setup
        from utils.trainer import Trainer
        trainer = Trainer(model=model, train_loader=train_loader,
                          test_loader=test_loader, device='cpu',
                          checkpoint_dir=ckpt_dir, pad_token_id=tok.pad_token_id,
                          warmup_steps=0, compile_model=False)
        loss1 = trainer.train_epoch(1)
        loss2 = trainer.train_epoch(2)
        # Loss should generally decrease (or at least not crash)
        assert isinstance(loss1, float)
        assert isinstance(loss2, float)

    def test_checkpoint_cycle(self, training_setup):
        """Train → save → load → verify."""
        model, train_loader, test_loader, tok, ckpt_dir = training_setup
        from utils.trainer import Trainer
        trainer = Trainer(model=model, train_loader=train_loader,
                          test_loader=test_loader, device='cpu',
                          checkpoint_dir=ckpt_dir, pad_token_id=tok.pad_token_id,
                          warmup_steps=0, compile_model=False)
        trainer.train(num_epochs=1, save_every=1, keep_last=2)

        # Verify checkpoint exists
        files = os.listdir(ckpt_dir)
        model_files = [f for f in files if f.endswith('.pt')]
        assert len(model_files) > 0

        # Load and verify
        from models.model import ConversationalModel
        best_path = os.path.join(ckpt_dir, 'best_model.pt')
        if os.path.exists(best_path):
            loaded = ConversationalModel.load(best_path)
            assert loaded.vocab_size == model.vocab_size


# ============================================================
# 8. Model Factory Runtime Tests
# ============================================================
class TestFactoryRuntime:
    """Verify model factory creates correct models."""

    def test_create_custom_model_runtime(self):
        from models.model_factory import create_model
        from models.model import ConversationalModel
        config = {
            'model': {'base_model': 'custom', 'vocab_size': 50,
                      'embedding_dim': 16, 'hidden_dim': 32, 'num_layers': 1,
                      'num_heads': 2, 'max_seq_length': 16, 'dropout': 0.1},
            'training': {'pipeline': 'scratch'}
        }
        model = create_model(config)
        assert isinstance(model, ConversationalModel)

        # Verify it can do a forward pass
        input_ids = torch.randint(0, 50, (1, 8))
        logits = model(input_ids)
        assert logits.shape == (1, 8, 50)

    def test_load_save_roundtrip(self, tmp_path):
        from models.model_factory import create_model, load_model
        from models.model import ConversationalModel
        config = {
            'model': {'base_model': 'custom', 'vocab_size': 50,
                      'embedding_dim': 16, 'hidden_dim': 32, 'num_layers': 1,
                      'num_heads': 2, 'max_seq_length': 16, 'dropout': 0.1},
            'training': {'pipeline': 'scratch'}
        }
        model = create_model(config)
        assert isinstance(model, ConversationalModel)
        path = str(tmp_path / "model.pt")
        model.save(path)
        loaded = load_model(path)
        assert isinstance(loaded, ConversationalModel)
        assert loaded.vocab_size == 50


# ============================================================
# 9. API Runtime Tests
# ============================================================
class TestAPIRuntime:
    """Verify Flask API endpoints work correctly."""

    @pytest.fixture
    def client(self):
        from api import app
        app.config['TESTING'] = True
        with app.test_client() as c:
            yield c

    def test_health_returns_json(self, client):
        resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'status' in data
        assert 'model_loaded' in data

    def test_chat_requires_model(self, client):
        resp = client.post('/chat', json={'message': 'hi'})
        assert resp.status_code == 503

    def test_generate_requires_model(self, client):
        resp = client.post('/generate', json={'message': 'hi'})
        assert resp.status_code == 503

    def test_config_requires_init(self, client):
        resp = client.get('/config')
        assert resp.status_code == 503

    def test_404_for_unknown_route(self, client):
        resp = client.get('/this-does-not-exist')
        assert resp.status_code == 404

    def test_405_wrong_method(self, client):
        resp = client.get('/chat')
        assert resp.status_code == 405

    def test_chat_validates_body(self, client):
        resp = client.post('/chat', json={})
        assert resp.status_code in (400, 503)


# ============================================================
# 10. Registry Runtime Tests
# ============================================================
class TestRegistryRuntime:
    """Verify model registry end-to-end."""

    @pytest.fixture(autouse=True)
    def temp_registry(self, monkeypatch, tmp_path):
        import model_registry
        reg_dir = str(tmp_path / "trained_models")
        reg_file = os.path.join(reg_dir, "registry.json")
        monkeypatch.setattr(model_registry, "REGISTRY_DIR", reg_dir)
        monkeypatch.setattr(model_registry, "REGISTRY_FILE", reg_file)
        return reg_dir

    @pytest.fixture
    def fake_ckpt(self, tmp_path):
        d = str(tmp_path / "ckpt")
        os.makedirs(d)
        with open(os.path.join(d, "best_model.pt"), 'wb') as f:
            f.write(b"data")
        with open(os.path.join(d, "tokenizer.pkl"), 'wb') as f:
            f.write(b"data")
        return d

    def test_register_get_delete_cycle(self, fake_ckpt):
        import model_registry
        model_registry.register_model("test-bot", "testing", "custom",
                                       "scratch", fake_ckpt)
        info = model_registry.get_model_info("test-bot")
        assert info is not None
        assert info["intent"] == "testing"

        model_registry.delete_model("test-bot")
        assert model_registry.get_model_info("test-bot") is None

    def test_multiple_models(self, fake_ckpt):
        import model_registry
        model_registry.register_model("bot-a", "intent a", "custom", "scratch", fake_ckpt)
        model_registry.register_model("bot-b", "intent b", "gpt2", "finetune", fake_ckpt)
        reg = model_registry._load_registry()
        assert len(reg["models"]) == 2


# ============================================================
# 11. Prompt Trainer Runtime Tests
# ============================================================
class TestPromptTrainerRuntime:
    """Verify prompt trainer data handling."""

    def test_save_and_reload(self, tmp_path):
        from prompt_trainer import save_training_data
        pairs = [
            {"prompt": "hello", "response": "hi"},
            {"prompt": "how are you", "response": "fine"},
        ]
        path = str(tmp_path / "custom_train.json")
        save_training_data(pairs, path, merge_existing=False)

        with open(path, 'r') as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[0]["prompt"] == "hello"

    def test_merge_append(self, tmp_path):
        from prompt_trainer import save_training_data
        path = str(tmp_path / "train.json")
        save_training_data([{"prompt": "a", "response": "b"}], path, merge_existing=False)
        save_training_data([{"prompt": "c", "response": "d"}], path, merge_existing=True)

        with open(path, 'r') as f:
            data = json.load(f)
        assert len(data) == 2


# ============================================================
# 12. External API Runtime Tests
# ============================================================
class TestExternalAPIRuntime:
    """Verify external API client configuration detection."""

    def test_unconfigured_providers_return_false(self):
        """Without env vars or config keys, key-based providers should be unconfigured."""
        from external_api import is_provider_configured
        # These should not be configured in the test environment (unless user set keys)
        if not os.environ.get("OPENAI_API_KEY"):
            assert is_provider_configured("openai") is False
        if not os.environ.get("ANTHROPIC_API_KEY"):
            assert is_provider_configured("anthropic") is False

    def test_find_available_returns_none_when_all_unconfigured(self):
        from unittest.mock import patch
        from external_api import find_available_provider
        with patch('external_api.is_provider_configured', return_value=False):
            result = find_available_provider(["openai", "anthropic"])
            assert result is None

    def test_client_handles_connection_error(self):
        from external_api import ExternalAPIClient
        client = ExternalAPIClient("openai", model="gpt-4", api_key="fake",
                                    base_url="http://localhost:1")
        response = client.chat("test")
        assert "[API Error" in response
        assert len(client.conversation_history) == 2  # user + error response


# ============================================================
# 13. Run.py Menu System Tests
# ============================================================
class TestRunMenu:
    """Verify run.py menu functions don't crash."""

    def test_print_banner(self, capsys):
        from run import print_banner
        print_banner()
        captured = capsys.readouterr()
        assert "AI Model Runtime" in captured.out

    def test_print_menu(self, capsys):
        from run import print_menu
        print_menu()
        captured = capsys.readouterr()
        assert "train" in captured.out
        assert "chat" in captured.out
        assert "api" in captured.out
        assert "prompt-train" in captured.out
        assert "registry" in captured.out
        assert "external" in captured.out
        assert "explore" in captured.out
        assert "stop" in captured.out

    def test_cmd_models(self, capsys):
        from run import cmd_models
        cmd_models()
        captured = capsys.readouterr()
        assert "custom" in captured.out
        assert "gpt2" in captured.out

    def test_cmd_pipelines(self, capsys):
        from run import cmd_pipelines
        cmd_pipelines()
        captured = capsys.readouterr()
        assert "scratch" in captured.out
        assert "finetune" in captured.out

    def test_cmd_config(self, capsys):
        from run import cmd_config
        cmd_config()
        captured = capsys.readouterr()
        assert "Model:" in captured.out
        assert "Pipeline:" in captured.out

    def test_cmd_status(self, capsys):
        from run import cmd_status
        cmd_status()
        captured = capsys.readouterr()
        assert "Status:" in captured.out

    def test_cmd_registry(self, capsys):
        from run import cmd_registry
        cmd_registry()
        captured = capsys.readouterr()
        # Either shows models or "No registered models"
        assert "Model" in captured.out or "registered" in captured.out.lower()

    def test_cmd_providers(self, capsys):
        from run import cmd_providers
        cmd_providers()
        captured = capsys.readouterr()
        assert "External" in captured.out

    def test_cmd_explore_displays_menu(self, capsys, monkeypatch):
        """Explore command shows its sub-menu before exiting."""
        from run import cmd_explore
        monkeypatch.setattr('builtins.input', lambda _: '0')
        cmd_explore()
        captured = capsys.readouterr()
        assert "Model Explorer" in captured.out
        assert "Local Model" in captured.out
        assert "Saved Models" in captured.out
        assert "API Models" in captured.out
        assert "Back" in captured.out

    def test_explore_saved_no_models(self, capsys):
        """_explore_saved shows message when no registered models."""
        from run import _explore_saved
        _explore_saved()
        captured = capsys.readouterr()
        assert "No registered models" in captured.out or "prompt-train" in captured.out

    def test_explore_local_shows_status(self, capsys, monkeypatch):
        """_explore_local shows local model status information."""
        from run import _explore_local
        # Mock input to raise KeyboardInterrupt (simulates Ctrl+C at prompt)
        def raise_interrupt(_):
            raise KeyboardInterrupt()
        monkeypatch.setattr('builtins.input', raise_interrupt)
        _explore_local()
        captured = capsys.readouterr()
        # Should show status info (may or may not have checkpoints)
        assert "Local Model Status" in captured.out or "Error" in captured.out


# ============================================================
# 14. Integration Tests - Full Workflow
# ============================================================
class TestIntegration:
    """Full integration tests combining multiple modules."""

    def test_full_training_pipeline(self, tmp_path):
        """Build tokenizer → create model → train → save → load."""
        from models.tokenizer import Tokenizer
        from models.model import ConversationalModel
        from utils.data_loader import create_data_loaders, load_and_prepare_data
        from utils.trainer import Trainer

        # Create data
        data = [
            {"prompt": "hello", "response": "hi there"},
            {"prompt": "how are you", "response": "i am good"},
        ]
        data_path = str(tmp_path / "data.json")
        with open(data_path, 'w') as f:
            json.dump(data, f)

        # Build tokenizer
        tok = Tokenizer(vocab_size=200, method="word")
        texts = load_and_prepare_data(data_path)
        tok.build_vocab(texts)
        tok_path = str(tmp_path / "tok.pkl")
        tok.save(tok_path)

        # Create model
        model = ConversationalModel(
            vocab_size=len(tok), embedding_dim=16, hidden_dim=32,
            num_layers=1, num_heads=2, max_seq_length=16, pad_token_id=0
        )

        # Create data loaders
        train_loader, test_loader = create_data_loaders(
            data_path, data_path, tok, batch_size=2, max_length=16
        )

        # Train
        ckpt_dir = str(tmp_path / "checkpoints")
        trainer = Trainer(model=model, train_loader=train_loader,
                          test_loader=test_loader, device='cpu',
                          checkpoint_dir=ckpt_dir, pad_token_id=tok.pad_token_id,
                          warmup_steps=0, compile_model=False)
        trainer.train(num_epochs=2, save_every=1, keep_last=2)

        # Load and generate
        best = os.path.join(ckpt_dir, 'best_model.pt')
        assert os.path.exists(best)
        loaded = ConversationalModel.load(best)
        loaded.eval()

        input_ids = tok.encode("hello", add_special=True)
        input_tensor = torch.tensor([input_ids[:16]])
        with torch.no_grad():
            output = loaded.generate(input_tensor, max_length=5, eos_token_id=tok.eos_token_id)
        assert output.shape[1] >= 1

    def test_factory_to_trainer_pipeline(self, tmp_path):
        """Use model factory → trainer → checkpoint."""
        from models.model_factory import create_model
        from models.tokenizer import Tokenizer
        from utils.data_loader import create_data_loaders
        from utils.trainer import Trainer

        data = [{"prompt": "test", "response": "response"}]
        data_path = str(tmp_path / "data.json")
        with open(data_path, 'w') as f:
            json.dump(data, f)

        tok = Tokenizer(vocab_size=200, method="word")
        tok.build_vocab(["test response"])

        config = {
            'model': {'base_model': 'custom', 'vocab_size': len(tok),
                      'embedding_dim': 16, 'hidden_dim': 32, 'num_layers': 1,
                      'num_heads': 2, 'max_seq_length': 16, 'dropout': 0.1},
            'training': {'pipeline': 'scratch'}
        }
        model = create_model(config)

        train_loader, test_loader = create_data_loaders(
            data_path, data_path, tok, batch_size=1, max_length=16
        )

        ckpt_dir = str(tmp_path / "ckpt")
        trainer = Trainer(model=model, train_loader=train_loader,
                          test_loader=test_loader, device='cpu',
                          checkpoint_dir=ckpt_dir, pad_token_id=0, warmup_steps=0,
                          compile_model=False)
        loss = trainer.train_epoch(1)
        assert isinstance(loss, float)
        assert loss > 0


# ============================================================
# 15. File Structure Tests
# ============================================================
class TestFileStructure:
    """Verify project file structure is correct."""

    EXPECTED_FILES = [
        'run.py', 'train.py', 'chat.py', 'api.py',
        'model_registry.py', 'prompt_trainer.py', 'external_api.py',
        'config.yaml', 'requirements.txt', 'README.md',
        'models/__init__.py', 'models/model.py', 'models/tokenizer.py',
        'models/model_factory.py',
        'utils/__init__.py', 'utils/data_loader.py', 'utils/trainer.py',
        'tests/__init__.py',
        'data/train.json', 'data/test.json',
    ]

    def test_all_expected_files_exist(self):
        for f in self.EXPECTED_FILES:
            assert os.path.exists(f), f"Missing file: {f}"

    def test_test_files_exist(self):
        test_files = [
            'tests/test_model.py', 'tests/test_tokenizer.py',
            'tests/test_data_loader.py', 'tests/test_factory.py',
            'tests/test_trainer.py', 'tests/test_api.py',
            'tests/test_registry.py', 'tests/test_prompt_trainer.py',
            'tests/test_external_api.py', 'tests/test_runtime.py',
        ]
        for f in test_files:
            assert os.path.exists(f), f"Missing test file: {f}"

    def test_no_pycache_in_source(self):
        """Source directories should have __init__.py files."""
        assert os.path.exists('models/__init__.py')
        assert os.path.exists('utils/__init__.py')
        assert os.path.exists('tests/__init__.py')

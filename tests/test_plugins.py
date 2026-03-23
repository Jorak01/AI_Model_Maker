"""Tests for the plugin system — HookSystem and PluginManager."""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHookSystem:
    def test_valid_hooks_defined(self):
        from plugins.loader import HookSystem
        hs = HookSystem()
        assert len(hs.VALID_HOOKS) > 0
        assert 'on_train_start' in hs.VALID_HOOKS
        assert 'on_generate_complete' in hs.VALID_HOOKS
        assert 'on_model_loaded' in hs.VALID_HOOKS

    def test_register_and_trigger(self):
        from plugins.loader import HookSystem
        hs = HookSystem()
        results = []
        hs.register('on_train_start', lambda **kw: results.append('fired'))
        hs.trigger('on_train_start')
        assert results == ['fired']

    def test_trigger_returns_results(self):
        from plugins.loader import HookSystem
        hs = HookSystem()
        hs.register('on_train_end', lambda **kw: 42)
        out = hs.trigger('on_train_end')
        assert out == [42]

    def test_trigger_passes_kwargs(self):
        from plugins.loader import HookSystem
        hs = HookSystem()
        hs.register('on_epoch_end', lambda epoch=0, **kw: epoch * 2)
        out = hs.trigger('on_epoch_end', epoch=5)
        assert out == [10]

    def test_unregister(self):
        from plugins.loader import HookSystem
        hs = HookSystem()
        cb = lambda **kw: 'x'
        hs.register('on_startup', cb)
        hs.unregister('on_startup', cb)
        assert hs.trigger('on_startup') == []

    def test_register_unknown_hook(self, capsys):
        from plugins.loader import HookSystem
        hs = HookSystem()
        hs.register('nonexistent', lambda: None)
        captured = capsys.readouterr()
        assert 'Unknown hook' in captured.out

    def test_list_hooks_empty(self):
        from plugins.loader import HookSystem
        hs = HookSystem()
        assert hs.list_hooks() == {}

    def test_list_hooks_with_registered(self):
        from plugins.loader import HookSystem
        hs = HookSystem()
        hs.register('on_startup', lambda **kw: None)
        hs.register('on_startup', lambda **kw: None)
        hooks = hs.list_hooks()
        assert hooks['on_startup'] == 2

    def test_trigger_handles_callback_error(self, capsys):
        from plugins.loader import HookSystem
        hs = HookSystem()
        def bad_cb(**kw):
            raise ValueError("boom")
        hs.register('on_shutdown', bad_cb)
        results = hs.trigger('on_shutdown')
        assert results == []
        assert 'Hook on_shutdown error' in capsys.readouterr().out

    def test_multiple_callbacks(self):
        from plugins.loader import HookSystem
        hs = HookSystem()
        hs.register('on_train_start', lambda **kw: 1)
        hs.register('on_train_start', lambda **kw: 2)
        assert hs.trigger('on_train_start') == [1, 2]


class TestPluginManager:
    def test_create_manager(self):
        from plugins.loader import PluginManager
        pm = PluginManager(plugin_dir="plugins")
        assert pm.plugin_dir == "plugins"
        assert isinstance(pm.plugins, dict)
        assert isinstance(pm.commands, dict)

    def test_discover_no_plugins(self, tmp_path):
        from plugins.loader import PluginManager
        pm = PluginManager(plugin_dir=str(tmp_path / "empty_plugins"))
        assert pm.discover() == []

    def test_discover_finds_plugin(self, tmp_path):
        from plugins.loader import PluginManager
        plugin_dir = tmp_path / "plugins"
        p = plugin_dir / "test_plugin"
        p.mkdir(parents=True)
        (p / "manifest.json").write_text(json.dumps({"name": "Test", "version": "1.0"}))
        pm = PluginManager(plugin_dir=str(plugin_dir))
        found = pm.discover()
        assert "test_plugin" in found

    def test_load_plugin_no_manifest(self, tmp_path):
        from plugins.loader import PluginManager
        plugin_dir = tmp_path / "plugins"
        (plugin_dir / "bad").mkdir(parents=True)
        pm = PluginManager(plugin_dir=str(plugin_dir))
        assert pm.load_plugin("bad") is False

    def test_load_plugin_with_manifest_only(self, tmp_path):
        from plugins.loader import PluginManager
        plugin_dir = tmp_path / "plugins"
        p = plugin_dir / "simple"
        p.mkdir(parents=True)
        manifest = {"name": "Simple Plugin", "version": "1.0", "description": "test"}
        (p / "manifest.json").write_text(json.dumps(manifest))
        pm = PluginManager(plugin_dir=str(plugin_dir))
        assert pm.load_plugin("simple") is True
        assert "simple" in pm.plugins
        assert pm.plugins["simple"]["manifest"]["name"] == "Simple Plugin"

    def test_load_plugin_with_main(self, tmp_path):
        from plugins.loader import PluginManager
        plugin_dir = tmp_path / "plugins"
        p = plugin_dir / "full"
        p.mkdir(parents=True)
        manifest = {"name": "Full", "version": "1.0", "commands": {"my-cmd": "my_func"},
                     "hooks": ["on_startup"], "menu_entries": [{"key": "my-cmd", "label": "My"}]}
        (p / "manifest.json").write_text(json.dumps(manifest))
        (p / "main.py").write_text("def my_func(): return 'hello'\ndef handle_on_startup(**kw): pass")
        pm = PluginManager(plugin_dir=str(plugin_dir))
        assert pm.load_plugin("full") is True
        assert "my-cmd" in pm.commands
        assert len(pm.menu_entries) == 1

    def test_run_command(self, tmp_path):
        from plugins.loader import PluginManager
        plugin_dir = tmp_path / "plugins"
        p = plugin_dir / "cmd"
        p.mkdir(parents=True)
        (p / "manifest.json").write_text(json.dumps({"name": "Cmd", "version": "1", "commands": {"do": "do_it"}}))
        (p / "main.py").write_text("def do_it(): return 'done'")
        pm = PluginManager(plugin_dir=str(plugin_dir))
        pm.load_plugin("cmd")
        assert pm.run_command("do") == "done"

    def test_unload_plugin(self, tmp_path):
        from plugins.loader import PluginManager
        plugin_dir = tmp_path / "plugins"
        p = plugin_dir / "removable"
        p.mkdir(parents=True)
        (p / "manifest.json").write_text(json.dumps({"name": "R", "version": "1", "commands": {"rm": "rm_func"}}))
        (p / "main.py").write_text("def rm_func(): pass")
        pm = PluginManager(plugin_dir=str(plugin_dir))
        pm.load_plugin("removable")
        assert "removable" in pm.plugins
        pm.unload_plugin("removable")
        assert "removable" not in pm.plugins
        assert "rm" not in pm.commands

    def test_load_all(self, tmp_path):
        from plugins.loader import PluginManager
        plugin_dir = tmp_path / "plugins"
        for name in ["a", "b"]:
            p = plugin_dir / name
            p.mkdir(parents=True)
            (p / "manifest.json").write_text(json.dumps({"name": name, "version": "1"}))
        pm = PluginManager(plugin_dir=str(plugin_dir))
        loaded = pm.load_all(verbose=False)
        assert loaded == 2

    def test_get_plugin_manager_singleton(self):
        from plugins.loader import get_plugin_manager
        pm1 = get_plugin_manager()
        pm2 = get_plugin_manager()
        assert pm1 is pm2

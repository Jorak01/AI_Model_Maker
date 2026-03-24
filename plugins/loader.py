"""Plugin loader and hook system for the AI Model Suite.

Plugins are directories under plugins/ with a manifest.json and a main.py.
The manifest defines menu entries, commands, and hooks.

Example plugin structure:
    plugins/
        my_plugin/
            manifest.json
            main.py

manifest.json:
{
    "name": "My Plugin",
    "version": "1.0",
    "description": "Does something cool",
    "commands": {"my-cmd": "cmd_my_command"},
    "hooks": ["on_train_start", "on_generate_complete"],
    "menu_category": "Plugins",
    "menu_entries": [{"key": "my-cmd", "label": "My Command"}]
}
"""

import os
import json
import importlib
import importlib.util
from typing import Dict, List, Callable, Any, Optional


# ---------------------------------------------------------------------------
# Hook System
# ---------------------------------------------------------------------------

class HookSystem:
    """Event hook system — plugins register callbacks for lifecycle events."""

    VALID_HOOKS = [
        'on_train_start', 'on_train_end', 'on_epoch_start', 'on_epoch_end',
        'on_generate_start', 'on_generate_complete',
        'on_model_loaded', 'on_model_saved',
        'on_chat_message', 'on_chat_response',
        'on_image_generated', 'on_dataset_created',
        'on_plugin_loaded', 'on_startup', 'on_shutdown',
    ]

    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {h: [] for h in self.VALID_HOOKS}

    def register(self, hook_name: str, callback: Callable):
        """Register a callback for a hook event."""
        if hook_name not in self._hooks:
            print(f"  ⚠ Unknown hook: {hook_name}")
            return
        self._hooks[hook_name].append(callback)

    def unregister(self, hook_name: str, callback: Callable):
        """Remove a callback from a hook."""
        if hook_name in self._hooks:
            self._hooks[hook_name] = [c for c in self._hooks[hook_name] if c != callback]

    def trigger(self, hook_name: str, **kwargs) -> List[Any]:
        """Fire all callbacks for a hook, return list of results."""
        results = []
        for cb in self._hooks.get(hook_name, []):
            try:
                result = cb(**kwargs)
                results.append(result)
            except Exception as e:
                print(f"  ⚠ Hook {hook_name} error in {cb.__name__}: {e}")
        return results

    def list_hooks(self) -> Dict[str, int]:
        """Return hook names and number of registered callbacks."""
        return {h: len(cbs) for h, cbs in self._hooks.items() if cbs}


# ---------------------------------------------------------------------------
# Plugin Manager
# ---------------------------------------------------------------------------

class PluginManager:
    """Discovers, loads, and manages plugins."""

    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = plugin_dir
        self.hooks = HookSystem()
        self.plugins: Dict[str, Dict] = {}  # name -> {manifest, module, commands}
        self.commands: Dict[str, Callable] = {}  # command_key -> function
        self.menu_entries: List[Dict] = []  # [{key, label, category}]

    def discover(self) -> List[str]:
        """Find all plugin directories with manifest.json."""
        found = []
        if not os.path.isdir(self.plugin_dir):
            return found
        for name in os.listdir(self.plugin_dir):
            plugin_path = os.path.join(self.plugin_dir, name)
            manifest_path = os.path.join(plugin_path, "manifest.json")
            if os.path.isdir(plugin_path) and os.path.exists(manifest_path):
                found.append(name)
        return found

    def load_plugin(self, name: str) -> bool:
        """Load a single plugin by directory name."""
        plugin_path = os.path.join(self.plugin_dir, name)
        manifest_path = os.path.join(plugin_path, "manifest.json")
        main_path = os.path.join(plugin_path, "main.py")

        if not os.path.exists(manifest_path):
            print(f"  ⚠ Plugin '{name}': missing manifest.json")
            return False

        # Load manifest
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ⚠ Plugin '{name}': invalid manifest: {e}")
            return False

        # Load module
        module = None
        if os.path.exists(main_path):
            try:
                spec = importlib.util.spec_from_file_location(
                    f"plugins.{name}.main", main_path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  # type: ignore
            except Exception as e:
                print(f"  ⚠ Plugin '{name}': failed to load main.py: {e}")
                return False

        # Register commands
        commands = manifest.get("commands", {})
        plugin_commands = {}
        for cmd_key, func_name in commands.items():
            if module and hasattr(module, func_name):
                func = getattr(module, func_name)
                self.commands[cmd_key] = func
                plugin_commands[cmd_key] = func

        # Register hooks
        hook_names = manifest.get("hooks", [])
        for hook_name in hook_names:
            handler_name = f"handle_{hook_name}"
            if module and hasattr(module, handler_name):
                self.hooks.register(hook_name, getattr(module, handler_name))

        # Register menu entries
        menu_entries = manifest.get("menu_entries", [])
        category = manifest.get("menu_category", "Plugins")
        for entry in menu_entries:
            entry["category"] = category
            self.menu_entries.append(entry)

        # Store plugin info
        self.plugins[name] = {
            "manifest": manifest,
            "module": module,
            "commands": plugin_commands,
            "path": plugin_path,
        }

        self.hooks.trigger('on_plugin_loaded', plugin_name=name, manifest=manifest)
        return True

    def load_all(self, verbose: bool = True) -> int:
        """Discover and load all plugins."""
        discovered = self.discover()
        loaded = 0
        for name in discovered:
            if self.load_plugin(name):
                if verbose:
                    manifest = self.plugins[name]["manifest"]
                    print(f"  ✓ Plugin: {manifest.get('name', name)} v{manifest.get('version', '?')}")
                loaded += 1
        return loaded

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        if name not in self.plugins:
            return False
        plugin = self.plugins[name]
        for cmd_key in plugin["commands"]:
            self.commands.pop(cmd_key, None)
        self.menu_entries = [e for e in self.menu_entries
                             if e.get("key") not in plugin["commands"]]
        del self.plugins[name]
        return True

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return a list of plugin info dicts.

        Each dict contains: name, version, description, loaded (bool),
        num_commands, and path.
        """
        result: List[Dict[str, Any]] = []
        # Include loaded plugins
        for name, info in self.plugins.items():
            m = info["manifest"]
            result.append({
                "name": m.get("name", name),
                "version": m.get("version", "?"),
                "description": m.get("description", ""),
                "loaded": True,
                "num_commands": len(info["commands"]),
                "path": info["path"],
            })
        # Include discovered-but-not-loaded plugins
        for dir_name in self.discover():
            if dir_name not in self.plugins:
                manifest_path = os.path.join(self.plugin_dir, dir_name, "manifest.json")
                try:
                    with open(manifest_path, 'r') as f:
                        m = json.load(f)
                except Exception:
                    m = {}
                result.append({
                    "name": m.get("name", dir_name),
                    "version": m.get("version", "?"),
                    "description": m.get("description", ""),
                    "loaded": False,
                    "num_commands": 0,
                    "path": os.path.join(self.plugin_dir, dir_name),
                })
        return result

    def get_command(self, key: str) -> Optional[Callable]:
        """Get a plugin command by key."""
        return self.commands.get(key)

    def run_command(self, key: str, **kwargs) -> Any:
        """Run a plugin command by key."""
        func = self.commands.get(key)
        if func:
            return func(**kwargs)
        print(f"  Unknown plugin command: {key}")
        return None


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Get or create the global plugin manager."""
    global _manager
    if _manager is None:
        _manager = PluginManager()
    return _manager

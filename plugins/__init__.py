"""Plugin system for AI Model Suite — auto-discovers and loads plugins from plugins/ directory."""

from plugins.loader import PluginManager, HookSystem, get_plugin_manager

__all__ = ['PluginManager', 'HookSystem', 'get_plugin_manager']

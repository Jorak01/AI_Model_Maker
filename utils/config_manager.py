"""Configuration Manager — Profiles, environment overrides, validation.

Features:
  - Named config profiles (fast, quality, gpu, etc.)
  - Environment variable overrides (AI_MODEL_DEVICE, etc.)
  - JSON schema validation for config.yaml
  - Config diffing between profiles
"""

import os
import copy
import yaml
from typing import Dict, Any, Optional, List, Tuple


# ---------------------------------------------------------------------------
# Config Schema for Validation
# ---------------------------------------------------------------------------

CONFIG_SCHEMA = {
    "model": {
        "required": ["base_model", "vocab_size", "embedding_dim", "hidden_dim",
                      "num_layers", "num_heads", "max_seq_length", "dropout"],
        "types": {
            "base_model": str, "vocab_size": int, "embedding_dim": int,
            "hidden_dim": int, "num_layers": int, "num_heads": int,
            "max_seq_length": int, "dropout": (int, float),
        },
        "ranges": {
            "vocab_size": (1, 1000000), "embedding_dim": (8, 16384),
            "hidden_dim": (8, 65536), "num_layers": (1, 128),
            "num_heads": (1, 256), "max_seq_length": (8, 131072),
            "dropout": (0.0, 1.0),
        },
    },
    "training": {
        "required": ["pipeline", "batch_size", "learning_rate", "num_epochs"],
        "types": {
            "pipeline": str, "batch_size": int, "learning_rate": float,
            "num_epochs": int, "gradient_clip": (int, float),
            "warmup_steps": int, "weight_decay": float,
        },
        "valid_values": {"pipeline": ["scratch", "finetune", "freeze"]},
    },
    "data": {"required": ["train_path", "test_path"]},
    "checkpoint": {"required": ["save_dir", "save_every", "keep_last"]},
    "generation": {
        "required": ["max_length", "temperature", "top_k", "top_p"],
        "ranges": {
            "temperature": (0.01, 5.0), "top_k": (0, 10000),
            "top_p": (0.0, 1.0), "repetition_penalty": (1.0, 10.0),
        },
    },
    "device": {"required": ["use_cuda", "cuda_device"]},
    "api": {"required": ["host", "port"]},
}

# Environment variable mapping
# These map env vars → (config_section, config_key, converter)
ENV_OVERRIDES = {
    # Device
    "AI_MODEL_DEVICE": ("device", "use_cuda", lambda v: v.lower() != "cpu"),
    "AI_MODEL_CUDA_DEVICE": ("device", "cuda_device", int),
    # Model / Training
    "AI_MODEL_BASE_MODEL": ("model", "base_model", str),
    "AI_MODEL_BASE": ("model", "base_model", str),
    "AI_MODEL_PIPELINE": ("training", "pipeline", str),
    "AI_MODEL_EPOCHS": ("training", "num_epochs", int),
    "AI_MODEL_BATCH_SIZE": ("training", "batch_size", int),
    "AI_MODEL_BATCH": ("training", "batch_size", int),
    "AI_MODEL_LR": ("training", "learning_rate", float),
    # Generation
    "AI_MODEL_MAX_LENGTH": ("generation", "max_length", int),
    "AI_MODEL_TEMPERATURE": ("generation", "temperature", float),
    # API server
    "AI_MODEL_API_PORT": ("api", "port", int),
    "AI_MODEL_API_HOST": ("api", "host", str),
}

# Env vars for external API keys, base URLs, and default models.
# These are applied separately onto the external_api section.
ENV_EXTERNAL_API = {
    "openai": {
        "api_key": "OPENAI_API_KEY",
        "base_url": "OPENAI_BASE_URL",
        "default_model": "OPENAI_DEFAULT_MODEL",
    },
    "anthropic": {
        "api_key": "ANTHROPIC_API_KEY",
        "base_url": "ANTHROPIC_BASE_URL",
        "default_model": "ANTHROPIC_DEFAULT_MODEL",
    },
    "google": {
        "api_key": "GOOGLE_API_KEY",
        "base_url": "GOOGLE_BASE_URL",
        "default_model": "GOOGLE_DEFAULT_MODEL",
    },
    "deepseek": {
        "api_key": "DEEPSEEK_API_KEY",
        "base_url": "DEEPSEEK_BASE_URL",
        "default_model": "DEEPSEEK_DEFAULT_MODEL",
    },
    "ollama": {
        "base_url": "OLLAMA_BASE_URL",
        "default_model": "OLLAMA_DEFAULT_MODEL",
    },
    "custom": {
        "api_key": "CUSTOM_API_KEY",
        "base_url": "CUSTOM_BASE_URL",
        "default_model": "CUSTOM_DEFAULT_MODEL",
    },
}

# Env vars for service ports (used by Docker and direct launch)
ENV_SERVICE_PORTS = {
    "AI_MODEL_COMPAT_PORT": ("compat_api_port", int, 8001),
    "AI_MODEL_WEBUI_PORT": ("webui_port", int, 7860),
    "AI_MODEL_DASHBOARD_PORT": ("dashboard_port", int, 5555),
}


# ---------------------------------------------------------------------------
# Config Loading & Profiles
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load config from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: str = "config.yaml"):
    """Save config to YAML file."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def list_profiles(config_dir: str = ".") -> List[str]:
    """List available config profiles."""
    profiles = []
    for f in os.listdir(config_dir):
        if f.startswith("config.") and f.endswith(".yaml") and f != "config.yaml":
            name = f[7:-5]  # config.NAME.yaml -> NAME
            profiles.append(name)
    return sorted(profiles)


def load_profile(name: str, config_dir: str = ".") -> Optional[Dict[str, Any]]:
    """Load a named config profile."""
    path = os.path.join(config_dir, f"config.{name}.yaml")
    if not os.path.exists(path):
        print(f"  Profile '{name}' not found at {path}")
        return None
    return load_config(path)


def save_profile(name: str, config: Dict[str, Any], config_dir: str = "."):
    """Save a named config profile."""
    path = os.path.join(config_dir, f"config.{name}.yaml")
    save_config(config, path)
    print(f"  ✓ Profile '{name}' saved to {path}")


def create_default_profiles(config_dir: str = "."):
    """Create standard config profiles if they don't exist."""
    base = load_config(os.path.join(config_dir, "config.yaml"))

    profiles = {
        "fast": {
            "description": "Fast training — fewer epochs, smaller batches",
            "overrides": {
                "training": {"num_epochs": 3, "batch_size": 16},
                "model": {"num_layers": 2, "num_heads": 2, "embedding_dim": 128, "hidden_dim": 256},
            }
        },
        "quality": {
            "description": "Quality training — more epochs, larger model",
            "overrides": {
                "training": {"num_epochs": 50, "batch_size": 8, "learning_rate": 5e-5},
                "model": {"num_layers": 6, "num_heads": 8, "embedding_dim": 512, "hidden_dim": 1024},
            }
        },
        "gpu": {
            "description": "GPU-optimized — CUDA enabled, larger batches",
            "overrides": {
                "device": {"use_cuda": True, "cuda_device": 0},
                "training": {"batch_size": 32},
            }
        },
    }

    created = 0
    for name, profile_info in profiles.items():
        path = os.path.join(config_dir, f"config.{name}.yaml")
        if not os.path.exists(path):
            cfg = copy.deepcopy(base)
            for section, overrides in profile_info["overrides"].items():
                if section in cfg:
                    cfg[section].update(overrides)
            cfg["_profile"] = {"name": name, "description": profile_info["description"]}
            save_config(cfg, path)
            created += 1

    return created


# ---------------------------------------------------------------------------
# Environment Variable Overrides
# ---------------------------------------------------------------------------

def apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ALL environment variable overrides to config.

    Handles:
      - Core config overrides (device, training, generation, API server)
      - External API keys, base URLs, and default models
      - Service port overrides
    """
    config = copy.deepcopy(config)
    applied = []

    # 1. Core config overrides
    for env_var, (section, key, converter) in ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is not None:
            try:
                converted = converter(value)
                if section in config:
                    config[section][key] = converted
                    applied.append(f"{env_var} → {section}.{key}")
            except (ValueError, TypeError) as e:
                print(f"  ⚠ Invalid {env_var}={value}: {e}")

    # 2. External API keys, base URLs, default models
    ext = config.get("external_api", {})
    for provider, mapping in ENV_EXTERNAL_API.items():
        if provider not in ext:
            ext[provider] = {}
        for cfg_key, env_var in mapping.items():
            value = os.environ.get(env_var)
            if value is not None and value != "":
                ext[provider][cfg_key] = value
                applied.append(f"{env_var} → external_api.{provider}.{cfg_key}")
    config["external_api"] = ext

    # 3. Service port overrides (stored in a flat 'ports' sub-dict)
    for env_var, (port_key, converter, default) in ENV_SERVICE_PORTS.items():
        value = os.environ.get(env_var)
        if value is not None:
            try:
                converted = converter(value)
                config.setdefault("ports", {})[port_key] = converted
                applied.append(f"{env_var} → ports.{port_key}")
            except (ValueError, TypeError) as e:
                print(f"  ⚠ Invalid {env_var}={value}: {e}")

    if applied:
        print(f"  Environment overrides applied:")
        for a in applied:
            print(f"    {a}")

    return config


def get_service_port(service: str, config: Optional[Dict] = None) -> int:
    """Get the port for a service, checking env vars first, then config.

    Args:
        service: One of 'api', 'compat_api', 'webui', 'dashboard'
        config: Optional config dict (loads from file if None)

    Returns:
        Port number
    """
    env_map = {
        "api": "AI_MODEL_API_PORT",
        "compat_api": "AI_MODEL_COMPAT_PORT",
        "webui": "AI_MODEL_WEBUI_PORT",
        "dashboard": "AI_MODEL_DASHBOARD_PORT",
    }
    defaults = {
        "api": 8000,
        "compat_api": 8001,
        "webui": 7860,
        "dashboard": 5555,
    }
    env_var = env_map.get(service)
    if env_var:
        val = os.environ.get(env_var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    if config:
        port_val = config.get("ports", {}).get(f"{service}_port")
        if port_val is not None:
            return int(port_val)
        if service == "api":
            return int(config.get("api", {}).get("port", defaults["api"]))
    return defaults.get(service, 8000)


# ---------------------------------------------------------------------------
# Config Validation
# ---------------------------------------------------------------------------

def validate_config(config: Dict[str, Any], verbose: bool = True) -> List[str]:
    """Validate config against schema. Returns list of errors."""
    errors: List[str] = []

    for section_name, rules in CONFIG_SCHEMA.items():
        if section_name not in config:
            errors.append(f"Missing section: {section_name}")
            continue

        section = config[section_name]

        # Check required keys
        for key in rules.get("required", []):
            if key not in section:
                errors.append(f"Missing key: {section_name}.{key}")

        # Check types
        for key, expected_type in rules.get("types", {}).items():
            if key in section:
                if not isinstance(section[key], expected_type):
                    errors.append(
                        f"Type error: {section_name}.{key} should be "
                        f"{expected_type}, got {type(section[key])}"
                    )

        # Check ranges
        for key, (lo, hi) in rules.get("ranges", {}).items():
            if key in section:
                val = section[key]
                if isinstance(val, (int, float)) and not (lo <= val <= hi):
                    errors.append(
                        f"Range error: {section_name}.{key}={val} "
                        f"(expected {lo}–{hi})"
                    )

        # Check valid values
        for key, valid in rules.get("valid_values", {}).items():
            if key in section and section[key] not in valid:
                errors.append(
                    f"Invalid value: {section_name}.{key}='{section[key]}' "
                    f"(expected one of {valid})"
                )

    # Check data files exist
    if "data" in config:
        for key in ("train_path", "test_path"):
            path = config["data"].get(key)
            if path and not os.path.exists(path):
                errors.append(f"File not found: data.{key} = '{path}'")

    if verbose:
        if errors:
            print(f"\n  ❌ Config validation: {len(errors)} error(s)")
            for e in errors:
                print(f"    • {e}")
        else:
            print(f"  ✓ Config validation passed")

    return errors


# ---------------------------------------------------------------------------
# Config Diff
# ---------------------------------------------------------------------------

def diff_configs(config_a: Dict, config_b: Dict,
                 name_a: str = "A", name_b: str = "B") -> List[str]:
    """Compare two configs, return list of differences."""
    diffs = []

    def _diff(a: Any, b: Any, path: str = ""):
        if isinstance(a, dict) and isinstance(b, dict):
            all_keys = set(list(a.keys()) + list(b.keys()))
            for key in sorted(all_keys):
                p = f"{path}.{key}" if path else key
                if key not in a:
                    diffs.append(f"  + {p}: {b[key]} (only in {name_b})")
                elif key not in b:
                    diffs.append(f"  - {p}: {a[key]} (only in {name_a})")
                else:
                    _diff(a[key], b[key], p)
        elif a != b:
            diffs.append(f"  ~ {path}: {a} → {b}")

    _diff(config_a, config_b)
    return diffs


# ---------------------------------------------------------------------------
# Interactive Config Manager
# ---------------------------------------------------------------------------

def interactive_config_manager():
    """Interactive configuration management."""
    print("\n" + "=" * 55)
    print("       Configuration Manager")
    print("=" * 55)

    while True:
        print("\n  Options:")
        print("  1  Validate current config")
        print("  2  List profiles")
        print("  3  Switch profile")
        print("  4  Create default profiles")
        print("  5  Show environment overrides")
        print("  6  Compare configs")
        print("  0  Back")

        try:
            choice = input("\n  config>> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if choice in ('0', 'back', 'quit', 'q'):
            break

        if choice == '1':
            config = load_config()
            validate_config(config)

        elif choice == '2':
            profiles = list_profiles()
            if profiles:
                print(f"\n  Available profiles:")
                for p in profiles:
                    print(f"    • {p} (config.{p}.yaml)")
            else:
                print("\n  No profiles found. Use option 4 to create defaults.")

        elif choice == '3':
            profiles = list_profiles()
            if not profiles:
                print("  No profiles available.")
                continue
            print(f"\n  Profiles: {', '.join(profiles)}")
            try:
                name = input("  Switch to: ").strip()
            except (KeyboardInterrupt, EOFError):
                continue
            profile = load_profile(name)
            if profile:
                save_config(profile, "config.yaml")
                print(f"  ✓ Switched to '{name}' profile")

        elif choice == '4':
            created = create_default_profiles()
            print(f"  ✓ Created {created} profile(s)")

        elif choice == '5':
            print(f"\n  Environment Variable Overrides:")
            print("  " + "-" * 50)
            for env_var, (section, key, _) in ENV_OVERRIDES.items():
                val = os.environ.get(env_var, "(not set)")
                print(f"  {env_var:<30} → {section}.{key:<15} = {val}")

        elif choice == '6':
            profiles = list_profiles()
            if not profiles:
                print("  No profiles to compare.")
                continue
            try:
                base = load_config()
                name = input(f"  Compare current with [{', '.join(profiles)}]: ").strip()
            except (KeyboardInterrupt, EOFError):
                continue
            other = load_profile(name)
            if other:
                diffs = diff_configs(base, other, "current", name)
                if diffs:
                    print(f"\n  Differences ({len(diffs)}):")
                    for d in diffs:
                        print(f"  {d}")
                else:
                    print("  Configs are identical.")

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).with_name("dashboard_config.json")


def load_dashboard_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load dashboard configuration from JSON.

    Returns a dictionary with at least `inputs` and `outputs` list keys.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)

    if not isinstance(data, dict):
        raise ValueError("Dashboard config root must be a JSON object")

    data.setdefault("inputs", [])
    data.setdefault("outputs", [])

    if not isinstance(data["inputs"], list) or not isinstance(data["outputs"], list):
        raise ValueError("Dashboard config fields 'inputs' and 'outputs' must be lists")

    return data


def save_dashboard_config(config: dict[str, Any], config_path: str | Path | None = None) -> Path:
    """Save dashboard configuration to JSON for use by GUI scripts or tooling."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, ensure_ascii=True)
        config_file.write("\n")
    return path


def load_smpc_config(config_path: str | Path | None = None):
    """Load SMPC configuration from the 'smpc' block in dashboard_config.json.

    Returns an SMPCConfig instance. Falls back to defaults when the block
    is absent.
    """
    from smpc_calculator import SMPCConfig
    raw = load_dashboard_config(config_path)
    smpc_block = raw.get("smpc", {})
    if smpc_block:
        return SMPCConfig.from_config_dict(smpc_block)
    return SMPCConfig()

from pathlib import Path
import json
from typing import Dict, Any


def run(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    root_str = state.get("artifacts_root") or configurable.get("output_root")
    root = Path(root_str) if root_str else Path.cwd()
    root.mkdir(parents=True, exist_ok=True)

    out = root / "inventory.json"
    sample = {"items": [], "note": "Phase0 stub inventory"}
    out.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    return {"inventory_path": str(out)}

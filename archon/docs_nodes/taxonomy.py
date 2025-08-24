import json
from pathlib import Path
from typing import Dict, Any


def run(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    # Determine artifacts root: prefer graph-provided artifacts_root, otherwise configurable.output_root, else fallback to CWD
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    root_str = state.get("artifacts_root") or configurable.get("output_root")
    root = Path(root_str) if root_str else Path.cwd()
    root.mkdir(parents=True, exist_ok=True)

    out = root / "taxonomy_map.json"
    sample = {"items": [], "note": "Phase0 stub taxonomy"}
    out.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    return {"taxonomy_map_path": str(out)}

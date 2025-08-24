import json
from pathlib import Path
from typing import Dict, Any


def run(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    root_str = state.get("artifacts_root") or configurable.get("output_root")
    root = Path(root_str) if root_str else Path.cwd()
    root.mkdir(parents=True, exist_ok=True)

    out = root / "rename_move_plan.json"
    sample = {"moves": [], "note": "Phase0 stub structure plan"}
    out.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    toc_dir = root / "toc_proposals"
    toc_dir.mkdir(parents=True, exist_ok=True)
    return {"move_plan_path": str(out), "toc_proposals_dir": str(toc_dir)}

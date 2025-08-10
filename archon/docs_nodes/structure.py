import json
from pathlib import Path
from typing import Dict, Any

ART_DIR = Path("generated/docs_reorg")


def run(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    out = ART_DIR / "rename_move_plan.json"
    sample = {"moves": [], "note": "Phase0 stub structure plan"}
    out.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    toc_dir = ART_DIR / "toc_proposals"
    toc_dir.mkdir(parents=True, exist_ok=True)
    return {"move_plan_path": str(out), "toc_proposals_dir": str(toc_dir)}

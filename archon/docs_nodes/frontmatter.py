import json
from pathlib import Path
from typing import Dict, Any

ART_DIR = Path("generated/docs_reorg")


def run(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    out = ART_DIR / "frontmatter_plan.json"
    sample = {"items": [], "note": "Phase0 stub frontmatter plan"}
    out.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    return {"frontmatter_plan_path": str(out)}

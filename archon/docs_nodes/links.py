from pathlib import Path
import json
from typing import Dict, Any

ART_DIR = Path("generated/docs_reorg")


def run(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    out = ART_DIR / "links_report.json"
    sample = {"broken": [], "maybe": [], "note": "Phase0 stub links report"}
    out.write_text(json.dumps(sample, indent=2), encoding="utf-8")
    return {"links_report_path": str(out)}

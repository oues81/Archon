from pathlib import Path
from typing import Dict, Any

ART_DIR = Path("generated/docs_reorg")


def assistant_brief(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    out = ART_DIR / "assistant_brief.md"
    out.write_text("# Assistant Brief\n\nPhase0 stub summary.\n", encoding="utf-8")
    return {"assistant_brief_path": str(out)}


def final_report(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    # For Phase 0, just echo a minimal summary in state
    return {"apply_summary": {"note": "Phase0 final report placeholder"}}

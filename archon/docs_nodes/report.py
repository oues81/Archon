from pathlib import Path
from typing import Dict, Any


def assistant_brief(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    root_str = state.get("artifacts_root") or configurable.get("output_root")
    root = Path(root_str) if root_str else Path.cwd()
    root.mkdir(parents=True, exist_ok=True)

    out = root / "assistant_brief.md"
    out.write_text("# Assistant Brief\n\nPhase0 stub summary.\n", encoding="utf-8")
    return {"assistant_brief_path": str(out)}


def final_report(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    # For Phase 0, just echo a minimal summary in state
    return {"apply_summary": {"note": "Phase0 final report placeholder"}}

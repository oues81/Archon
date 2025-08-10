from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

# Phase 2: record backup summary; keep frontmatter merge as no-op for now

def run(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    backups_root = Path(state.get("backups_root") or (Path("generated") / "backups"))
    backups_root.mkdir(parents=True, exist_ok=True)
    session = state.get("backup_session")
    summary_path = backups_root / "backup_summary.json"
    data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "session": session,
        "frontmatter_applied": False,
    }
    try:
        summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass
    return {"frontmatter_applied": False, "backup_summary": str(summary_path)}

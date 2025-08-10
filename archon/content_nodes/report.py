from pathlib import Path
from typing import Dict, Any, List
import json
import glob
import datetime

ART_DIR = Path("generated/restruct")


def assistant_brief(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    # Resolve artifacts root
    try:
        cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    except Exception:
        cfg = {}
    art_root = None
    if isinstance(state, dict):
        art_root = state.get("artifacts_root")
    if not art_root and isinstance(cfg, dict):
        art_root = cfg.get("output_root")
    root = Path(art_root) if isinstance(art_root, str) and art_root.strip() else ART_DIR
    root.mkdir(parents=True, exist_ok=True)
    out = root / "assistant_brief.md"
    # Load artifacts
    inv = {}
    ins = {}
    plan = {}
    journal = {}
    backup_summary = {}

    inv_path = state.get("inventory_path") if isinstance(state, dict) else None
    if not inv_path:
        p = root / "global_inventory.json"
        inv_path = str(p) if p.exists() else None
    if inv_path and Path(inv_path).exists():
        try:
            inv = json.loads(Path(inv_path).read_text(encoding="utf-8"))
        except Exception:
            inv = {}

    ins_path = state.get("insights_path") if isinstance(state, dict) else None
    if not ins_path:
        p = root / "content_insights_root.json"
        ins_path = str(p) if p.exists() else None
    if ins_path and Path(ins_path).exists():
        try:
            ins = json.loads(Path(ins_path).read_text(encoding="utf-8"))
        except Exception:
            ins = {}

    plan_path = state.get("move_plan_path") if isinstance(state, dict) else None
    if not plan_path:
        p = root / "rename_move_plan.json"
    # Analysis summary (new)
    analysis_path = state.get("analysis_summary_path") if isinstance(state, dict) else None
    if not analysis_path:
        p = root / "analysis_summary.md"
        analysis_path = str(p) if p.exists() else None
        plan_path = str(p) if p.exists() else None
    if plan_path and Path(plan_path).exists():
        try:
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        except Exception:
            plan = {}

    # Attempt to find the latest apply journal
    backups_root = Path(state.get("backups_root") or (Path("generated") / "backups"))
    try:
        sessions = sorted(Path(backups_root).glob("apply_*"))
        if sessions:
            latest = sessions[-1]
            jpath = latest / "apply_journal.json"
            if jpath.exists():
                journal = json.loads(jpath.read_text(encoding="utf-8"))
    except Exception:
        journal = {}

    # Backup summary
    try:
        bsum_path = backups_root / "backup_summary.json"
        if bsum_path.exists():
            backup_summary = json.loads(bsum_path.read_text(encoding="utf-8"))
    except Exception:
        backup_summary = {}

    # Compute quick stats
    inv_count = int(inv.get("count") or len(inv.get("items") or [])) if isinstance(inv, dict) else 0
    clusters = (ins.get("clusters") if isinstance(ins, dict) else None) or []
    unassigned = (ins.get("unassigned") if isinstance(ins, dict) else None) or []
    moves = (plan.get("moves") if isinstance(plan, dict) else None) or []
    applied = (journal.get("applied") if isinstance(journal, dict) else None) or []
    conflicts = (journal.get("conflicts") if isinstance(journal, dict) else None) or []
    skipped = (journal.get("skipped") if isinstance(journal, dict) else None) or []

    md = []
    md.append("# Assistant Brief (Content Restructurer)\n")
    md.append(f"Generated: {datetime.datetime.utcnow().isoformat()}Z\n")
    md.append("\n## Summary\n")
    md.append(f"- Inventory items: {inv_count}\n")
    md.append(f"- Clusters: {len(clusters)} | Unassigned: {len(unassigned)}\n")
    md.append(f"- Planned moves: {len(moves)}\n")
    if analysis_path:
        md.append(f"- Analysis summary: `{analysis_path}`\n")
    if applied or conflicts or skipped:
        md.append("- Apply results: ")
        md.append(f"applied={len(applied)} conflicts={len(conflicts)} skipped={len(skipped)}\n")
    else:
        md.append("- Apply results: none (not executed yet)\n")

    md.append("\n## Safety & Gating\n")
    md.append("- Part 1 is non-destructive (generated/ only).\n")
    md.append("- Part 2 requires approval token and never overwrites or deletes.\n")
    md.append("- Backups are stored under generated/backups/.\n")

    md.append("\n## Next actions\n")
    md.append("- Review clusters and the move plan.\n")
    md.append("- Run a dry-run validation to confirm conflicts.\n")
    md.append("- If acceptable, proceed with approval token to apply.\n")

    # Links to artifacts
    md.append("\n## Artifacts\n")
    if inv_path:
        md.append(f"- Inventory: `{inv_path}`\n")
    if ins_path:
        md.append(f"- Insights: `{ins_path}`\n")
    if plan_path:
        md.append(f"- Move Plan: `{plan_path}`\n")
    if analysis_path:
        md.append(f"- Analysis Summary: `{analysis_path}`\n")
    if journal:
        md.append(f"- Apply Journal: `{(backups_root / 'apply_*').as_posix()}` (latest used)\n")
    if backup_summary:
        md.append(f"- Backup Summary: `{(backups_root / 'backup_summary.json').as_posix()}`\n")

    out.write_text("\n".join(md), encoding="utf-8")
    return {"assistant_brief_path": str(out)}


def final_report(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    # Resolve artifacts root
    try:
        cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    except Exception:
        cfg = {}
    art_root = None
    if isinstance(state, dict):
        art_root = state.get("artifacts_root")
    if not art_root and isinstance(cfg, dict):
        art_root = cfg.get("output_root")
    root = Path(art_root) if isinstance(art_root, str) and art_root.strip() else ART_DIR
    root.mkdir(parents=True, exist_ok=True)
    out = root / "final_report.json"
    data: Dict[str, Any] = {}
    # Collect same info as in brief
    inv_path = state.get("inventory_path") if isinstance(state, dict) else None
    if not inv_path:
        p = root / "global_inventory.json"
        inv_path = str(p) if p.exists() else None
    ins_path = state.get("insights_path") if isinstance(state, dict) else None
    if not ins_path:
        p = root / "content_insights_root.json"
        ins_path = str(p) if p.exists() else None
    plan_path = state.get("move_plan_path") if isinstance(state, dict) else None
    if not plan_path:
        p = root / "rename_move_plan.json"
        plan_path = str(p) if p.exists() else None
    backups_root = Path(state.get("backups_root") or (Path("generated") / "backups"))
    latest_session = None
    try:
        sessions = sorted(Path(backups_root).glob("apply_*"))
        if sessions:
            latest_session = str(sessions[-1])
    except Exception:
        latest_session = None

    data["artifacts"] = {
        "inventory": inv_path,
        "insights": ins_path,
        "move_plan": plan_path,
        "backups_root": str(backups_root),
        "latest_apply_session": latest_session,
    }

    # Try to load counts for convenience
    try:
        inv = json.loads(Path(inv_path).read_text(encoding="utf-8")) if inv_path and Path(inv_path).exists() else {}
        ins = json.loads(Path(ins_path).read_text(encoding="utf-8")) if ins_path and Path(ins_path).exists() else {}
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8")) if plan_path and Path(plan_path).exists() else {}
        data["stats"] = {
            "inventory_count": int(inv.get("count") or len(inv.get("items") or [])) if isinstance(inv, dict) else 0,
            "clusters": len((ins.get("clusters") if isinstance(ins, dict) else []) or []),
            "unassigned": len((ins.get("unassigned") if isinstance(ins, dict) else []) or []),
            "planned_moves": len((plan.get("moves") if isinstance(plan, dict) else []) or []),
        }
    except Exception:
        data["stats"] = {}

    # Include timings if present in state
    try:
        timings = state.get("timings") if isinstance(state, dict) else None
        if isinstance(timings, dict):
            order = list(timings.get("_order") or [])
            total_ms = 0
            per_nodes = []
            for name in order:
                info = timings.get(name) or {}
                dur = int(info.get("duration_ms") or 0)
                total_ms += dur
                per_nodes.append({"node": name, "duration_ms": dur})
            data["timings"] = timings
            data["timings_summary"] = {"total_ms": total_ms, "order": order, "per_node": per_nodes}
    except Exception:
        pass

    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"final_report_path": str(out), "final_report": data}

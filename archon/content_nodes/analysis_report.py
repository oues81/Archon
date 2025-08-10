from pathlib import Path
from typing import Dict, Any
import json
import time
import logging

ART_DIR = Path("generated/restruct")
logger = logging.getLogger(__name__)


def _resolve_root(config: Dict[str, Any]) -> Path:
    cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    out = None
    if isinstance(cfg, dict):
        out = cfg.get("output_root")
    if isinstance(out, str) and out.strip():
        p = Path(out)
        p.mkdir(parents=True, exist_ok=True)
        return p
    ART_DIR.mkdir(parents=True, exist_ok=True)
    return ART_DIR


def run(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a compact analysis summary from prior artifacts.
    - Reads inventory, insights, and move plan
    - Writes analysis_summary.md under output_root (or default ART_DIR)
    """
    root = _resolve_root(config)
    out = root / "analysis_summary.md"

    # Resolve artifact paths from state or defaults under root
    def _json_load(p: Path) -> Dict[str, Any]:
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        except Exception:
            return {}

    inv_p = Path(state.get("inventory_path")) if isinstance(state, dict) and state.get("inventory_path") else (root / "global_inventory.json")
    ins_p = Path(state.get("insights_path")) if isinstance(state, dict) and state.get("insights_path") else (root / "content_insights_root.json")
    plan_p = Path(state.get("move_plan_path")) if isinstance(state, dict) and state.get("move_plan_path") else (root / "rename_move_plan.json")

    # Optionally ingest previous artifacts (e.g., from legacy workflows)
    cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    workspace_root = None
    try:
        if isinstance(cfg, dict):
            wr = cfg.get("workspace_root")
            if isinstance(wr, str) and wr.strip():
                workspace_root = Path(wr)
    except Exception:
        workspace_root = None
    prev_inv_p = prev_ins_p = prev_plan_p = None
    try:
        base_prev = (workspace_root / "generated/reorg") if workspace_root else Path("generated/reorg")
        if base_prev.exists():
            cand_inv = base_prev / "global_inventory.json"
            cand_ins = base_prev / "content_insights_root.json"
            cand_plan = base_prev / "rename_move_plan.json"
            prev_inv_p = cand_inv if cand_inv.exists() else None
            prev_ins_p = cand_ins if cand_ins.exists() else None
            prev_plan_p = cand_plan if cand_plan.exists() else None
    except Exception:
        pass

    # Optionally ingest previous artifacts (e.g., from legacy workflows)
    cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    workspace_root = None
    try:
        if isinstance(cfg, dict):
            wr = cfg.get("workspace_root")
            if isinstance(wr, str) and wr.strip():
                workspace_root = Path(wr)
    except Exception:
        workspace_root = None
    prev_inv_p = prev_ins_p = prev_plan_p = None
    try:
        base_prev = (workspace_root / "generated/reorg") if workspace_root else Path("generated/reorg")
        if base_prev.exists():
            cand_inv = base_prev / "global_inventory.json"
            cand_ins = base_prev / "content_insights_root.json"
            cand_plan = base_prev / "rename_move_plan.json"
            prev_inv_p = cand_inv if cand_inv.exists() else None
            prev_ins_p = cand_ins if cand_ins.exists() else None
            prev_plan_p = cand_plan if cand_plan.exists() else None
    except Exception:
        pass

    inv = _json_load(inv_p)
    ins = _json_load(ins_p)
    plan = _json_load(plan_p)
    prev_inv = _json_load(prev_inv_p) if prev_inv_p else {}
    prev_ins = _json_load(prev_ins_p) if prev_ins_p else {}
    prev_plan = _json_load(prev_plan_p) if prev_plan_p else {}

    inv_count = int(inv.get("count") or len(inv.get("items") or [])) if isinstance(inv, dict) else 0
    prev_inv_count = int(prev_inv.get("count") or len(prev_inv.get("items") or [])) if isinstance(prev_inv, dict) else None
    clusters = (ins.get("clusters") if isinstance(ins, dict) else None) or []
    unassigned = (ins.get("unassigned") if isinstance(ins, dict) else None) or []
    moves = (plan.get("moves") if isinstance(plan, dict) else None) or []
    prev_clusters = (prev_ins.get("clusters") if isinstance(prev_ins, dict) else None) or []
    prev_unassigned = (prev_ins.get("unassigned") if isinstance(prev_ins, dict) else None) or []
    prev_moves = (prev_plan.get("moves") if isinstance(prev_plan, dict) else None) or []

    # Heuristics for quick insights
    hotspots = []
    try:
        # Large clusters and folders with many files
        large_clusters = sorted(clusters, key=lambda c: len(c.get("members") or []), reverse=True)[:5]
        for c in large_clusters:
            hotspots.append(f"Cluster '{c.get('label','?')}' — members={len(c.get('members') or [])} — risk=max")
    except Exception:
        pass

    risk_summary = {
        "total_moves": len(moves),
        "potential_conflicts": sum(1 for m in moves if isinstance(m, dict) and m.get("conflict_note")),
        "high_risk_moves": sum(1 for m in moves if isinstance(m, dict) and (m.get("risk") or "").lower() in {"high", "very high"}),
    }

    md = []
    md.append("# Analysis Summary")
    md.append(f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    md.append("")
    # Delta vs previous (if any)
    if prev_inv or prev_ins or prev_plan:
        md.append("## Delta vs Previous Artifacts")
        if prev_inv_count is not None:
            md.append(f"- Inventory: {prev_inv_count} → {inv_count} (Δ {inv_count - prev_inv_count:+})")
        md.append(f"- Clusters: {len(prev_clusters)} → {len(clusters)} (Δ {len(clusters) - len(prev_clusters):+})")
        md.append(f"- Unassigned: {len(prev_unassigned)} → {len(unassigned)} (Δ {len(unassigned) - len(prev_unassigned):+})")
        md.append(f"- Planned moves: {len(prev_moves)} → {len(moves)} (Δ {len(moves) - len(prev_moves):+})")
        md.append("")
    md.append("## Executive Summary")
    md.append(f"- Inventory items: {inv_count}")
    md.append(f"- Clusters: {len(clusters)} | Unassigned: {len(unassigned)}")
    md.append(f"- Planned moves: {len(moves)} (conflicts≈{risk_summary['potential_conflicts']}, high‑risk≈{risk_summary['high_risk_moves']})")
    if hotspots:
        md.append("- Hotspots:")
        for h in hotspots:
            md.append(f"  - {h}")
    else:
        md.append("- Hotspots: none detected")

    md.append("")
    # Delta vs previous (if any)
    if prev_inv or prev_ins or prev_plan:
        md.append("## Delta vs Previous Artifacts")
        if prev_inv_count is not None:
            md.append(f"- Inventory: {prev_inv_count} → {inv_count} (Δ {inv_count - prev_inv_count:+})")
        md.append(f"- Clusters: {len(prev_clusters)} → {len(clusters)} (Δ {len(clusters) - len(prev_clusters):+})")
        md.append(f"- Unassigned: {len(prev_unassigned)} → {len(unassigned)} (Δ {len(unassigned) - len(prev_unassigned):+})")
        md.append(f"- Planned moves: {len(prev_moves)} → {len(moves)} (Δ {len(moves) - len(prev_moves):+})")
        md.append("")
    md.append("## Suggested Next Steps")
    md.append("- Review top clusters and ensure labeling matches documentation taxonomy")
    md.append("- Inspect conflicts and adjust targets before approval")
    md.append("- Validate unassigned files; reclassify or exclude if needed")

    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("✅ [analysis_report.done] module=archon.archon.content_nodes.analysis_report:run | out=%s", str(out))
    return {"analysis_summary_path": str(out)}

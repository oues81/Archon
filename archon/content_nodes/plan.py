from pathlib import Path
import json
from typing import Dict, Any, List
from k.llm import get_llm_provider
import time
import logging

ART_DIR = Path("generated/restruct")
logger = logging.getLogger(__name__)


def run(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    # Resolve artifacts root (timestamped run directory if provided)
    try:
        cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    except Exception:
        cfg = {}
    root = None
    if isinstance(state, dict):
        root = state.get("artifacts_root") or None
    if not root and isinstance(cfg, dict):
        root = cfg.get("output_root")
    art_root = Path(root) if isinstance(root, str) and root.strip() else ART_DIR
    art_root.mkdir(parents=True, exist_ok=True)
    out = art_root / "rename_move_plan.json"

    # Load inventory and insights if available
    inv_path = None
    if isinstance(state, dict):
        inv_path = state.get("inventory_path")
    if not inv_path:
        cand = art_root / "global_inventory.json"
        inv_path = str(cand) if cand.exists() else None

    insights_path = None
    if isinstance(state, dict):
        insights_path = state.get("insights_path")
    if not insights_path:
        cand2 = art_root / "content_insights_root.json"
        insights_path = str(cand2) if cand2.exists() else None

    inventory: Dict[str, Any] = {"items": []}
    insights: Dict[str, Any] = {"clusters": [], "unassigned": []}
    try:
        if inv_path and Path(inv_path).exists():
            inventory = json.loads(Path(inv_path).read_text(encoding="utf-8"))
    except Exception:
        inventory = {"items": []}
    try:
        if insights_path and Path(insights_path).exists():
            insights = json.loads(Path(insights_path).read_text(encoding="utf-8"))
    except Exception:
        insights = {"clusters": [], "unassigned": []}

    items: List[Dict[str, Any]] = inventory.get("items") or []
    clusters: List[Dict[str, Any]] = insights.get("clusters") or []

    # Trim to keep prompt size reasonable
    items_sample = items[:600]
    clusters_sample = []
    for c in clusters[:40]:
        cs = dict(c)
        if isinstance(cs.get("members"), list):
            cs["members"] = cs["members"][:200]
        clusters_sample.append(cs)

    provider = get_llm_provider()
    model = getattr(provider.config, "reasoner_model", None) or provider.config.primary_model
    # Apply TIMEOUT_S from config if provided
    try:
        cfg_llm = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
        if isinstance(cfg_llm, dict):
            llm_conf = cfg_llm.get("llm_config") or {}
            t_s = llm_conf.get("TIMEOUT_S") if isinstance(llm_conf, dict) else None
            if isinstance(t_s, (int, float)) and hasattr(provider, "config") and hasattr(provider.config, "timeout"):
                provider.config.timeout = int(t_s)
    except Exception:
        pass

    logger.info(
        "🧭 [plan.start] module=archon.archon.content_nodes.plan:run_plan | items=%s | items_sample=%s | clusters=%s | clusters_sample=%s | model=%s",
        len(items), len(items_sample), len(clusters), len(clusters_sample), model
    )
    start_ts = time.time()

    system = (
        "You are a repository restructuring planner.\n"
        "Propose a SAFE rename/move plan. STRICT RULES:\n"
        "- No deletions.\n"
        "- No overwrites: if target exists, mark conflict_note.\n"
        "- Keep apply=false for every move (Part 1 only).\n"
        "Return STRICT JSON: {\n"
        "  \"moves\": [ { \"source\": str, \"target\": str, \"apply\": false, \"reason\": str, \"risk\": str, \"conflict_note\": str|null } ],\n"
        "  \"notes\": str\n"
        "}. No extra text outside JSON."
    )

    user = {
        "task": "Generate a non-destructive rename/move plan",
        "context": {
            "inventory_sample": items_sample,
            "clusters_sample": clusters_sample
        },
        "constraints": [
            "No deletion entries",
            "Mark conflicts instead of overwriting",
            "Use existing repository paths; do not invent non-sensical folders"
        ]
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)},
    ]

    plan = None
    try:
        txt = provider.generate(messages, model=model, extra={"response_format": {"type": "json_object"}})
        plan = json.loads(txt)
    except Exception as e:
        logger.warning("⚠️  [plan.error] LLM or parse failed: %s", str(e))
        plan = {"moves": [], "notes": "fallback"}

    # Normalize and enforce safety flags
    moves = plan.get("moves") or []
    safe_moves = []
    for m in moves:
        src = (m or {}).get("source")
        tgt = (m or {}).get("target")
        reason = (m or {}).get("reason") or ""
        risk = (m or {}).get("risk") or "low"
        conflict = (m or {}).get("conflict_note")
        if isinstance(src, str) and isinstance(tgt, str) and src.strip() and tgt.strip():
            safe_moves.append({
                "source": src.strip(),
                "target": tgt.strip(),
                "apply": False,
                "reason": reason,
                "risk": risk,
                "conflict_note": conflict if isinstance(conflict, str) else None
            })

    # Optional deterministic conflict-rename pass (deduplicate targets only within plan)
    auto_renamed = 0
    try:
        use_conflict_rename = bool((cfg or {}).get("use_conflict_rename"))
    except Exception:
        use_conflict_rename = False
    if use_conflict_rename and safe_moves:
        seen: Dict[str, int] = {}
        adjusted: List[Dict[str, Any]] = []
        for m in safe_moves:
            src = m["source"]
            tgt = m["target"]
            base = tgt
            if tgt not in seen:
                seen[tgt] = 1
                adjusted.append(m)
                continue
            # Collision: generate deterministic suffixed target
            parent = str(Path(tgt).parent)
            stem = Path(tgt).stem
            suffix = Path(tgt).suffix
            # first conflict: append __from_<src_parent>
            src_parent = Path(src).parent.name or "src"
            candidate = str(Path(parent) / f"{stem}__from_{src_parent}{suffix}") if parent and parent != "." else f"{stem}__from_{src_parent}{suffix}"
            k = 1
            while candidate in seen:
                k += 1
                candidate = str(Path(parent) / f"{stem}__from_{src_parent}__{k}{suffix}") if parent and parent != "." else f"{stem}__from_{src_parent}__{k}{suffix}"
            new_m = dict(m)
            new_m["original_target"] = tgt
            new_m["target"] = candidate
            auto_renamed += 1
            seen[candidate] = 1
            adjusted.append(new_m)
        safe_moves = adjusted

    out_data = {
        "note": "Rename/Move plan (Part 1, non-destructive)",
        "moves": safe_moves,
        "model": model,
        "source_inventory": inv_path,
        "source_insights": insights_path,
        "auto_renamed_count": auto_renamed,
    }
    out.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    dur_ms = int((time.time() - start_ts) * 1000)
    logger.info(
        "✅ [plan.done] module=archon.archon.content_nodes.plan:run_plan | moves=%s | dur_ms=%s | out=%s",
        len(safe_moves), dur_ms, str(out)
    )
    return {"move_plan_path": str(out), "move_count": len(safe_moves)}

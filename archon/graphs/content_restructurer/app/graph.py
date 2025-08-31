# -*- coding: utf-8 -*-
from typing import Dict, Any
from pathlib import Path
import time
import os
import logging
import logfire
from archon.utils.utils import configure_logging

try:
    from langgraph.graph import StateGraph, END
except Exception:  # pragma: no cover - fallback placeholder
    StateGraph = None  # type: ignore
    END = None  # type: ignore

from archon.archon.restruct_common.ensure_dirs import ensure_content_dirs
from archon.archon.restruct_common import approval as approval_node
from archon.archon.restruct_common import git_apply as git_apply_node
from archon.archon.content_nodes import inventory as inv_node
from archon.archon.content_nodes import semantic as sem_node
from archon.archon.content_nodes import plan as plan_node
from archon.archon.content_nodes import report as report_node
from archon.archon.content_nodes import analysis_report as ar_node

_content_flow = None

# Initialize standardized logging and Logfire
_log_summary = configure_logging()
logger = logging.getLogger(__name__)
try:
    logfire.configure(service_name="archon")
    logger.info("✅ Logfire configured successfully")
except Exception as e:
    logger.warning(f"⚠️ Unable to configure Logfire: {e}")
    os.environ.setdefault("LOGFIRE_DISABLE", "1")


def _ensure_dirs(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer a caller-provided output_root (timestamped run directory). Fallback to default content roots.
    try:
        cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    except Exception:
        cfg = {}
    output_root = None
    if isinstance(cfg, dict):
        output_root = cfg.get("output_root")

    if isinstance(output_root, str) and output_root.strip():
        out = Path(output_root)
        out.mkdir(parents=True, exist_ok=True)
        corr = (state.get("correlation_id") if isinstance(state, dict) else None) or cfg.get("correlation_id") or "content-session"
        artifacts_root = os.path.join(str(out), str(corr))
        # Backups under generated/backups/restruct/<run_id>
        run_id = str(cfg.get("run_id") or out.name)
        bkp = Path("generated") / "backups" / "restruct" / run_id
        bkp.mkdir(parents=True, exist_ok=True)
        return {"artifacts_root": artifacts_root, "backups_root": str(bkp)}
    else:
        art, bkp = ensure_content_dirs()
        return {"artifacts_root": str(art), "backups_root": str(bkp)}


def _time_node(name: str, fn):
    def _wrapped(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        try:
            result = fn(state, config)
        finally:
            end = time.time()
        duration_ms = int((end - start) * 1000)
        timings = {}
        if isinstance(state, dict):
            timings = dict(state.get("timings") or {})
        order = list(timings.get("_order") or [])
        order.append(name)
        timings[name] = {"start": start, "end": end, "duration_ms": duration_ms}
        timings["_order"] = order
        # Merge with existing state to preserve previously computed keys
        base = dict(state) if isinstance(state, dict) else {}
        if isinstance(result, dict):
            base.update(result)
        base["timings"] = timings
        return base
    return _wrapped


def _assistant_brief(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    return report_node.assistant_brief(state, config)


def _final_report(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    return report_node.final_report(state, config)


def get_content_flow():
    global _content_flow
    if _content_flow is not None:
        return _content_flow

    if StateGraph is None:
        raise RuntimeError("LangGraph is not available in this environment")

    builder = StateGraph(dict)  # state is a plain dict

    # Nodes
    builder.add_node("ensure_dirs", _time_node("ensure_dirs", _ensure_dirs))
    builder.add_node("inventory", _time_node("inventory", inv_node.run))
    builder.add_node("semantic", _time_node("semantic", sem_node.run))
    builder.add_node("plan", _time_node("plan", plan_node.run))
    builder.add_node("analysis_report", _time_node("analysis_report", ar_node.run))
    builder.add_node("assistant_brief", _time_node("assistant_brief", _assistant_brief))
    builder.add_node("approval", _time_node("approval", approval_node.run))
    builder.add_node("apply_moves", _time_node("apply_moves", git_apply_node.run))
    builder.add_node("final_report", _time_node("final_report", _final_report))

    # Edges
    builder.set_entry_point("ensure_dirs")
    builder.add_edge("ensure_dirs", "inventory")
    builder.add_edge("inventory", "semantic")
    builder.add_edge("semantic", "plan")
    builder.add_edge("plan", "analysis_report")
    builder.add_edge("analysis_report", "assistant_brief")
    builder.add_edge("assistant_brief", "approval")
    builder.add_edge("approval", "apply_moves")
    builder.add_edge("apply_moves", "final_report")
    builder.add_edge("final_report", END)

    _content_flow = builder.compile()
    return _content_flow

# -*- coding: utf-8 -*-
from typing import Dict, Any
from pathlib import Path
import time
import os
import logging
import logfire
from k.core.utils.utils import configure_logging

try:
    from langgraph.graph import StateGraph, END
except Exception:  # pragma: no cover - fallback placeholder
    StateGraph = None  # type: ignore
    END = None  # type: ignore

from k.restruct_common.ensure_dirs import ensure_docs_dirs
from k.restruct_common import approval as approval_node
from k.restruct_common import git_apply as git_apply_node
from k.restruct_common import backups as backups_node
from k.docs_nodes import inventory as inv_node
from k.docs_nodes import links as links_node
from k.docs_nodes import taxonomy as taxo_node
from k.docs_nodes import frontmatter as fm_node
from k.docs_nodes import structure as struct_node
from k.docs_nodes import report as report_node

_docs_flow = None

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
    # Prefer a caller-provided output_root. Fallback to default docs roots.
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
        run_id = str(cfg.get("run_id") or out.name)
        bkp = Path("generated") / "backups" / "docs_reorg" / run_id
        bkp.mkdir(parents=True, exist_ok=True)
        return {"artifacts_root": str(out), "backups_root": str(bkp)}
    art, bkp = ensure_docs_dirs()
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
        delta = {"timings": timings}
        if isinstance(result, dict):
            delta.update(result)
        return delta
    return _wrapped


def _assistant_brief(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    return report_node.assistant_brief(state, config)


def _final_report(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    return report_node.final_report(state, config)


def get_docs_flow():
    global _docs_flow
    if _docs_flow is not None:
        return _docs_flow

    if StateGraph is None:
        raise RuntimeError("LangGraph is not available in this environment")

    builder = StateGraph(dict)  # state is a plain dict

    # Nodes
    builder.add_node("ensure_dirs", _time_node("ensure_dirs", _ensure_dirs))
    builder.add_node("inventory", _time_node("inventory", inv_node.run))
    builder.add_node("links", _time_node("links", links_node.run))
    builder.add_node("taxonomy", _time_node("taxonomy", taxo_node.run))
    builder.add_node("frontmatter", _time_node("frontmatter", fm_node.run))
    builder.add_node("structure", _time_node("structure", struct_node.run))
    builder.add_node("assistant_brief", _time_node("assistant_brief", _assistant_brief))
    builder.add_node("approval", _time_node("approval", approval_node.run))
    builder.add_node("apply_moves", _time_node("apply_moves", git_apply_node.run))
    builder.add_node("apply_backups_frontmatter", _time_node("apply_backups_frontmatter", backups_node.run))
    builder.add_node("final_report", _time_node("final_report", _final_report))

    # Edges (Phase 0: linear happy path)
    builder.set_entry_point("ensure_dirs")
    builder.add_edge("ensure_dirs", "inventory")
    builder.add_edge("inventory", "links")
    builder.add_edge("links", "taxonomy")
    builder.add_edge("taxonomy", "frontmatter")
    builder.add_edge("frontmatter", "structure")
    builder.add_edge("structure", "assistant_brief")
    builder.add_edge("assistant_brief", "approval")
    builder.add_edge("approval", "apply_moves")
    builder.add_edge("apply_moves", "apply_backups_frontmatter")
    builder.add_edge("apply_backups_frontmatter", "final_report")
    builder.add_edge("final_report", END)

    _docs_flow = builder.compile()
    return _docs_flow

"""DocsMaintainerGraph orchestrator.

Standardizes config consumption across docs nodes:
- Uses config["configurable"]["thread_id"], ["run_id"] if provided (no-op here but reserved for checkpointers/logging).
- Initializes state["artifacts_root"] from config["configurable"]["output_root"] + correlation_id.
- Passes config through to all nodes so they can use output_root/artifacts_root consistently.

This phase uses stub nodes that do not call LLMs yet; when LLMs are added,
ensure TIMEOUT_S is read from config["configurable"]["llm_config"]["TIMEOUT_S"].
"""
from __future__ import annotations

from typing import Any, Dict
import os

from .docs_nodes import inventory, structure, taxonomy, links, frontmatter, report


def _ensure_artifacts_root(state: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Ensure state['artifacts_root'] := <output_root>/<correlation_id> or fallback.

    - If config.configurable.output_root is provided, use that root combined with correlation_id.
    - Otherwise fallback to CWD/<correlation_id>.
    """
    cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
    if state.get("artifacts_root"):
        return
    corr = state.get("correlation_id") or "docs-session"
    out_root = cfg.get("output_root")
    if out_root:
        artifacts_root = os.path.join(str(out_root), str(corr))
    else:
        artifacts_root = os.path.join(os.getcwd(), str(corr))
    os.makedirs(artifacts_root, exist_ok=True)
    state["artifacts_root"] = artifacts_root


def run_docs_flow(initial_state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the DocsMaintainer flow sequentially.

    Order:
      inventory -> structure -> taxonomy -> links -> frontmatter -> report.assistant_brief -> report.final_report
    """
    state = dict(initial_state)
    _ensure_artifacts_root(state, config)

    # Process nodes sequentially, merging state
    for node in (
        inventory.run,
        structure.run,
        taxonomy.run,
        links.run,
        frontmatter.run,
        report.assistant_brief,
        report.final_report,
    ):
        updates = node(state, config)
        if isinstance(updates, dict):
            state.update(updates)

    return state

# -*- coding: utf-8 -*-
"""
Runner: Execute CV Graph based on shared Archon State Markdown file.

Usage:
  python -m archon.archon.run_cv_from_archon_state --path /mnt/c/projects/rh_cv_vector/docs/archon_state.md

Environment variables (optional):
  ARCHON_STATE_PATH: path to the shared Markdown state file.

Behavior:
  - Reads the last fenced JSON block from the Markdown file.
  - Builds initial CVState and runs the CV agentic flow.
  - Writes outputs (scores, recommendation, merged_for_sharepoint, artifacts_root)
    back into the same fenced JSON block.
  - Appends a timestamped message.

Notes:
  - This runner is conservative and does not attempt to parse binary CVs. Prefer
    providing `cv_path` (server-side accessible path) for better results.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from archon.archon.graphs.cv_graph.app.graph import build_graph
from archon.archon.archon_state_io import (
    append_message_in_file,
    load_state_from_file,
    update_outputs_in_file,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_initial_state(shared_raw: Dict[str, Any]) -> Dict[str, Any]:
    inputs = shared_raw.get("inputs") or {}
    profil_poste_json = inputs.get("profil_poste_json") or {}
    cv_path = inputs.get("cv_path") or ""
    file_b64 = inputs.get("file_base64") or ""
    correlation_id = inputs.get("correlation_id") or "cv-graph"
    perform_upsert = bool(inputs.get("perform_upsert") or False)

    cv_text = ""
    if cv_path and Path(cv_path).exists():
        try:
            # naive text read; upstream parsing should be preferred
            cv_text = Path(cv_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            cv_text = ""
    elif file_b64:
        try:
            raw = base64.b64decode(file_b64, validate=False)
            cv_text = raw.decode("utf-8", errors="ignore")
        except Exception:
            cv_text = ""

    initial: Dict[str, Any] = {
        "profil_poste_json": profil_poste_json,
        "cv_path": cv_path,
        "cv_text": cv_text,
        "correlation_id": correlation_id,
        "perform_upsert": perform_upsert,
        "messages": [],
    }
    return initial


async def _run_once(state_path: str) -> None:
    ss = load_state_from_file(state_path)
    if ss is None:
        raise FileNotFoundError(f"No JSON fence found in state file: {state_path}")

    # Prepare and run canonical cv_graph pipeline
    initial_state = _build_initial_state(ss.raw)
    graph = build_graph()
    # Use async invoke if available, else sync
    if hasattr(graph, "ainvoke"):
        result = await graph.ainvoke(initial_state)
    else:
        result = graph.invoke(initial_state)

    # Collect outputs
    outputs: Dict[str, Any] = {
        "score_skills": result.get("score_skills"),
        "score_experience": result.get("score_experience"),
        "score_education": result.get("score_education"),
        "score_langues": result.get("score_langues"),
        "score_localisation": result.get("score_localisation"),
        "score_global": result.get("score_global"),
        "recommandation": result.get("recommandation"),
        "match_commentaire": result.get("match_commentaire"),
        "merged_for_sharepoint": result.get("merged_for_sharepoint", {}),
        "artifacts_root": result.get("artifacts_root", ""),
    }

    # Write back
    update_outputs_in_file(state_path, outputs)
    append_message_in_file(
        state_path,
        author="archon",
        note="CV Graph executed; outputs updated.",
        iso_ts=_now_iso(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CV Graph from shared Archon State file")
    parser.add_argument("--path", default=os.environ.get("ARCHON_STATE_PATH", ""), help="Path to archon_state.md")
    args = parser.parse_args()

    if not args.path:
        raise SystemExit("--path not provided and ARCHON_STATE_PATH not set")

    asyncio.run(_run_once(args.path))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""CV Graph pipeline — canonical implementation under graphs/cv_graph.
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
import json
from datetime import datetime, timezone
import os
import logging
import logfire
from k.core.utils.utils import configure_logging

from k.graphs.cv_graph.agents import (
    preprocessing,
    contact_info,
    location,
    education,
    job,
    competence,
    availability,
)
from k.graphs.cv_graph.agents import fill_all
from k.graphs.cv_graph.schemas.candidate_schema import CANDIDATE_SCHEMA


_log_summary = configure_logging()
logger = logging.getLogger(__name__)
try:
    logfire.configure(service_name="archon")
    logger.info("✅ Logfire configured successfully")
except Exception as e:
    logger.warning(f"⚠️ Unable to configure Logfire: {e}")
    os.environ.setdefault("LOGFIRE_DISABLE", "1")


class _Pipeline:
    def __init__(self) -> None:
        pass

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.invoke(state)

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        corr = state.get("correlation_id", "cv-graph")
        out_dir = Path("out/tmp_ingest") / corr
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Preprocessing
        pre = preprocessing.run(state.get("cv_path"))
        (out_dir / "preprocessing.json").write_text(json.dumps(pre, ensure_ascii=False, indent=2), encoding="utf-8")

        # 2) Contact info (deterministic only for email/phone/linkedin + file meta)
        ci = contact_info.run(pre.get("normalized_text", ""), state.get("cv_path"))
        (out_dir / "contact_info.json").write_text(json.dumps(ci, ensure_ascii=False, indent=2), encoding="utf-8")

        # 3) LLM-backed agents (stubs for now) — no deterministic expansion
        txt = pre.get("normalized_text", "")
        loc = location.run(txt)
        edu = education.run(txt)
        jb = job.run(txt)
        comp = competence.run(txt)
        avail = availability.run(txt)
        (out_dir / "location.json").write_text(json.dumps(loc, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "education.json").write_text(json.dumps(edu, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "job.json").write_text(json.dumps(jb, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "competence.json").write_text(json.dumps(comp, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "availability.json").write_text(json.dumps(avail, ensure_ascii=False, indent=2), encoding="utf-8")

        # 4) Merge minimal and fill-all defaults
        merged: Dict[str, Any] = {}
        merged.update(ci)
        merged.update(loc)
        merged.update(edu)
        merged.update(jb)
        merged.update(comp)
        merged.update(avail)
        merged.update({
            # carry hashes/ids if present
            "ResumeHash": ci.get("ResumeHash") or pre.get("ResumeHash", ""),
            "NomFichier": ci.get("NomFichier") or pre.get("file_meta", {}).get("NomFichier", ""),
            "UrlFichier": ci.get("UrlFichier") or pre.get("file_meta", {}).get("UrlFichier", ""),
            "DocumentId": pre.get("DocumentId", ""),
        })
        (out_dir / "merged.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

        validated = fill_all(CANDIDATE_SCHEMA, merged)
        # add analysis timestamp
        try:
            validated["DateAnalyse"] = datetime.now(timezone.utc).isoformat()
        except Exception:
            pass
        (out_dir / "validated_merged.json").write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
        # write canonical payload copy
        (out_dir / "payload_canonical.json").write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")

        audit = {
            "correlation_id": corr,
            "sources": {k: ("det" if k in ci and ci[k] else "default") for k in validated.keys()},
        }
        (out_dir / "audit_report.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

        return {"candidate_json": validated, "audit_report": audit}


def build_graph() -> Any:
    return _Pipeline()

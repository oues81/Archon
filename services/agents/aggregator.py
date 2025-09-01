# -*- coding: utf-8 -*-
"""AggregatorAgent (skeleton): merge, validate (later), and fill-all defaults.
"""
from __future__ import annotations
from typing import Dict, Any

DEFAULTS = {
    "string": "Non disponible",
    "number": 0,
    "boolean": False,
}


def fill_all(schema: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    for k, spec in (schema or {}).items():
        t = spec.get("type", "string") if isinstance(spec, dict) else "string"
        if out.get(k) in (None, ""):
            out[k] = DEFAULTS.get(t, DEFAULTS["string"])
    # Drop extras not in schema
    allowed = set(schema.keys()) if isinstance(schema, dict) else set(out.keys())
    return {k: out.get(k) for k in allowed}

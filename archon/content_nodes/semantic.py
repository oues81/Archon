from pathlib import Path
import json
from typing import Dict, Any, List, Optional
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
    out = art_root / "content_insights_root.json"

    # Locate inventory from state or default path
    inv_path = None
    if isinstance(state, dict):
        inv_path = state.get("inventory_path")
    if not inv_path:
        cand = art_root / "global_inventory.json"
        inv_path = str(cand) if cand.exists() else None

    inventory: Dict[str, Any] = {"items": []}
    if inv_path and Path(inv_path).exists():
        try:
            inventory = json.loads(Path(inv_path).read_text(encoding="utf-8"))
        except Exception:
            inventory = {"items": []}

    # Truncate for prompt safety
    items: List[Dict[str, Any]] = inventory.get("items") or []
    max_items = 400  # keep token usage reasonable
    short_items = items[:max_items]

    provider = get_llm_provider()
    model = getattr(provider.config, "advisor_model", None) or provider.config.primary_model
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
    # Optional temperature from config
    temperature: Optional[float] = None
    try:
        cfg_temp = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
        if isinstance(cfg_temp, dict):
            t = cfg_temp.get("temperature")
            if isinstance(t, (int, float)):
                temperature = float(t)
    except Exception:
        temperature = None

    logger.info(
        "🔎 [semantic.start] module=archon.archon.content_nodes.semantic:run_semantic | inv_items=%s | sample=%s | model=%s",
        len(items), len(short_items), model
    )
    t0 = time.time()

    system = (
        "You are a documentation and codebase organization advisor.\n"
        "Group files into meaningful clusters by purpose/topic.\n"
        "Return STRICT JSON with the shape: {\n"
        "  \"clusters\": [ { \"label\": str, \"members\": [str], \"confidence\": number, \"rationale\": str } ],\n"
        "  \"unassigned\": [str],\n"
        "  \"notes\": str\n"
        "}.\n"
        "Do not include any extra text outside JSON. Members are file paths from input."
    )
    user = {
        "task": "Cluster these repository files by semantics and purpose",
        "hints": {
            "examples": [
                "docs/* grouped by product area and lifecycle stage",
                "scripts/* grouped by ingestion/build/deploy",
                "services/* grouped by domain context",
            ],
            "constraints": [
                "Only cluster items provided. Do not invent paths.",
                "Prefer fewer, high-quality clusters with rationale",
                "Use unassigned for ambiguous/outliers"
            ]
        },
        "items_sample": short_items
    }

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)},
    ]

    def _strip_fences(s: str) -> str:
        s = s.strip()
        # remove ```json ... ``` or ``` ... ``` wrappers
        if s.startswith("```"):
            # drop first fence line
            parts = s.split("\n")
            if parts:
                parts = parts[1:]
            # remove trailing fence if present
            if parts and parts[-1].strip().startswith("```"):
                parts = parts[:-1]
            s = "\n".join(parts).strip()
        return s

    def _extract_json_block(s: str) -> str:
        # attempt to find first balanced JSON object or array
        s = s.strip()
        start_indices = [s.find("{"), s.find("[")]
        start_indices = [i for i in start_indices if i != -1]
        if not start_indices:
            return s
        start = min(start_indices)
        stack = []
        for idx in range(start, len(s)):
            ch = s[idx]
            if ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if not stack:
                    break
                open_ch = stack.pop()
                if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                    # mismatch; ignore
                    pass
                if not stack:
                    # end of top-level
                    return s[start:idx + 1]
        return s[start:] if start < len(s) else s

    def _parse_json_strict(txt: str) -> Dict[str, Any]:
        cleaned = _strip_fences(txt)
        try:
            return json.loads(cleaned)
        except Exception:
            block = _extract_json_block(cleaned)
            return json.loads(block)

    extra = {"response_format": {"type": "json_object"}}
    if temperature is not None:
        extra["temperature"] = max(0.0, min(1.0, float(temperature)))

    result: Dict[str, Any]
    try:
        txt = provider.generate(messages, model=model, extra=extra)
        result = _parse_json_strict(txt)
    except Exception as e1:
        # Retry once with stricter directive
        retry_messages = [
            {"role": "system", "content": system + "\nReturn ONLY valid JSON. No prose, no code fences."},
            {"role": "user", "content": json.dumps(user)},
        ]
        try:
            txt2 = provider.generate(retry_messages, model=model, extra={**extra, "temperature": 0.1})
            result = _parse_json_strict(txt2)
        except Exception as e2:
            logger.warning("⚠️  [semantic.error] LLM or parse failed: %s", str(e2))
            result = {"clusters": [], "unassigned": [i.get("path") for i in short_items], "notes": "fallback"}

    # Normalize result
    clusters = result.get("clusters") or []
    unassigned = result.get("unassigned") or []
    notes = result.get("notes") or ""
    out_data = {
        "note": "Semantic clusters (Part 1, advisory)",
        "clusters": clusters,
        "unassigned": unassigned,
        "source_inventory": inv_path,
        "limits": {"sampled": len(short_items)},
        "model": model,
        "notes": notes,
    }
    out.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    dur_ms = int((time.time() - t0) * 1000)
    logger.info(
        "✅ [semantic.done] module=archon.archon.content_nodes.semantic:run_semantic | clusters=%s | unassigned=%s | dur_ms=%s | out=%s",
        len(clusters), len(unassigned), dur_ms, str(out)
    )
    return {"insights_path": str(out)}

import traceback
import sys
import os
import json
import time
import base64
import hashlib
import logging
import asyncio
import dataclasses
from datetime import datetime
from pathlib import Path

from archon.utils.utils import configure_logging

_log_summary = configure_logging()
logger = logging.getLogger(__name__)
logger.info(
    f"🚀 API startup | console={_log_summary.get('console')} file={_log_summary.get('file')}"
    f" path={_log_summary.get('file_path')} json={_log_summary.get('json')}"
)

# Appliquer le correctif pour TypedDict
try:
    from patch_typing import *
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Optional, Dict, Any, List

# Import du router des profils (support both legacy and proper package path)
try:
    # Preferred: within package namespace
    from archon.api.profiles import router as profiles_router
    profiles_available = True
except Exception:
    try:
        # Legacy fallback when running with repo root on PYTHONPATH
        from api.profiles import router as profiles_router  # type: ignore
        profiles_available = True
    except Exception:
        profiles_router = None
        profiles_available = False
        logging.warning("Module archon.api.profiles non disponible")

from archon.archon.graphs.archon.app.graph import get_agentic_flow
from archon.archon.graphs.docs_maintainer_graph import get_docs_flow
from archon.archon.graphs.content_restructurer_graph import get_content_flow
from archon.utils.utils import write_to_log
from archon.archon.utils.logging_utils import redact_pii
from archon.llm import get_llm_provider
from archon.archon.security.hmac import verify as verify_hmac
import requests

try:
    from langgraph.types import Command
except ImportError:
    # Fallback implementation if langgraph.types.Command is not available
    class Command:
        """Dummy implementation of Command for compatibility"""
        pass
    
app = FastAPI()

# --- Helpers ---
def sanitize_output(text: str) -> str:
    """Remove common 'thinking' or meta wrappers that some models leak into final output.

    Keeps this conservative: only strips known markers at the start/end without altering content in-between.
    """
    if not isinstance(text, str) or not text:
        return text
    t = text.strip()
    # Remove common XML-like think blocks at the boundaries
    if t.lower().startswith("<think>") and t.lower().endswith("</think>"):
        return t[7:-8].strip()
    # Remove leading labels often used by some models
    for prefix in ("Thinking:", "Think:", "Thought:", "Analysis:"):
        if t.startswith(prefix):
            return t[len(prefix):].lstrip()
    # Strip surrounding triple backtick fences, with or without language tag
    if t.startswith("```") and t.endswith("```"):
        inner = t[3:-3].strip()
        # If a language tag is present at the start of inner, remove first line
        if "\n" in inner:
            first_line, rest = inner.split("\n", 1)
            if first_line.strip().lower() in ("json", "markdown", "md", "text", "yaml", "yml", "toml"):
                return rest.strip()
        return inner
    return text

def _log_safe(msg: str, **fields: Any) -> None:
    """PII-safe logging: only log metadata like correlation_id, tool, durations, statuses.

    Redacts any unexpected content using redact_pii() as a defense-in-depth measure.
    """
    whitelist = {"correlation_id", "tool", "duration_ms", "status", "http", "resume_hash", "latency_ms", "status_code", "endpoint"}
    safe_raw = {k: v for k, v in fields.items() if k in whitelist}
    safe = redact_pii(safe_raw)
    logger.info(f"{msg} | {json.dumps(safe, ensure_ascii=False)}")

# Montage du router des profils s'il est disponible
if profiles_available and profiles_router:
    app.include_router(profiles_router, prefix="/api")
    logging.info("✅ Router des profils monte avec succes")
else:
    logging.warning("⚠️ Router des profils non disponible")

class InvokeRequest(BaseModel):
    message: str
    thread_id: str
    is_first_message: bool = False
    config: Optional[Dict[str, Any]] = None
    profile_name: Optional[str] = None  # Ajout du nom du profil

# --- Advisor Ingestion Schemas ---
class AdvisorIngestRequest(BaseModel):
    # Mode direct
    filename: Optional[str] = None
    file_base64: Optional[str] = None
    # Mode SharePoint
    share_url: Optional[str] = None
    site_id: Optional[str] = None
    drive_id: Optional[str] = None
    item_id: Optional[str] = None
    pages: Optional[str] = None
    # Contrôles
    consent: bool = True
    policy_id: Optional[str] = None
    profile: Optional[str] = None
    privacy_mode: Optional[str] = Field(default="local", pattern=r"^(local|cloud_guarded)$")
    metadata: Optional[Dict[str, Any]] = None
    # Options PR-4
    do_export: Optional[bool] = False
    export_format: Optional[str] = Field(default="json", pattern=r"^(json|csv|md)$")
    do_search: Optional[bool] = False
    search_query: Optional[str] = None
    top_k: Optional[int] = Field(default=5, ge=1, le=50)

class AdvisorIngestResponse(BaseModel):
    parsed: Dict[str, Any]
    score: Dict[str, Any]
    anonymized: Optional[Dict[str, Any]] = None
    index_ref: Dict[str, Any]
    logs_ref: Dict[str, Any]
    correlation_id: Optional[str] = None
    export_ref: Optional[Dict[str, Any]] = None
    search_results: Optional[List[Dict[str, Any]]] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}    

@app.head("/health")
async def health_head():
    """HEAD variant for health checks (no body)."""
    return {"status": "ok"}

@app.post("/advisor/ingest_cv")
async def advisor_ingest_cv(req: Request, payload: AdvisorIngestRequest) -> AdvisorIngestResponse:
    """Advisor-first ingestion endpoint (skeleton).

    - Verifies HMAC if HMAC_SECRET is set
    - Validates mutually exclusive modes: (filename+file_base64) XOR (share reference)
    - Propagates X-Correlation-Id
    - Returns a minimal structured response (no MCP calls yet)
    """
    start_t = time.time()
    try:
        raw_body = await req.body()
        auth = req.headers.get("Authorization")
        ts = req.headers.get("X-Timestamp")
        corr_id = req.headers.get("X-Correlation-Id") or ""
        x_source = (req.headers.get("X-Source") or "").strip().lower()
        recursion_guard = x_source == "rhcv"
        if recursion_guard:
            _log_safe("recursion_guard", endpoint="/advisor/ingest_cv", correlation_id=corr_id, status="active")

        # Enforce HMAC only if secret is configured
        hmac_required = bool(os.getenv("HMAC_SECRET"))
        ok, msg = verify_hmac(auth, ts, raw_body, required=hmac_required)
        if not ok:
            raise HTTPException(status_code=401, detail=f"HMAC verification failed: {msg}")

        # Ensure correct LLM profile (default to openrouter_default if unspecified)
        try:
            requested_profile = (payload.profile or "").strip() or (req.headers.get("X-Profile-Name") or "").strip()
            if not requested_profile:
                requested_profile = "openrouter_default"
            provider = get_llm_provider()
            if hasattr(provider, "reload_profile"):
                ok_reload = provider.reload_profile(requested_profile)
                if not ok_reload:
                    # Try fallback to openrouter_default if explicit name failed
                    if requested_profile != "openrouter_default":
                        _log_safe("profile_reload_warn", endpoint="/advisor/ingest_cv", correlation_id=corr_id, status="fallback", tool="profile")
                        provider.reload_profile("openrouter_default")
            # Sanity log
            try:
                cfg = getattr(provider, "config", None)
                if cfg:
                    _log_safe("profile_active", endpoint="/advisor/ingest_cv", correlation_id=corr_id, status="ok", tool="profile")
            except Exception:
                pass
        except Exception as _e:
            logger.warning(f"Advisor ingest: profile selection failed: {_e}")

        # Validate mode selection
        has_file = bool(payload.filename and payload.file_base64)
        has_sp = bool(payload.share_url) or bool(payload.site_id and payload.drive_id and payload.item_id)
        if has_file == has_sp:
            # Either both or neither provided
            raise HTTPException(status_code=400, detail="Provide exactly one input mode: (filename+file_base64) OR (share_url | site_id+drive_id+item_id)")

        # Basic MIME/size guard for direct mode (lightweight check; full check to be added later)
        if has_file:
            # Filename extension guard
            allowed_ext = {".pdf", ".doc", ".docx"}
            try:
                ext = Path(payload.filename).suffix.lower()
            except Exception:
                ext = ""
            if ext not in allowed_ext:
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")
            # Size guard (decoded)
            try:
                max_bytes = int(os.getenv("ARCHON_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
            except Exception:
                max_bytes = 10 * 1024 * 1024
            try:
                decoded = base64.b64decode((payload.file_base64 or "").encode("utf-8"), validate=False)
                if len(decoded) > max_bytes:
                    raise HTTPException(status_code=413, detail=f"File too large. Max {max_bytes} bytes")
            except HTTPException:
                raise
            except Exception:
                # If decode fails here, parsing step will handle; don't double-raise
                pass

        # SharePoint pages validation (if provided)
        if has_sp and (payload.pages or "").strip():
            import re as _re
            pages = (payload.pages or "").strip()
            pattern = r"^\d+(\s*-\s*\d+)?(\s*,\s*\d+(\s*-\s*\d+)?)*$"
            if not _re.match(pattern, pages):
                raise HTTPException(status_code=400, detail="Invalid pages format. Use e.g. '1-3,5'")

        # --- PR-3: idempotence hash (stateless for now) ---
        def _sha256_hex(b: bytes) -> str:
            h = hashlib.sha256()
            h.update(b or b"")
            return h.hexdigest()

        resume_hash = None
        try:
            if has_file and payload.file_base64:
                resume_hash = _sha256_hex(base64.b64decode(payload.file_base64.encode('utf-8'), validate=False))
            elif has_sp:
                meta = json.dumps({
                    "share_url": payload.share_url,
                    "site_id": payload.site_id,
                    "drive_id": payload.drive_id,
                    "item_id": payload.item_id,
                    "pages": payload.pages,
                }, sort_keys=True).encode('utf-8')
                resume_hash = _sha256_hex(meta)
        except Exception:
            resume_hash = None

        # --- Helper to call MCP tools via JSON-RPC ---
        # Prefer explicit ARCHON_MCP_URL, else fallback to MCP_SERVER_URL (docker-compose env),
        # else default to localhost for bare-metal runs.
        MCP_URL = (os.getenv("ARCHON_MCP_URL") or os.getenv("MCP_SERVER_URL") or "http://localhost:8100").rstrip('/')
        CV_API_URL = os.getenv("CV_API_URL")

        def _mcp_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
            rpc = {
                "jsonrpc": "2.0",
                "id": f"{int(time.time()*1000)}-{tool_name}",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            hdrs = {"Content-Type": "application/json", "X-Source": "archon"}
            if corr_id:
                hdrs["X-Correlation-Id"] = corr_id
            # minimal retry on 429/504
            delays = [0, 1.5]
            last_exc = None
            started = time.time()
            for d in delays:
                if d:
                    time.sleep(d)
                try:
                    r = requests.post(f"{MCP_URL}/mcp", json=rpc, headers=hdrs, timeout=int(os.getenv("ARCHON_HTTP_TIMEOUT", "120")))
                    if r.status_code in (200, 304):
                        _log_safe("mcp_call_ok", correlation_id=corr_id, tool=tool_name, duration_ms=int((time.time()-started)*1000), http=r.status_code, status="ok")
                        return r.json()
                    if r.status_code not in (429, 504):
                        _log_safe("mcp_call_fail_status", correlation_id=corr_id, tool=tool_name, duration_ms=int((time.time()-started)*1000), http=r.status_code, status="fail")
                        break
                except Exception as e:
                    last_exc = e
            _log_safe("mcp_call_error", correlation_id=corr_id, tool=tool_name, duration_ms=int((time.time()-started)*1000), status="error")
            if last_exc:
                raise last_exc
            raise HTTPException(status_code=502, detail=f"MCP call failed for {tool_name}: HTTP {r.status_code}")

        def _extract_or_raise(tool: str, resp: Dict[str, Any]) -> Dict[str, Any]:
            """Extract JSON result.content[0].json or map JSON-RPC error to HTTP.
            Known mappings:
            - 429: Too Many Requests
            - 504: Gateway Timeout
            - 413: Payload Too Large (if reported upstream)
            - 409: Conflict (e.g., index exists when upsert=False)
            - -32602: Invalid params -> 400
            - -32052: SharePoint parse failure -> 501 (graceful degradation)
            - Others -> 502
            """
            if not isinstance(resp, dict):
                raise HTTPException(status_code=502, detail=f"Invalid MCP response for {tool}")
            if "error" in resp and resp["error"]:
                err = resp["error"] or {}
                code = err.get("code")
                message = err.get("message") or f"{tool} failed"
                # Direct HTTP code passthrough when applicable
                if isinstance(code, int) and code in (409, 413, 429, 504):
                    raise HTTPException(status_code=code, detail=message)
                # JSON-RPC style mappings
                if code == -32602:
                    raise HTTPException(status_code=400, detail=message)
                if code == -32052 and tool == "cv_parse_sharepoint":
                    raise HTTPException(status_code=501, detail=message)
                # Default
                raise HTTPException(status_code=502, detail=message)
            try:
                return (((resp or {}).get("result") or {}).get("content") or [{}])[0].get("json") or {}
            except Exception:
                raise HTTPException(status_code=502, detail=f"Malformed MCP result for {tool}")

        # --- PR-4: Orchestration using existing MCP tools ---
        # 1) Health check (best-effort)
        try:
            _ = _mcp_call("cv_health_check", {"api_url": CV_API_URL, "correlation_id": corr_id})
        except Exception:
            pass

        # 2) Parse
        parsed: Dict[str, Any] = {}
        if has_file:
            args = {"filename": payload.filename, "file_base64": payload.file_base64}
            if CV_API_URL:
                args["api_url"] = CV_API_URL
            _log_safe("tool_start", correlation_id=corr_id, tool="cv_parse_v2", status="start")
            _t_parse = time.time()
            pr = _mcp_call("cv_parse_v2", args)
        else:
            sp_args: Dict[str, Any] = {}
            if payload.share_url:
                sp_args["share_url"] = payload.share_url
            else:
                sp_args.update({"site_id": payload.site_id, "drive_id": payload.drive_id, "item_id": payload.item_id})
            if payload.pages:
                sp_args["pages"] = payload.pages
            if CV_API_URL:
                sp_args["api_url"] = CV_API_URL
            _log_safe("tool_start", correlation_id=corr_id, tool="cv_parse_sharepoint", status="start")
            _t_parse = time.time()
            pr = _mcp_call("cv_parse_sharepoint", sp_args)
        parsed = _extract_or_raise("cv_parse_v2" if has_file else "cv_parse_sharepoint", pr)
        try:
            _log_safe("tool_done", correlation_id=corr_id, tool=("cv_parse_v2" if has_file else "cv_parse_sharepoint"), duration_ms=int((time.time()-_t_parse)*1000), status="ok")
        except Exception:
            pass

        # 3) Score
        score: Dict[str, Any] = {}
        try:
            _log_safe("tool_start", correlation_id=corr_id, tool="cv_score", status="start")
            _t_score = time.time()
            sc = _mcp_call("cv_score", {"parsed": parsed, "profile_id": (payload.metadata or {}).get("profile_id") if payload.metadata else None})
            score = _extract_or_raise("cv_score", sc)
            try:
                _log_safe("tool_done", correlation_id=corr_id, tool="cv_score", duration_ms=int((time.time()-_t_score)*1000), status="ok")
            except Exception:
                pass
        except HTTPException:
            raise
        except Exception:
            score = {}

        # 4) Anonymize if privacy/policy requires before export/index
        anonymized: Optional[Dict[str, Any]] = None
        need_anon = (payload.privacy_mode == "cloud_guarded") or (payload.consent is False) or bool(payload.policy_id)
        if need_anon:
            try:
                _log_safe("tool_start", correlation_id=corr_id, tool="cv_anonymize", status="start")
                _t_anon = time.time()
                an = _mcp_call("cv_anonymize", {"parsed": parsed})
                anonymized = _extract_or_raise("cv_anonymize", an)
                try:
                    _log_safe("tool_done", correlation_id=corr_id, tool="cv_anonymize", duration_ms=int((time.time()-_t_anon)*1000), status="ok")
                except Exception:
                    pass
                if not anonymized:
                    # Treat empty/malformed anonymization as failure when required
                    anonymized = None
            except HTTPException:
                raise
            except Exception:
                anonymized = None

        # 5) Index (skip if consent=false or privacy=local)
        index_ref: Dict[str, Any] = {"skipped": True}
        may_index = payload.consent and (payload.privacy_mode != "local")
        if may_index:
            try:
                content_for_index = anonymized or parsed
                _log_safe("tool_start", correlation_id=corr_id, tool="cv_index", status="start")
                _t_index = time.time()
                ix = _mcp_call("cv_index", {"doc_id": resume_hash or None, "content": content_for_index, "metadata": {"resume_hash": resume_hash, "source": "advisor_ingest"}, "upsert": True})
                index_ref = _extract_or_raise("cv_index", ix)
                try:
                    _log_safe("tool_done", correlation_id=corr_id, tool="cv_index", duration_ms=int((time.time()-_t_index)*1000), status="ok")
                except Exception:
                    pass
            except HTTPException as he:
                # Bubble up 409/413/429/504 etc.
                raise he
            except Exception:
                index_ref = {"error": "index_failed"}

        # 6) Optional export
        export_ref: Optional[Dict[str, Any]] = None
        if bool(getattr(payload, "do_export", False)):
            try:
                # If anonymization is required but failed or empty, do not export raw data
                if need_anon and not anonymized:
                    raise HTTPException(status_code=502, detail="Anonymization required for export but failed")
                exp_args: Dict[str, Any] = {"parsed": anonymized or parsed, "format": (payload.export_format or "json")}
                _log_safe("tool_start", correlation_id=corr_id, tool="cv_export", status="start")
                _t_export = time.time()
                ex = _mcp_call("cv_export", exp_args)
                export_ref = _extract_or_raise("cv_export", ex)
                try:
                    _log_safe("tool_done", correlation_id=corr_id, tool="cv_export", duration_ms=int((time.time()-_t_export)*1000), status="ok")
                except Exception:
                    pass
            except HTTPException:
                raise
            except Exception:
                export_ref = {"error": "export_failed"}

        # 7) Optional search
        search_results: Optional[List[Dict[str, Any]]] = None
        if bool(getattr(payload, "do_search", False)) and (payload.search_query or "").strip():
            try:
                sq = (payload.search_query or "").strip()
                top_k = payload.top_k or 5
                _log_safe("tool_start", correlation_id=corr_id, tool="cv_search", status="start")
                _t_search = time.time()
                sr = _mcp_call("cv_search", {"query": sq, "top_k": top_k})
                data = _extract_or_raise("cv_search", sr)
                # Expect either {results: [...]} or list directly
                if isinstance(data, dict) and isinstance(data.get("results"), list):
                    search_results = data.get("results")
                elif isinstance(data, list):
                    search_results = data
                else:
                    search_results = []
                try:
                    _log_safe("tool_done", correlation_id=corr_id, tool="cv_search", duration_ms=int((time.time()-_t_search)*1000), status="ok")
                except Exception:
                    pass
            except HTTPException:
                raise
            except Exception:
                search_results = []

        resp = AdvisorIngestResponse(
            parsed=parsed or {"status": "empty"},
            score=score or {"status": "empty"},
            anonymized=anonymized,
            index_ref=index_ref,
            logs_ref={"correlation_id": corr_id or None, "resume_hash": resume_hash},
            correlation_id=corr_id or None,
            export_ref=export_ref,
            search_results=search_results,
        )
        _log_safe("ingest_latency", endpoint="/advisor/ingest_cv", latency_ms=int((time.time()-start_t)*1000), status_code=200, correlation_id=corr_id, resume_hash=resume_hash)
        return resp
    except HTTPException as he:
        try:
            # Best-effort latency log on errors
            corr_id = req.headers.get("X-Correlation-Id") or ""
            _log_safe("ingest_latency", endpoint="/advisor/ingest_cv", latency_ms=int((time.time()-start_t)*1000), status_code=he.status_code, correlation_id=corr_id)
        except Exception:
            pass
        raise
    except Exception as e:
        logger.error(f"Exception in advisor_ingest_cv: {e}", exc_info=True)
        try:
            corr_id = req.headers.get("X-Correlation-Id") or ""
            _log_safe("ingest_latency", endpoint="/advisor/ingest_cv", latency_ms=int((time.time()-start_t)*1000), status_code=500, correlation_id=corr_id)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/run")
async def agent_run(payload: InvokeRequest):
    """Stable endpoint alias for running the agent.

    This is a thin wrapper around /invoke to provide a canonical URL for external
    callers (e.g., RHCV assistant). The request/response schema is identical to /invoke.
    """
    return await invoke_agent(payload)

@app.post("/invoke")
async def invoke_agent(request: InvokeRequest):
    """
    Process a message through the agentic flow. This endpoint can dynamically
    apply a configuration profile for the duration of the request.
    """
    try:
        # Get the singleton instance of the LLM provider
        provider = get_llm_provider()

        # Reload the provider with the requested profile if specified
        if request.profile_name:
            if not provider.reload_profile(request.profile_name):
                raise ValueError(f"Failed to load configuration for profile '{request.profile_name}'")

        if not provider.config:
            raise ValueError("LLM provider configuration is not loaded.")

        # Build the configuration dictionary from the provider's config
        llm_config = {
            'LLM_PROVIDER': provider.config.provider,
            'REASONER_MODEL': provider.config.reasoner_model,
            'PRIMARY_MODEL': provider.config.primary_model,
            'CODER_MODEL': provider.config.coder_model,
            'ADVISOR_MODEL': provider.config.advisor_model,
            # For Ollama-based providers this is used, but some agents expect BASE_URL as well
            'OLLAMA_BASE_URL': provider.config.base_url,
            # Ensure BASE_URL is present for providers like OpenRouter/OpenAI that rely on it
            'BASE_URL': provider.config.base_url,
        }

        # Inject a default timeout into llm_config if not set elsewhere
        try:
            timeout_env = os.getenv("ARCHON_HTTP_TIMEOUT", "120")
            timeout_s = int(timeout_env) if str(timeout_env).isdigit() else 120
        except Exception:
            timeout_s = 120
        llm_config['TIMEOUT_S'] = llm_config.get('TIMEOUT_S', timeout_s)

        # Add the API key with the expected name for compatibility
        if provider.config.api_key:
            llm_config['LLM_API_KEY'] = provider.config.api_key
            if provider.config.provider == 'openrouter':
                llm_config['OPENROUTER_API_KEY'] = provider.config.api_key
            elif provider.config.provider == 'openai':
                llm_config['OPENAI_API_KEY'] = provider.config.api_key
    except Exception as e:
        logger.error(f"Error loading profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load profile {request.profile_name}: {str(e)}")

    # Prepare configuration for the agentic flow (and approval controls if provided)
    # Merge the full caller-provided configurable, but enforce thread_id and llm_config.
    base_cfg = {"thread_id": request.thread_id, "llm_config": llm_config}
    caller_cfg = {}
    try:
        caller_cfg = (request.config or {}).get("configurable", {}) if isinstance(request.config, dict) else {}
        if not isinstance(caller_cfg, dict):
            caller_cfg = {}
    except Exception:
        caller_cfg = {}
    merged_cfg = {**caller_cfg, **base_cfg}
    # Inject timestamped run_id and output_root if not provided
    try:
        if not merged_cfg.get("run_id"):
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            merged_cfg["run_id"] = f"{ts}_{request.thread_id}"
        if not merged_cfg.get("output_root"):
            merged_cfg["output_root"] = str(Path("generated") / "sessions" / merged_cfg["run_id"])
    except Exception:
        pass
    config = {"configurable": merged_cfg}

    # Structured request log with profile + models + effective config snapshot
    try:
        provider_name = llm_config.get('LLM_PROVIDER')
        profile_used = request.profile_name or getattr(getattr(get_llm_provider(), 'config', None), 'profile_name', None) or 'default'
        models = {
            'PRIMARY_MODEL': llm_config.get('PRIMARY_MODEL'),
            'REASONER_MODEL': llm_config.get('REASONER_MODEL'),
            'ADVISOR_MODEL': llm_config.get('ADVISOR_MODEL'),
            'CODER_MODEL': llm_config.get('CODER_MODEL'),
        }
        cfg_snap = {
            'delegate_to': merged_cfg.get('delegate_to'),
            'requested_action': merged_cfg.get('requested_action'),
            'targets': (merged_cfg.get('targets') or [])[:5],
            'include_ext': (merged_cfg.get('include_ext') or [])[:10],
            'max_files': merged_cfg.get('max_files'),
            'allow_directory_moves': merged_cfg.get('allow_directory_moves'),
        }
        # Pretty banner and model/profile summary
        logger.info("\n" +
                    "✨================================ INVOKE START ================================✨\n"
                    f"🧵 thread={request.thread_id}  |  👤 profile={request.profile_name}  |  ☁️ provider={provider_name}\n"
                    f"🤖 models: PRIMARY={llm_config.get('PRIMARY_MODEL')} | REASONER={llm_config.get('REASONER_MODEL')} | ADVISOR={llm_config.get('ADVISOR_MODEL')} | CODER={llm_config.get('CODER_MODEL')}\n"
                    f"🛠️  effective_cfg={cfg_snap}\n"
                    f"📂 run_id={merged_cfg.get('run_id')} | output_root={merged_cfg.get('output_root')}\n"
                    "==============================================================================")
    except Exception:
        logger.info(f"Starting invoke_agent for thread {request.thread_id} with profile {request.profile_name or 'default'}")
    
    # Optional output size cap from caller config, with env fallback
    max_response_chars: Optional[int] = None
    try:
        if request.config and isinstance(request.config, dict):
            cfg = request.config.get("configurable", {})
            if isinstance(cfg, dict) and cfg.get("max_response_chars") is not None:
                max_response_chars = int(cfg.get("max_response_chars"))
    except Exception:
        max_response_chars = None
    # Environment fallback if not provided in request
    if max_response_chars is None:
        try:
            env_val = os.environ.get("MAX_RESPONSE_CHARS") or os.environ.get("MCP_MAX_RESPONSE_CHARS")
            if env_val is not None:
                max_response_chars = int(env_val)
            else:
                # Default sensible cap for responsiveness
                max_response_chars = 3500
        except Exception:
            max_response_chars = 3500

    response = ""
    final_state = None
    steps_out: List[Dict[str, Any]] = []
    node_started_at: Dict[str, float] = {}
    
    def _summarize_json_text(txt: str, max_len: int = 500) -> str:
        try:
            data = json.loads(txt)
            if isinstance(data, dict):
                keys = list(data.keys())
                # include lightweight hints
                summary = {
                    "type": "object",
                    "keys": keys[:10],
                    "size": len(keys),
                }
                # common known fields
                for k in ("count", "clusters", "unassigned", "notes", "limits", "model"):
                    if k in data:
                        summary[k] = data.get(k)
                # items length hint
                if "items" in data and isinstance(data["items"], list):
                    summary["items_len"] = len(data["items"])
                return json.dumps(summary)[:max_len]
            if isinstance(data, list):
                head = data[:1]
                return json.dumps({"type": "array", "len": len(data), "head": head})[:max_len]
        except Exception:
            pass
        return (txt or "")[:max_len]
    
    # This is the original, correct logic for handling the agent state
    # Prepare the initial state for the agentic flow
    # The key is to provide the user's message as the 'latest_user_message'
    # and ensure the scope is initialized with the user's request
    initial_state = {
        "latest_user_message": request.message,
        "next_user_message": "",
        "messages": [{
            "role": "user",
            "content": request.message
        }],
        "scope": request.message,  # Pass the user's message as the initial scope
        "advisor_output": "",
        "file_list": [],
        "refined_prompt": "",
        "refined_tools": "",
        "refined_agent": "",
        "generated_code": None,
        "error": None
    }
    input_for_flow = initial_state

    try:
        # Choose flow by profile_name (or optional config flag), else default
        flow = None
        try:
            # Delegation has highest precedence
            delegate_to = None
            cfg = (request.config or {}) if isinstance(request.config, dict) else {}
            if isinstance(cfg, dict):
                delegate_to = (cfg.get("configurable", {}) or {}).get("delegate_to")

            if isinstance(delegate_to, str) and delegate_to.strip():
                name = delegate_to.strip()
                if name == "DocsMaintainer":
                    flow = get_docs_flow()
                elif name == "ContentRestructurer":
                    flow = get_content_flow()

            # If no delegation, select by profile_name
            if flow is None and request.profile_name:
                name = request.profile_name.strip()
                if name == "DocsMaintainer":
                    flow = get_docs_flow()
                elif name == "ContentRestructurer":
                    flow = get_content_flow()

            # Optional explicit flow hint in config
            if flow is None and isinstance(cfg, dict):
                flow_hint = (cfg.get("configurable", {}) or {}).get("flow")
                if isinstance(flow_hint, str):
                    name = flow_hint.strip()
                    if name == "DocsMaintainer":
                        flow = get_docs_flow()
                    elif name == "ContentRestructurer":
                        flow = get_content_flow()
        except Exception:
            flow = None

        flow_name = "default"
        if flow is None:
            flow = get_agentic_flow()
        else:
            flow_name = "DocsMaintainer" if flow is get_docs_flow() else ("ContentRestructurer" if flow is get_content_flow() else "custom")
        logger.info(f"🗺️  flow_selected: {flow_name}  |  thread={request.thread_id}")
        logger.info("📌 graph_plan: ensure_dirs ▶ inventory ▶ semantic ▶ plan ▶ analysis_report ▶ assistant_brief ▶ approval ▶ apply_moves ▶ final_report")
        logger.debug(f"About to start flow.astream with input: {input_for_flow}")
        logger.debug(f"Config: {config}")

        iteration_count = 0
        # Stage milestone flags to avoid duplicate logs
        s_inventory = s_semantic = s_plan = s_analysis_report = s_approval = s_apply = s_report = False
        STAGE_MODULES = {
            "inventory": "archon.archon.content_nodes.inventory:run_inventory",
            "semantic": "archon.archon.content_nodes.semantic:run_semantic",
            "plan": "archon.archon.content_nodes.plan:run_plan",
            "analysis_report": "archon.archon.content_nodes.analysis_report:run",
            "assistant_brief": "archon.archon.graph_service:assistant_brief",
            "approval": "archon.archon.graph_service:approval_gate",
            "apply_moves": "archon.archon.restruct_common.git_apply:apply_moves",
            "final_report": "archon.archon.graph_service:final_report",
        }
        async for state_update in flow.astream(input_for_flow, config, stream_mode="values"):
            iteration_count += 1
            logger.debug(f"Iteration {iteration_count}, state_update: {state_update}")
            final_state = state_update
            # Emit stage-aligned logs when keys appear in state
            try:
                for node in STAGE_MODULES:
                    if not locals()[f"s_{node}"] and (state_update.get(node) or state_update.get(f"{node}_path") or state_update.get(f"{node}_count") is not None):
                        mod = STAGE_MODULES.get(node, "<unknown>")
                        # Compute step duration if we have a start
                        try:
                            now_ts = datetime.now().timestamp()
                            if node not in node_started_at:
                                node_started_at[node] = now_ts
                        except Exception:
                            now_ts = None
                        if node == "inventory":
                            count = (state_update.get("inventory_count") or state_update.get("count") or 0)
                            inv_path = state_update.get("inventory_path") or "generated/restruct/global_inventory.json"
                            logger.info(f"✅ [{node}_done] module={mod} | thread={request.thread_id} | count={count} | out={inv_path}")
                        elif node == "semantic":
                            insight_path = state_update.get("insights_path") or state_update.get("content_insights_root") or "generated/restruct/content_insights_root.json"
                            clusters = state_update.get("clusters") if isinstance(state_update.get("clusters"), int) else "?"
                            logger.info(f"✅ [{node}_done] module={mod} | thread={request.thread_id} | clusters={clusters} | out={insight_path}")
                        elif node == "plan":
                            moves = (state_update.get("planned_moves") or state_update.get("moves") or 0)
                            plan_path = state_update.get("plan_path") or "generated/restruct/rename_move_plan.json"
                            logger.info(f"✅ [{node}_done] module={mod} | thread={request.thread_id} | moves={moves} | out={plan_path}")
                        elif node == "analysis_report":
                            logger.info(f"✅ [analysis_report_done] module={mod} | thread={request.thread_id} | summary={state_update.get('analysis_summary_path')}")
                        elif node == "assistant_brief":
                            logger.info(f"📝 [report_done] module={mod} | thread={request.thread_id} | brief={state_update.get('assistant_brief_path')} | report={state_update.get('final_report_path')}")
                        elif node == "approval":
                            logger.info(f"🧪 [approval_gate] module={mod} | thread={request.thread_id} | preflight={bool(state_update.get('preflight_ok'))}")
                        elif node == "apply_moves":
                            logger.info(
                                f"🪄 [apply] module={mod} | thread={request.thread_id} | applied={state_update.get('applied',0)} | will_apply={state_update.get('will_apply',0)} | conflicts={state_update.get('conflicts',0)}"
                            )
                            s_approval = True
                        # Append standardized step record with preview + metrics
                        try:
                            start_ts = node_started_at.get(node)
                            dur_ms = int((now_ts - start_ts) * 1000) if (now_ts and start_ts) else None
                        except Exception:
                            dur_ms = None
                        preview = ""
                        metrics: Dict[str, Any] = {}
                        artifacts: Dict[str, Any] = {}
                        # Node-specific preview/metrics from state_update and artifacts
                        try:
                            if node == "inventory":
                                inv_path = state_update.get("inventory_path")
                                count = state_update.get("inventory_count") or state_update.get("count")
                                metrics = {"count": count}
                                if inv_path:
                                    artifacts["inventory_path"] = inv_path
                                if isinstance(inv_path, str) and Path(inv_path).exists():
                                    # Read small slice
                                    txt = Path(inv_path).read_text(encoding="utf-8")
                                    preview = _summarize_json_text(txt)
                                else:
                                    preview = f"inventory_count={count}"
                            elif node == "semantic":
                                insight_path = state_update.get("insights_path") or state_update.get("content_insights_root")
                                clusters = state_update.get("clusters")
                                unassigned = state_update.get("unassigned")
                                metrics = {
                                    "clusters": clusters if isinstance(clusters, int) else None,
                                    "unassigned": unassigned if isinstance(unassigned, int) else None,
                                }
                                if insight_path:
                                    artifacts["insights_path"] = insight_path
                                if isinstance(insight_path, str) and Path(insight_path).exists():
                                    txt = Path(insight_path).read_text(encoding="utf-8")
                                    preview = _summarize_json_text(txt)
                                    # Try refine model from file
                                    try:
                                        data = json.loads(txt)
                                        model_from_file = data.get("model")
                                    except Exception:
                                        model_from_file = None
                                else:
                                    model_from_file = None
                            elif node == "plan":
                                plan_path = state_update.get("plan_path") or state_update.get("rename_move_plan")
                                planned = state_update.get("planned_moves") or state_update.get("moves")
                                metrics = {"planned_moves": planned}
                                if plan_path:
                                    artifacts["plan_path"] = plan_path
                                if isinstance(plan_path, str) and Path(plan_path).exists():
                                    txt = Path(plan_path).read_text(encoding="utf-8")
                                    preview = _summarize_json_text(txt)
                                else:
                                    preview = f"planned_moves={planned}"
                            elif node == "analysis_report":
                                rpt = state_update.get('analysis_summary_path')
                                if rpt:
                                    artifacts["analysis_summary_path"] = rpt
                                if isinstance(rpt, str) and Path(rpt).exists():
                                    txt = Path(rpt).read_text(encoding="utf-8")
                                    preview = txt[:500]
                            elif node == "assistant_brief":
                                brief = state_update.get('assistant_brief_path')
                                if brief:
                                    artifacts["assistant_brief_path"] = brief
                                if isinstance(brief, str) and Path(brief).exists():
                                    txt = Path(brief).read_text(encoding="utf-8")
                                    preview = txt[:500]
                            elif node == "approval":
                                metrics = {"preflight_ok": bool(state_update.get('preflight_ok'))}
                                preview = "preflight_ok=" + str(metrics["preflight_ok"]).lower()
                            elif node == "apply_moves":
                                applied = state_update.get('applied', 0)
                                will_apply = state_update.get('will_apply', 0)
                                conflicts = state_update.get('conflicts', 0)
                                metrics = {"applied": applied, "will_apply": will_apply, "conflicts": conflicts}
                                preview = f"applied={applied} will_apply={will_apply} conflicts={conflicts}"
                            elif node == "final_report":
                                final_path = state_update.get("final_report_path")
                                if final_path:
                                    artifacts["final_report_path"] = final_path
                                if isinstance(final_path, str) and Path(final_path).exists():
                                    txt = Path(final_path).read_text(encoding="utf-8")
                                    preview = txt[:500]
                        except Exception:
                            preview = preview or ""
                        try:
                            step_model = (locals().get('model_from_file') if 'model_from_file' in locals() else None) or llm_config.get('PRIMARY_MODEL')
                        except Exception:
                            step_model = llm_config.get('PRIMARY_MODEL')
                        try:
                            steps_out.append({
                                "agent": mod,
                                "model": step_model,
                                "status": "finished",
                                "duration_ms": dur_ms,
                                "node": node,
                                "prompt": (request.message or "")[:500],
                                "preview": (preview or "")[:500],
                                "metrics": metrics or {},
                                "artifacts": artifacts or {},
                            })
                        except Exception:
                            pass
                        locals()[f"s_{node}"] = True
            except Exception:
                pass
        
        logger.info(f"🏁 INVOKE DONE | thread={request.thread_id} | iterations={iteration_count}")
        logger.info(f"WORKFLOW_OK | run_id={merged_cfg.get('run_id')} | output_root={merged_cfg.get('output_root')} | iterations={iteration_count}")
        logger.info("✨================================ INVOKE END ==================================✨")
        if final_state:
            generated_content = final_state.get("generated_code")
            # Case 1: Content is an error message from the agent
            if isinstance(generated_content, str) and generated_content.strip().startswith("Error:"):
                logger.warning(f"Agent returned an error: {generated_content}")
                response = generated_content
            # Case 2: Content is valid code
            elif generated_content:
                response = generated_content
            # Case 3: No content was generated, check for an error in the state
            else:
                # Prefer returning report artifacts if present
                brief_path = final_state.get("assistant_brief_path")
                final_report_path = final_state.get("final_report_path")
                if isinstance(final_report_path, str) and Path(final_report_path).exists():
                    try:
                        response = Path(final_report_path).read_text(encoding="utf-8")
                        logger.info("Returning final_report.json content (%s chars)", len(response))
                    except Exception:
                        response = f"Report available at {final_report_path}"
                elif isinstance(brief_path, str) and Path(brief_path).exists():
                    try:
                        response = Path(brief_path).read_text(encoding="utf-8")
                        logger.info("Returning assistant_brief.md content (%s chars)", len(response))
                    except Exception:
                        response = f"Assistant brief available at {brief_path}"
                else:
                    error_message = final_state.get("error")
                    if error_message:
                        logger.warning(f"Agent finished with an error state: {error_message}")
                        response = f"Error: {error_message}"
                    else:
                        logger.warning("No generated content and no error message in final state.")
                        response = "Error: The agent finished its work but did not produce any output."
        else:
            logger.error("Agent workflow finished without a final state.")
            response = "Error: The agent workflow failed to complete."

        # Sanitize stray meta/thinking markers before truncation
        if isinstance(response, str):
            response = sanitize_output(response)

        # Optional detailed logging of model response prior to truncation
        try:
            verbose = str(os.getenv("LOG_VERBOSE_RESPONSES", "false")).lower() in {"1", "true", "yes", "on"}
            max_log = int(os.getenv("LOG_RESPONSE_LOG_CHARS", "1000"))
        except Exception:
            verbose = False
            max_log = 1000
        if isinstance(response, str):
            if verbose:
                logger.info(f"📄 Full response (pre-trunc): {response}")
            else:
                snippet = response if len(response) <= max_log else response[:max_log] + " …[snip]"
                logger.debug(f"📄 Response snippet (pre-trunc {max_log}c): {snippet}")

        # Apply optional truncation just before returning
        if isinstance(response, str) and max_response_chars is not None and len(response) > max_response_chars:
            logger.info(
                f"Truncating response for thread {request.thread_id} to {max_response_chars} chars"
            )
            response = response[:max_response_chars] + "\n[TRUNCATED]"

        return {"response": response, "steps": steps_out}

    except Exception as e:
        logger.error(f"Exception in invoke_agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/test')
async def test_provider(request: InvokeRequest):
    """Endpoint dédié aux tests de fournisseurs
    - Honore `profile_name` si fourni (sinon profil actif par défaut)
    - Sanitize + troncature optionnelle (max_response_chars)
    - Retourne un JSON strict et concis avec métadonnées
    """
    try:
        start_ts = datetime.now()

        # Préparer le provider et charger le profil demandé
        provider = get_llm_provider()
        profile_used = None
        if request.profile_name:
            ok = provider.reload_profile(request.profile_name)
            if not ok:
                raise HTTPException(status_code=400, detail=f"Invalid or unavailable profile: {request.profile_name}")
            profile_used = request.profile_name
        else:
            # Meilleur effort pour rapporter le profil actif
            profile_used = getattr(getattr(provider, 'config', None), 'profile_name', None) or os.environ.get("LLM_PROFILE") or "default"

        # Déterminer le provider effectif (best-effort)
        provider_name = None
        cfg = getattr(provider, 'config', None)
        if cfg and isinstance(cfg, dict):
            provider_name = cfg.get('provider') or cfg.get('name')
        if not provider_name:
            env_name = os.environ.get("LLM_PROVIDER", "")
            provider_name = env_name if env_name else "unknown"

        # Construire le flux et la requête de test
        agentic_flow = get_agentic_flow()
        test_prompt = f"Test de connexion avec le fournisseur: {request.message}"

        # Extraire max_response_chars s'il est fourni
        max_chars = None
        try:
            if request.config and isinstance(request.config, dict):
                conf = request.config.get("configurable") or {}
                if isinstance(conf, dict):
                    mrc = conf.get("max_response_chars")
                    if isinstance(mrc, int) and mrc > 0:
                        max_chars = mrc
        except Exception:
            pass
        # Fallback via env
        if max_chars is None:
            try:
                env_val = os.environ.get("MCP_MAX_RESPONSE_CHARS") or os.environ.get("MAX_RESPONSE_CHARS")
                if env_val:
                    ival = int(env_val)
                    if ival > 0:
                        max_chars = ival
            except Exception:
                max_chars = None

        inputs = {
            "messages": [{"content": test_prompt, "type": "human"}],
            "thread_id": request.thread_id,
            "configurable": {
                "thread_id": request.thread_id,
                "checkpoint_ns": "test",
                "checkpoint_id": f"test-{datetime.now().isoformat()}"
            }
        }
        config = {"configurable": inputs["configurable"]}

        # Appel
        result = await agentic_flow.ainvoke(inputs, config)

        # Extraire un échantillon texte concis depuis le résultat
        def to_text(data: Any) -> str:
            try:
                if isinstance(data, str):
                    return data
                if isinstance(data, dict):
                    for key in ("response", "generated_code", "advisor_output", "scope"):
                        val = data.get(key)
                        if isinstance(val, str) and val.strip():
                            return val
                    # messages -> dernier contenu texte
                    msgs = data.get("messages")
                    if isinstance(msgs, list) and msgs:
                        last = msgs[-1]
                        if isinstance(last, dict):
                            c = last.get("content")
                            if isinstance(c, str) and c.strip():
                                return c
                # fallback generic
                return str(data)
            except Exception:
                return str(data)

        response_text = to_text(result)
        if isinstance(response_text, str):
            response_text = sanitize_output(response_text)

        truncated = False
        if isinstance(response_text, str) and max_chars is not None and len(response_text) > max_chars:
            response_text = response_text[:max_chars]
            truncated = True

        latency_ms = max(0, int((datetime.now() - start_ts).total_seconds() * 1000))

        return {
            "status": "success",
            "provider": provider_name,
            "profile_used": profile_used,
            "latency_ms": latency_ms,
            "truncated": truncated,
            "response_sample": response_text,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Exception in test_provider: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def configure_app():
    """Configure and return the FastAPI application"""
    try:
        # Add startup event
        @app.on_event("startup")
        async def startup_event():
            logger.info("🚀 Starting Archon Graph Service...")
            try:
                # Initialize any required services here
                logger.info("✅ Services initialized successfully")
            except Exception as e:
                logger.critical(f"❌ Failed to initialize services: {e}", exc_info=True)
                raise
        
        # Add shutdown event
        @app.on_event("shutdown")
        async def shutdown_event():
            logger.info("🛑 Shutting down Archon Graph Service...")
            
        return app
        
    except Exception as e:
        logger.critical(f"❌ Failed to configure application: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import uvicorn
    
    try:
        # Configure the application
        app = configure_app()
        
        # Configure Uvicorn
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8110,
            log_level="info",
            reload=True,
            reload_dirs=[str(Path(__file__).parent)],
            workers=1
        )
        
        # Create and run the server
        server = uvicorn.Server(config)
        logger.info(f"🌐 Starting server on http://{config.host}:{config.port}")
        server.run()
        
    except Exception as e:
        logger.critical(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)

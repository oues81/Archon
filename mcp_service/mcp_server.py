from fastapi import FastAPI, HTTPException, Request, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, AsyncGenerator
import re
import csv
import io
import math
from urllib.parse import urlparse
import os
import sys
import json
import logging
import time
import asyncio
from functools import wraps
from sse_starlette.sse import EventSourceResponse
import requests
import uuid
from typing import Callable
from pathlib import Path
from threading import Thread

"""In-memory CV index (prototype)"""
_cv_index_store: Dict[str, Dict[str, Any]] = {}
_cv_index_vectors: Dict[str, List[float]] = {}

def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def _load_profiles_seed() -> Optional[Dict[str, Any]]:
    try:
        path = _project_root() / 'config' / 'profils' / 'profils_postes.fr.json'
        if not path.exists():
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def _profiles_cache_file() -> Path:
    """Return the path to the RHCV profiles JSON cache file.
    
    We store cache under project data/out/ to avoid touching source folders.
    Structure: {"etag": "<sha1>", "body": <profiles_json>}
    """
    return _project_root() / 'data' / 'out' / 'rhcv_profiles_cache.json'

def _load_profiles_cache() -> Optional[Dict[str, Any]]:
    try:
        p = _profiles_cache_file()
        if not p.exists():
            return None
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def _save_profiles_cache(obj: Dict[str, Any]) -> None:
    try:
        p = _profiles_cache_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False)
    except Exception:
        # best-effort cache
        pass

def _build_rhcv_headers_for_cache(correlation_id: Optional[str] = None) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if correlation_id:
        headers["X-Correlation-Id"] = correlation_id
    api_key = os.getenv("CV_API_KEY") or os.getenv("RH_CV_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
    return headers

def _fetch_profiles_json_with_cache(api_url: Optional[str], correlation_id: Optional[str], expected_etag: Optional[str]) -> Optional[Dict[str, Any]]:
    """Fetch RHCV profiles JSON using HTTP-only ETag cache.
    
    - expected_etag should be the sha1 from /profiles/version.
    - If local cache etag matches, return cached body.
    - Else GET /profiles/json with If-None-Match; on 200, update cache; on 304, keep cache.
    """
    try:
        if not api_url:
            return None
        base = str(api_url).rstrip('/')
        url = f"{base}/profiles/json"
        cache = _load_profiles_cache()
        if cache and expected_etag and cache.get("etag") == expected_etag:
            return cache.get("body")
        headers = _build_rhcv_headers_for_cache(correlation_id)
        if expected_etag:
            headers["If-None-Match"] = expected_etag
        # Use requests (sync) here
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 200:
            body = resp.json()
            _save_profiles_cache({"etag": expected_etag or cache.get("etag") if cache else None, "body": body})
            return body
        if resp.status_code == 304 and cache:
            return cache.get("body")
        # Fallback: if unexpected code but cache exists, return cache
        if cache:
            return cache.get("body")
    except Exception:
        # On any error, attempt to use cache
        try:
            cache = _load_profiles_cache()
            if cache:
                return cache.get("body")
        except Exception:
            return None
    return None

def _extract_profile(spec: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
    # Try common shapes: dict with key profile_id, or list of dicts with id/name
    if isinstance(spec, dict) and profile_id in spec:
        return spec.get(profile_id) or {}
    if isinstance(spec, list):
        for it in spec:
            if not isinstance(it, dict):
                continue
            pid = it.get('id') or it.get('name') or it.get('title')
            if pid == profile_id:
                return it
    return {}

def _norm_tokens(s: str) -> List[str]:
    return re.findall(r"[\w\-\+/#]+", s.lower()) if s else []

def _skills_from_profile(p: Dict[str, Any]) -> List[str]:
    cands = []
    for k in ('skills', 'competences', 'compétences'):
        v = p.get(k)
        if isinstance(v, list):
            cands.extend([str(x) for x in v])
    return list({c.lower(): None for c in cands}.keys())

def _keywords_from_profile(p: Dict[str, Any]) -> List[str]:
    cands = []
    for k in ('keywords', 'mots_cles', 'mots-clés'):
        v = p.get(k)
        if isinstance(v, list):
            cands.extend([str(x) for x in v])
    return list({c.lower(): None for c in cands}.keys())

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))

def _mask_text(text: str, strategy: Dict[str, bool]) -> (str, List[Dict[str, str]]):
    ents: List[Dict[str, str]] = []
    out = text or ""
    if strategy.get('mask_email'):
        for m in re.finditer(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", out):
            orig = m.group(0)
            out = out.replace(orig, '██')
            ents.append({"type": "EMAIL", "original": orig, "masked": "██"})
    if strategy.get('mask_phone'):
        for m in re.finditer(r"\+?\d[\d\s().-]{7,}\d", out):
            orig = m.group(0)
            out = out.replace(orig, '██')
            ents.append({"type": "PHONE", "original": orig, "masked": "██"})
    # Addresses heuristic minimal (optional)
    if strategy.get('mask_addresses'):
        for m in re.finditer(r"\b\d{1,4}\s+[^\n,]{3,}\b", out):
            orig = m.group(0)
            out = out.replace(orig, '██')
            ents.append({"type": "ADDRESS", "original": orig, "masked": "██"})
    return out, ents

# Optional dependencies for crawler/RAG
try:
    from archon.archon.crawler_core import CrawlerConfig, run_crawl
    _crawler_available = True
except Exception as _e:
    logging.getLogger(__name__).warning(f"Crawler core unavailable: {_e}")
    _crawler_available = False

try:
    from supabase import create_client as _create_supabase_client
except Exception as _e:
    _create_supabase_client = None  # type: ignore
    logging.getLogger(__name__).warning(f"Supabase client unavailable: {_e}")

try:
    from openai import AsyncOpenAI as _AsyncOpenAI
except Exception as _e:
    _AsyncOpenAI = None  # type: ignore
    logging.getLogger(__name__).warning(f"OpenAI compat client unavailable: {_e}")

# Profile and provider utilities
try:
    from archon.utils.utils import (
        get_all_profiles,
        get_current_profile,
        set_current_profile,
    )
    from archon.llm import get_llm_provider
    _profiles_available = True
except Exception as _e:
    logging.getLogger(__name__).warning(f"Profile utilities unavailable: {_e}")
    _profiles_available = False

# Advisor agent availability (optional import)
try:
    from archon.archon.advisor_agent import (
        get_default_agent as _get_advisor_agent,
        AdvisorDeps as _AdvisorDeps,
    )
    _advisor_available = True
except Exception as _e:
    _advisor_available = False
    logging.getLogger(__name__).warning(f"Advisor agent unavailable: {_e}")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_server.log')
    ]
)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="Archon MCP Server",
    description="MCP Server for Archon AI Agent Builder",
    version="1.0.0"
)

# File-wide async queue for MCP SSE responses
_mcp_response_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()

# Conversation state and graph endpoint (aligns with main.py behavior)
active_threads: Dict[str, List[str]] = {}
GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://archon:8110")

# --- Local singletons for Supabase and Embeddings (used by tools) ---
_supabase_client = None
_embed_client = None
_crawl_status_map: Dict[str, Dict[str, Any]] = {}

def _ensure_env_from_profile():
    """Populate critical env vars from workbench env_vars.json active profile.
    For embedding/LLM, we intentionally override any pre-set env to enforce the active profile as single source of truth.
    Secrets are never logged.
    """
    try:
        # Candidate paths inside container and dev host mount
        candidates = [
            Path('/app/src/archon/workbench/env_vars.json'),
            Path('/app/workbench/env_vars.json'),
            Path(__file__).resolve().parent.parent / 'workbench' / 'env_vars.json',
            Path('/home/oues/archon/src/archon/workbench/env_vars.json'),
        ]
        cfg_path = next((p for p in candidates if p.exists()), None)
        if not cfg_path:
            return
        with open(cfg_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        current = data.get('current_profile')
        profiles = data.get('profiles') or {}
        prof = profiles.get(current) or {}

        def find_in_profiles(key: str) -> Optional[str]:
            # Prefer current profile, then any profile containing the key
            if isinstance(prof, dict) and key in prof and prof.get(key):
                return str(prof.get(key))
            for name, p in profiles.items():
                if isinstance(p, dict) and key in p and p.get(key):
                    return str(p.get(key))
            return None
        # Supabase
        if not os.environ.get('SUPABASE_URL'):
            val = find_in_profiles('SUPABASE_URL')
            if val:
                os.environ['SUPABASE_URL'] = val
        if not os.environ.get('SUPABASE_SERVICE_KEY'):
            val = find_in_profiles('SUPABASE_SERVICE_KEY')
            if val:
                os.environ['SUPABASE_SERVICE_KEY'] = val
        # LLM from active profile (override to avoid stale env)
        val = find_in_profiles('LLM_API_KEY')
        if val:
            if os.environ.get('LLM_API_KEY') and os.environ.get('LLM_API_KEY') != val:
                logger.info("Overriding LLM_API_KEY from active profile")
            os.environ['LLM_API_KEY'] = val
        base_url = (
            prof.get('BASE_URL') or prof.get('LLM_BASE_URL')
            or find_in_profiles('BASE_URL') or find_in_profiles('LLM_BASE_URL')
        )
        if base_url:
            if os.environ.get('LLM_BASE_URL') and os.environ.get('LLM_BASE_URL') != base_url:
                logger.info("Overriding LLM_BASE_URL from active profile")
            os.environ['LLM_BASE_URL'] = base_url
        val = find_in_profiles('PRIMARY_MODEL')
        if val:
            os.environ['PRIMARY_MODEL'] = val
        # Embeddings (Ollama OpenAI-compatible) — override to enforce profile
        emb_base = prof.get('EMBEDDING_BASE_URL') or find_in_profiles('EMBEDDING_BASE_URL')
        if emb_base:
            # Normalize Ollama endpoints to /v1 for OpenAI-compatible client
            if emb_base.endswith('/api'):
                emb_base = emb_base[:-4] + '/v1'
            elif emb_base.endswith(':11434'):
                emb_base = emb_base + '/v1'
            if os.environ.get('EMBEDDING_BASE_URL') and os.environ.get('EMBEDDING_BASE_URL') != emb_base:
                logger.info("Overriding EMBEDDING_BASE_URL from active profile")
            os.environ['EMBEDDING_BASE_URL'] = emb_base
        val = find_in_profiles('EMBEDDING_MODEL')
        if val:
            os.environ['EMBEDDING_MODEL'] = val
        val = find_in_profiles('EMBEDDING_API_KEY')
        if val:
            if os.environ.get('EMBEDDING_API_KEY') and os.environ.get('EMBEDDING_API_KEY') != val:
                logger.info("Overriding EMBEDDING_API_KEY from active profile")
            os.environ['EMBEDDING_API_KEY'] = val
        # Optional dimension hint
        if not os.environ.get('EMBEDDING_DIMENSIONS'):
            val = find_in_profiles('EMBEDDING_DIMENSIONS')
            if val:
                os.environ['EMBEDDING_DIMENSIONS'] = str(val)
    except Exception:
        # Do not crash if profile loading fails
        pass

def _get_supabase():
    global _supabase_client
    if _supabase_client is None and _create_supabase_client is not None:
        _ensure_env_from_profile()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY missing")
        _supabase_client = _create_supabase_client(url, key)
    return _supabase_client

def _clear_source_records(source_tag: str, table: str = "site_pages") -> int:
    sb = _get_supabase()
    try:
        res = sb.table(table).delete().filter("metadata->>source", "eq", source_tag).execute()
        count = getattr(res, "count", None)
        return int(count) if count is not None else 0
    except Exception as e:
        logger.warning(f"clear_source_records failed: {e}")
        return 0

async def _embed_text(text: str) -> List[float]:
    global _embed_client
    _ensure_env_from_profile()
    model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    base = os.environ.get("EMBEDDING_BASE_URL")
    api_key = os.environ.get("EMBEDDING_API_KEY", "sk-no-key")
    if _AsyncOpenAI is None or not base:
        dim = int(os.environ.get("EMBEDDING_DIMENSIONS", 768))
        return [0.0] * dim
    # Initialize client lazily and fetch embeddings
    if _embed_client is None:
        _embed_client = _AsyncOpenAI(base_url=base, api_key=api_key)
    try:
        resp = await _embed_client.embeddings.create(model=model, input=text)
        if hasattr(resp, "data") and resp.data:
            emb = resp.data[0].embedding  # type: ignore[attr-defined]
            return list(emb)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
    dim = int(os.environ.get("EMBEDDING_DIMENSIONS", 768))
    return [0.0] * dim

def _run_async(coro):
    """Run an async coroutine safely whether or not an event loop is already running.
    If a loop is running (e.g., within uvicorn), execute the coroutine in a separate thread
    using its own event loop. Otherwise, run it directly.
    """
    try:
        asyncio.get_running_loop()
        result_container: Dict[str, Any] = {}
        def _runner():
            result_container['value'] = asyncio.run(coro)
        t = Thread(target=_runner, daemon=True)
        t.start()
        t.join()
        return result_container.get('value')
    except RuntimeError:
        return asyncio.run(coro)

def _make_request(thread_id: str, user_input: str, config: dict, profile_name: Optional[str] = None) -> Dict[str, Any]:
    """Synchronous request to the graph service (mirrors main.py behavior)."""
    try:
        payload: Dict[str, Any] = {
            "message": user_input,
            "thread_id": thread_id,
            "is_first_message": not active_threads.get(thread_id, []),
            "config": config,
        }
        if profile_name is not None:
            payload["profile_name"] = profile_name

        response = requests.post(
            f"{GRAPH_SERVICE_URL}/invoke",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out for thread {thread_id}")
        raise TimeoutError("Request to graph service timed out. The operation took longer than expected.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for thread {thread_id}: {e}")
        raise

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic pour les requêtes/réponses
class MCPRequest(BaseModel):
    method: str
    jsonrpc: str = "2.0"
    params: Optional[Dict[str, Any]] = None
    id: Optional[str] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[str] = None

# Endpoint de base
@app.get("/")
async def root():
    return {"message": "Archon MCP Server is running"}

# Endpoint de santé
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "archon-mcp",
        "version": "1.0.0"
    }

@app.head("/health")
async def health_head():
    # Return minimal headers/body for HEAD probes
    return {
        "status": "ok",
        "service": "archon-mcp",
        "version": "1.0.0"
    }

# Endpoint pour lister les ressources
@app.get("/resources")
async def list_resources():
    return {
        "resources": [
            {
                "name": "archon",
                "description": "Archon AI Agent Builder",
                "version": "1.0.0"
            }
        ]
    }

# Endpoint pour les événements SSE (nécessaire pour Windsurf)
@app.get("/events")
async def event_stream():
    async def event_generator():
        try:
            # Emit immediate ready event to avoid client timeouts
            yield {
                "event": "ready",
                "data": json.dumps({"status": "ready", "timestamp": time.time()})
            }
            while True:
                # Stream any queued MCP responses first
                try:
                    item = await asyncio.wait_for(_mcp_response_queue.get(), timeout=5)
                    yield {
                        "event": "message",
                        "data": json.dumps(item)
                    }
                except asyncio.TimeoutError:
                    # Heartbeat every ~5s if no messages
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps({"status": "alive", "timestamp": time.time()})
                    }
        except asyncio.CancelledError:
            logger.info("SSE connection closed by client")
        except Exception as e:
            logger.error(f"Error in SSE stream: {str(e)}")
    
    return EventSourceResponse(event_generator())

# Alias SSE attendu par certains clients (ex: Windsurf) sur /sse
@app.head("/sse")
async def sse_head():
    # Permettre aux clients de sonder l'existence de l'endpoint SSE
    return Response(status_code=200)

@app.get("/sse")
async def sse_stream():
    async def event_generator():
        try:
            # Emit immediate ready event to avoid client timeouts
            yield {
                "event": "ready",
                "data": json.dumps({"status": "ready", "timestamp": time.time()})
            }
            while True:
                try:
                    item = await asyncio.wait_for(_mcp_response_queue.get(), timeout=5)
                    yield {
                        "event": "message",
                        "data": json.dumps(item)
                    }
                except asyncio.TimeoutError:
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps({"status": "alive", "timestamp": time.time()})
                    }
        except asyncio.CancelledError:
            logger.info("SSE connection closed by client")
        except Exception as e:
            logger.error(f"Error in SSE stream: {str(e)}")

    return EventSourceResponse(event_generator())

# --- Profile management HTTP endpoints ---
@app.get("/profiles/list")
async def http_profiles_list():
    if not _profiles_available:
        raise HTTPException(status_code=503, detail="Profile utilities unavailable")
    try:
        return {"profiles": get_all_profiles()}
    except Exception as e:
        logger.error(f"Error listing profiles: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profiles/active")
async def http_profile_active():
    if not _profiles_available:
        raise HTTPException(status_code=503, detail="Profile utilities unavailable")
    try:
        return {"active_profile": get_current_profile()}
    except Exception as e:
        logger.error(f"Error getting active profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

class _SelectBody(BaseModel):
    profile_name: str = Field(..., min_length=1)

@app.post("/profiles/select")
async def http_profile_select(body: _SelectBody):
    if not _profiles_available:
        raise HTTPException(status_code=503, detail="Profile utilities unavailable")
    try:
        provider = get_llm_provider()
        ok = provider.reload_profile(body.profile_name)
        if not ok:
            raise HTTPException(status_code=400, detail=f"Invalid or unavailable profile: {body.profile_name}")
        # Persister le profil actif pour que /profiles/active le reflète
        try:
            set_current_profile(body.profile_name)
        except Exception as _e:
            logger.warning(f"Selected profile reloaded but failed to persist active profile: {_e}")
        return {"status": "success", "active_profile": body.profile_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def _process_mcp_payload(data: Dict[str, Any]) -> Dict[str, Any]:
    # Vérification de la requête
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Invalid request format")

    # Traitement de la requête
    response: Dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": data.get("id")
    }

    method = data.get("method", "")

    if method == "initialize":
        # Advertise tool capability so clients will request tool listing
        response["result"] = {
            "capabilities": {
                "workspace": {"workspaceFolders": True},
                "textDocument": {
                    "synchronization": {"dynamicRegistration": True},
                    "completion": {"dynamicRegistration": True}
                },
                # Non-standard hints commonly used by MCP clients
                "tools": {"listChanged": True, "supportsListing": True, "supportsCalling": True},
            },
            "serverInfo": {
                "name": "Archon MCP Server",
                "version": "1.0.0"
            }
        }
    elif method == "initialized":
        response["result"] = {}
    elif method == "list_resources":
        response["result"] = {
            "resources": [
                {
                    "name": "archon",
                    "description": "Archon AI Agent Builder",
                    "version": "1.0.0"
                }
            ]
        }
    elif method == "list_profiles":
        if not _profiles_available:
            response["error"] = {"code": -32001, "message": "Profile utilities unavailable"}
        else:
            response["result"] = {"profiles": get_all_profiles()}
    elif method == "get_active_profile":
        if not _profiles_available:
            response["error"] = {"code": -32001, "message": "Profile utilities unavailable"}
        else:
            response["result"] = {"active_profile": get_current_profile()}
    elif method == "set_profile":
        if not _profiles_available:
            response["error"] = {"code": -32001, "message": "Profile utilities unavailable"}
        else:
            params = data.get("params") or {}
            pname = params.get("profile_name") if isinstance(params, dict) else None
            if not pname:
                response["error"] = {"code": -32602, "message": "Missing 'profile_name'"}
            else:
                provider = get_llm_provider()
                ok = provider.reload_profile(pname)
                if not ok:
                    response["error"] = {"code": -32002, "message": f"Invalid or unavailable profile: {pname}"}
                else:
                    response["result"] = {"status": "success", "active_profile": pname}
    # Tools discovery (support common aliases)
    elif method in {"list_tools", "listTools", "tools/list"}:
        logger.info("Handling tools list request")
        tools = [
            {
                "name": "ping",
                "description": "Health check tool that returns pong.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "set_profile",
                "description": "Set active Archon profile.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"profile_name": {"type": "string"}},
                    "required": ["profile_name"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "list_profiles",
                "description": "List available Archon profiles.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "get_active_profile",
                "description": "Get the currently active Archon profile.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "create_thread",
                "description": "Create a new Archon conversation thread and return its ID.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "advisor",
                "description": "Run the Advisor agent (generalist PM/orchestrator) with optional file context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "file_list": {"type": "array", "items": {"type": "string"}},
                        "profile_name": {"type": "string"}
                    },
                    "required": ["message"],
                    "additionalProperties": False
                }
            },
            {
                "name": "cv_score",
                "description": "Score a parsed CV or raw document (schema TBD).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_base64": {"type": "string"},
                        "filename": {"type": "string"},
                        "api_url": {"type": "string"},
                        "correlation_id": {"type": "string"},
                        "profile_name": {"type": "string"}
                    },
                    "required": [],
                    "additionalProperties": False
                }
            },
            {
                "name": "cv_anonymize",
                "description": "Anonymize PII from a CV (schema TBD).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_base64": {"type": "string"},
                        "filename": {"type": "string"},
                        "api_url": {"type": "string"},
                        "correlation_id": {"type": "string"},
                        "profile_name": {"type": "string"}
                    },
                    "required": [],
                    "additionalProperties": False
                }
            },
            {
                "name": "cv_export",
                "description": "Export parsed CV to a target format (schema TBD).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_base64": {"type": "string"},
                        "filename": {"type": "string"},
                        "format": {"type": "string"},
                        "api_url": {"type": "string"},
                        "correlation_id": {"type": "string"},
                        "profile_name": {"type": "string"}
                    },
                    "required": [],
                    "additionalProperties": False
                }
            },
            {
                "name": "cv_index",
                "description": "Index a CV/document into a store (schema TBD).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_base64": {"type": "string"},
                        "filename": {"type": "string"},
                        "api_url": {"type": "string"},
                        "correlation_id": {"type": "string"},
                        "profile_name": {"type": "string"}
                    },
                    "required": [],
                    "additionalProperties": False
                }
            },
            {
                "name": "cv_search",
                "description": "Search indexed CVs/documents (schema TBD).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "number"},
                        "api_url": {"type": "string"},
                        "correlation_id": {"type": "string"},
                        "profile_name": {"type": "string"}
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            },
            {
                "name": "cv_analyzer",
                "description": "Parse CVs via RH CV Parser API (direct call).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "files": {"type": "array", "items": {"type": "string"}},
                        "text_base64": {"type": "string"},
                        "filename": {"type": "string"},
                        "pages": {"type": ["string", "null"]},
                        "api_url": {"type": "string"},
                        "profile_name": {"type": "string"}
                    },
                    "additionalProperties": False
                }
            },
            {
                "name": "cv_parse_v2",
                "description": "Proxy to RH CV Parser API /parse_v2 (filename + file_base64).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_base64": {"type": "string"},
                        "filename": {"type": "string"},
                        "api_url": {"type": "string"},
                        "correlation_id": {"type": "string"},
                        "profile_name": {"type": "string"},
                        "strict_gating": {"type": "boolean"},
                        "min_version": {"type": "string"}
                    },
                    "required": ["file_base64", "filename"],
                    "additionalProperties": False
                }
            },
            {
                "name": "cv_health_check",
                "description": "Preflight: ping, openapi(/parse_v2), version and profiles/version checks.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "api_url": {"type": "string"},
                        "correlation_id": {"type": "string"},
                        "require_v2": {"type": "boolean"},
                        "min_version": {"type": "string"},
                        "profile_name": {"type": "string"}
                    },
                    "additionalProperties": False
                }
            },
            {
                "name": "cv_parse_sharepoint",
                "description": "Parse a CV by SharePoint reference via RHCV (POST /parse_sharepoint). Input one-of: {site_id, drive_id, item_id, pages?} or {share_url, pages?}. Optional: api_url, correlation_id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "site_id": {"type": "string"},
                        "drive_id": {"type": "string"},
                        "item_id": {"type": "string"},
                        "share_url": {"type": "string"},
                        "pages": {"type": "string"},
                        "api_url": {"type": "string"},
                        "correlation_id": {"type": "string"}
                    },
                    "additionalProperties": False
                }
            },
            {
                "name": "run_agentic_flow_archon_meta_agents_creator",
                "description": "Run the Archon Meta‑Agents Creator (generic agentic flow). Requires a general profile.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "thread_id": {"type": "string"},
                        "user_input": {"type": "string"},
                        "profile_name": {"type": "string"},
                        "config": {"type": "object"}
                    },
                    "required": ["thread_id", "user_input"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "run_agentic_flow_docs_maintainer",
                "description": "Execute the DocsMaintainer agentic flow (uses dedicated profile).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "thread_id": {"type": "string"},
                        "user_input": {"type": "string"},
                        "profile_name": {"type": "string"},
                        "config": {"type": "object"}
                    },
                    "required": ["thread_id", "user_input"],
                    "additionalProperties": False
                }
            },
            {
                "name": "run_agentic_flow_content_restructurer",
                "description": "Execute the ContentRestructurer agentic flow (uses dedicated profile).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "thread_id": {"type": "string"},
                        "user_input": {"type": "string"},
                        "profile_name": {"type": "string"},
                        "config": {"type": "object"}
                    },
                    "required": ["thread_id", "user_input"],
                    "additionalProperties": False
                }
            },
            {
                "name": "crawl_docs",
                "description": "Launch a configurable documentation crawl with SSE progress.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "config": {"type": "object"},
                        "clear_existing": {"type": "boolean"}
                    },
                    "required": [],
                    "additionalProperties": False
                }
            },
            {
                "name": "rag_query",
                "description": "Query indexed docs via vector similarity (Supabase pgvector/RPC).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer"},
                        "source_tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["query"],
                    "additionalProperties": False
                }
            },
            {
                "name": "init_site_pages",
                "description": "Check Supabase schema for site_pages and provide SQL to create table/RPCs if missing.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}
            },
            {
                "name": "site_pages_overview",
                "description": "Read-only DB overview: total rows in site_pages, distinct source tags (if supported), and a small sample.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}
            },
            {
                "name": "site_pages_stats",
                "description": "Detailed stats for site_pages: totals, distinct sources, per-source counts and latest indexed_at. Optional filter by source_tags.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source_tags": {"type": "array", "items": {"type": "string"}}
                    },
                    "additionalProperties": False
                }
            },
            {
                "name": "crawl_status",
                "description": "Get current progress/status for a running crawl by crawl_id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "crawl_id": {"type": "string"}
                    },
                    "required": ["crawl_id"],
                    "additionalProperties": False
                }
            },
            # (aliases removed from discovery to avoid confusion)
            # (moved list_profiles & get_active_profile to top)
        ]
        response["result"] = {"tools": tools, "nextCursor": None}
    # Tool invocation
    elif method in {"tools/call", "callTool", "call_tool", "call-tool"}:
        params = data.get("params") or {}
        tool_name = params.get("name") if isinstance(params, dict) else None
        tool_args = params.get("arguments") if isinstance(params, dict) else None
        logger.info(f"Handling tool call: name={tool_name}, args={tool_args}")
        # Map aliases to canonical tool names
        alias_map = {
            # Legacy dotted aliases
            "content.restructure": "run_agentic_flow_content_restructurer",
            "docs.maintain": "run_agentic_flow_docs_maintainer",
            # Legacy canonical names
            "run_agent": "run_agentic_flow_archon_meta_agents_creator",
            "run_docs_maintainer": "run_agentic_flow_docs_maintainer",
            "run_content_restructurer": "run_agentic_flow_content_restructurer",
            # (underscore aliases removed)
        }
        if tool_name in alias_map:
            tool_name = alias_map[tool_name]
        if not tool_name:
            response["error"] = {"code": -32602, "message": "Missing 'name' for tool call"}
        else:
            if tool_name == "ping":
                response["result"] = {"content": [{"type": "text", "text": "pong"}]}
            elif tool_name == "set_profile":
                if not _profiles_available:
                    response["error"] = {"code": -32001, "message": "Profile utilities unavailable"}
                else:
                    pname = tool_args.get("profile_name") if isinstance(tool_args, dict) else None
                    if not pname:
                        response["error"] = {"code": -32602, "message": "Missing 'profile_name'"}
                    else:
                        provider = get_llm_provider()
                        ok = provider.reload_profile(pname)
                        if not ok:
                            response["error"] = {"code": -32002, "message": f"Invalid or unavailable profile: {pname}"}
                        else:
                            # Persist active profile so other endpoints reflect the change
                            try:
                                set_current_profile(pname)
                            except Exception as _e:
                                logger.warning(f"Selected profile reloaded but failed to persist active profile: {_e}")
                            response["result"] = {"content": [{"type": "text", "text": f"active_profile={pname}"}]}
            elif tool_name == "create_thread":
                # Generate a UUID thread and return it
                tid = str(uuid.uuid4())
                active_threads[tid] = []
                response["result"] = {"content": [{"type": "text", "text": tid}]}
            elif tool_name == "advisor":
                # Run the Advisor agent as a generalist tool
                if not _advisor_available:
                    # Attempt lazy import now in case environment was not ready at startup
                    try:
                        from archon.archon.advisor_agent import (
                            get_default_agent as _get_advisor_agent,
                            AdvisorDeps as _AdvisorDeps,
                        )
                        avail_err = None
                    except Exception as _e:
                        avail_err = str(_e)
                        _get_advisor_agent = None  # type: ignore
                        _AdvisorDeps = None  # type: ignore
                    if _get_advisor_agent is None or _AdvisorDeps is None:
                        response["error"] = {"code": -32011, "message": f"Advisor agent unavailable: {avail_err}"}
                        return response
                else:
                    if not isinstance(tool_args, dict):
                        response["error"] = {"code": -32602, "message": "Invalid arguments for advisor"}
                    else:
                        message = tool_args.get("message")
                        if not message:
                            response["error"] = {"code": -32602, "message": "'message' is required"}
                        else:
                            file_list = tool_args.get("file_list") or []
                            profile_name = tool_args.get("profile_name")
                            # Hydrate environment from profile (optionally switch profile if provided)
                            try:
                                if profile_name and _profiles_available:
                                    provider = get_llm_provider()
                                    ok = provider.reload_profile(profile_name)
                                    if ok:
                                        try:
                                            set_current_profile(profile_name)
                                        except Exception as _e:
                                            logger.warning(f"Selected profile reloaded but failed to persist active profile: {_e}")
                                _ensure_env_from_profile()
                            except Exception:
                                pass
                            # Emit started event
                            try:
                                _mcp_response_queue.put_nowait({
                                    "jsonrpc": "2.0",
                                    "method": "tool_progress",
                                    "params": {"status": "started", "tool": tool_name, "prompt": (message or "")[:500]},
                                })
                            except Exception:
                                pass
                            import time as _time
                            _t0 = _time.time()
                            try:
                                advisor = _get_advisor_agent()
                                deps = _AdvisorDeps(file_list=list(file_list) if isinstance(file_list, list) else [])
                                result = _run_async(advisor.run(message, deps))
                                # Try common shapes, fallback to str
                                try:
                                    text_resp = (
                                        result.get("response") if isinstance(result, dict) else str(result)
                                    )
                                except Exception:
                                    text_resp = str(result)
                                response["result"] = {"content": [{"type": "text", "text": text_resp}]}
                                # Emit finished event
                                try:
                                    _dt = int((_time.time() - _t0) * 1000)
                                    _mcp_response_queue.put_nowait({
                                        "jsonrpc": "2.0",
                                        "method": "tool_progress",
                                        "params": {
                                            "status": "finished",
                                            "tool": tool_name,
                                            "duration_ms": _dt,
                                            "summary": (text_resp or "")[:400],
                                            "prompt": (message or "")[:500],
                                            "preview": (text_resp or "")[:500],
                                        },
                                    })
                                except Exception:
                                    pass
                            except Exception as e:
                                response["error"] = {"code": -32012, "message": f"advisor failed: {e}"}
                                try:
                                    _dt = int((_time.time() - _t0) * 1000)
                                    _mcp_response_queue.put_nowait({
                                        "jsonrpc": "2.0",
                                        "method": "tool_progress",
                                        "params": {"status": "failed", "tool": tool_name, "duration_ms": _dt, "error": str(e)},
                                    })
                                except Exception:
                                    pass
            elif tool_name == "cv_analyzer":
                # Parse CVs via external RH CV Parser API
                if not isinstance(tool_args, dict):
                    response["error"] = {"code": -32602, "message": "Invalid arguments for cv_analyzer"}
                else:
                    files = tool_args.get("files")
                    text_b64 = tool_args.get("text_base64")
                    filename = tool_args.get("filename")
                    pages = tool_args.get("pages")
                    api_url = tool_args.get("api_url")
                    profile_name = tool_args.get("profile_name")
                    # Validate modes
                    if (files and (text_b64 or filename)) or (not files and not (text_b64 and filename)):
                        response["error"] = {
                            "code": -32602,
                            "message": "Provide either 'files' or both 'text_base64' and 'filename'",
                        }
                    else:
                        # Hydrate env optionally
                        try:
                            if profile_name and _profiles_available:
                                provider = get_llm_provider()
                                ok = provider.reload_profile(profile_name)
                                if ok:
                                    try:
                                        set_current_profile(profile_name)
                                    except Exception as _e:
                                        logger.warning(
                                            f"Selected profile reloaded but failed to persist active profile: {_e}"
                                        )
                            _ensure_env_from_profile()
                        except Exception:
                            pass
                        # Lazy import
                        try:
                            from archon.archon.cv_analyzer import analyze as _cv_analyze
                        except Exception as _e:
                            response["error"] = {"code": -32021, "message": f"cv_analyzer unavailable: {_e}"}
                            return response
                        # Emit started
                        try:
                            _mcp_response_queue.put_nowait(
                                {
                                    "jsonrpc": "2.0",
                                    "method": "tool_progress",
                                    "params": {"status": "started", "tool": tool_name},
                                }
                            )
                        except Exception:
                            pass
                        import time as _time
                        _t0 = _time.time()
                        try:
                            result = _run_async(
                                _cv_analyze(
                                    files=files if isinstance(files, list) else None,
                                    text_base64=text_b64,
                                    filename=filename,
                                    pages=pages,
                                    api_url=api_url,
                                )
                            )
                            response["result"] = {"content": [{"type": "json", "json": result}]}
                            try:
                                _dt = int((_time.time() - _t0) * 1000)
                                _mcp_response_queue.put_nowait(
                                    {
                                        "jsonrpc": "2.0",
                                        "method": "tool_progress",
                                        "params": {
                                            "status": "finished",
                                            "tool": tool_name,
                                            "duration_ms": _dt,
                                            "summary": f"items={len(result.get('items', []))}",
                                        },
                                    }
                                )
                            except Exception:
                                pass
                        except Exception as e:
                            response["error"] = {"code": -32022, "message": f"cv_analyzer failed: {e}"}
                            try:
                                _dt = int((_time.time() - _t0) * 1000)
                                _mcp_response_queue.put_nowait(
                                    {
                                        "jsonrpc": "2.0",
                                        "method": "tool_progress",
                                        "params": {"status": "failed", "tool": tool_name, "duration_ms": _dt, "error": str(e)},
                                    }
                                )
                            except Exception:
                                pass
            elif tool_name == "cv_parse_v2":
                # Direct proxy to /parse_v2
                if not isinstance(tool_args, dict):
                    response["error"] = {"code": -32602, "message": "Invalid arguments for cv_parse_v2"}
                else:
                    file_b64 = tool_args.get("file_base64")
                    filename = tool_args.get("filename")
                    api_url = tool_args.get("api_url")
                    corr_id = tool_args.get("correlation_id")
                    profile_name = tool_args.get("profile_name")
                    strict_gating = bool(tool_args.get("strict_gating", False))
                    min_version = tool_args.get("min_version")
                    if not file_b64 or not filename:
                        response["error"] = {"code": -32602, "message": "'file_base64' and 'filename' are required"}
                    else:
                        # hydrate env if requested
                        try:
                            if profile_name and _profiles_available:
                                provider = get_llm_provider()
                                ok = provider.reload_profile(profile_name)
                                if ok:
                                    try:
                                        set_current_profile(profile_name)
                                    except Exception as _e:
                                        logger.warning(f"Failed to persist active profile: {_e}")
                            _ensure_env_from_profile()
                        except Exception:
                            pass
                        try:
                            from archon.archon.cv_analyzer import parse_v2 as _cv_parse_v2
                        except Exception as _e:
                            response["error"] = {"code": -32031, "message": f"cv_parse_v2 unavailable: {_e}"}
                            return response
                        # Optional strict gating preflight
                        if strict_gating:
                            try:
                                from archon.archon.cv_analyzer import health_check as _cv_health_check
                            except Exception as _e:
                                response["error"] = {"code": -32041, "message": f"cv_health_check unavailable: {_e}"}
                                return response
                            try:
                                gate = _run_async(_cv_health_check(
                                    api_url=api_url,
                                    correlation_id=corr_id,
                                    require_v2=True,
                                    min_version=min_version,
                                ))
                                if not (isinstance(gate, dict) and gate.get("ok")):
                                    msg = gate.get("error") if isinstance(gate, dict) else "health check failed"
                                    response["error"] = {"code": -32033, "message": f"cv_parse_v2 gated: {msg}"}
                                    return response
                            except Exception as e:
                                response["error"] = {"code": -32034, "message": f"cv_parse_v2 gating failure: {e}"}
                                return response
                        try:
                            _mcp_response_queue.put_nowait({
                                "jsonrpc": "2.0",
                                "method": "tool_progress",
                                "params": {"status": "started", "tool": tool_name},
                            })
                        except Exception:
                            pass
                        import time as _time
                        _t0 = _time.time()
                        try:
                            result = _run_async(_cv_parse_v2(
                                file_base64=file_b64,
                                filename=filename,
                                api_url=api_url,
                                correlation_id=corr_id,
                            ))
                            response["result"] = {"content": [{"type": "json", "json": result}]}
                            try:
                                _dt = int((_time.time() - _t0) * 1000)
                                _mcp_response_queue.put_nowait({
                                    "jsonrpc": "2.0",
                                    "method": "tool_progress",
                                    "params": {"status": "finished", "tool": tool_name, "duration_ms": _dt},
                                })
                            except Exception:
                                pass
                        except Exception as e:
                            response["error"] = {"code": -32032, "message": f"cv.parse_v2 failed: {e}"}
                            try:
                                _dt = int((_time.time() - _t0) * 1000)
                                _mcp_response_queue.put_nowait({
                                    "jsonrpc": "2.0",
                                    "method": "tool_progress",
                                    "params": {"status": "failed", "tool": tool_name, "duration_ms": _dt, "error": str(e)},
                                })
                            except Exception:
                                pass
            elif tool_name == "cv_health_check":
                if not isinstance(tool_args, dict):
                    response["error"] = {"code": -32602, "message": "Invalid arguments for cv_health_check"}
                else:
                    api_url = tool_args.get("api_url")
                    corr_id = tool_args.get("correlation_id")
                    require_v2 = tool_args.get("require_v2", True)
                    min_version = tool_args.get("min_version")
                    profile_name = tool_args.get("profile_name")
                    # hydrate env if requested
                    try:
                        if profile_name and _profiles_available:
                            provider = get_llm_provider()
                            ok = provider.reload_profile(profile_name)
                            if ok:
                                try:
                                    set_current_profile(profile_name)
                                except Exception as _e:
                                    logger.warning(f"Failed to persist active profile: {_e}")
                        _ensure_env_from_profile()
                    except Exception:
                        pass
                    try:
                        from archon.archon.cv_analyzer import health_check as _cv_health_check
                    except Exception as _e:
                        response["error"] = {"code": -32041, "message": f"cv_health_check unavailable: {_e}"}
                        return response
                    try:
                        _mcp_response_queue.put_nowait({
                            "jsonrpc": "2.0",
                            "method": "tool_progress",
                            "params": {"status": "started", "tool": tool_name},
                        })
                    except Exception:
                        pass
                    import time as _time
                    _t0 = _time.time()
                    try:
                        result = _run_async(_cv_health_check(
                            api_url=api_url,
                            correlation_id=corr_id,
                            require_v2=bool(require_v2),
                            min_version=min_version,
                        ))
                        response["result"] = {"content": [{"type": "json", "json": result}]}
                        try:
                            _dt = int((_time.time() - _t0) * 1000)
                            _mcp_response_queue.put_nowait({
                                "jsonrpc": "2.0",
                                "method": "tool_progress",
                                "params": {"status": "finished", "tool": tool_name, "duration_ms": _dt},
                            })
                        except Exception:
                            pass
                    except Exception as e:
                        response["error"] = {"code": -32042, "message": f"cv_health_check failed: {e}"}
                        try:
                            _dt = int((_time.time() - _t0) * 1000)
                            _mcp_response_queue.put_nowait({
                                "jsonrpc": "2.0",
                                "method": "tool_progress",
                                "params": {"status": "failed", "tool": tool_name, "duration_ms": _dt, "error": str(e)},
                            })
                        except Exception:
                            pass
            elif tool_name in ("cv_score", "cv_anonymize", "cv_export", "cv_index", "cv_search"):
                # Implemented CV tools per contracts
                if not isinstance(tool_args, dict):
                    response["error"] = {"code": -32602, "message": f"Invalid arguments for {tool_name}"}
                else:
                    profile_name = tool_args.get("profile_name")
                    api_url = tool_args.get("api_url")
                    corr_id = tool_args.get("correlation_id")
                    try:
                        if profile_name and _profiles_available:
                            provider = get_llm_provider()
                            ok = provider.reload_profile(profile_name)
                            if ok:
                                try:
                                    set_current_profile(profile_name)
                                except Exception as _e:
                                    logger.warning(f"Failed to persist active profile: {_e}")
                        _ensure_env_from_profile()
                    except Exception:
                        pass
                    try:
                        _mcp_response_queue.put_nowait({
                            "jsonrpc": "2.0",
                            "method": "tool_progress",
                            "params": {"status": "started", "tool": tool_name},
                        })
                    except Exception:
                        pass
                    import time as _time
                    _t0 = _time.time()
                    try:
                        if tool_name == "cv_score":
                            profile_id = tool_args.get("profile_id")
                            parsed = tool_args.get("parsed") or {}
                            provided_sha1 = tool_args.get("profiles_sha1")
                            # Validate profiles_sha1 via API if provided
                            api_sha1 = None
                            parser_ver = None
                            try:
                                from archon.archon.cv_analyzer import get_profiles_version as _get_prof_ver
                                from archon.archon.cv_analyzer import get_version as _get_parser_ver
                                ver = _run_async(_get_prof_ver(api_url=api_url, correlation_id=corr_id))
                                if isinstance(ver, dict) and ver.get("ok"):
                                    data = ver.get("data") or {}
                                    api_sha1 = (data.get("sha1") if isinstance(data, dict) else None) or data.get("version")
                                vinfo = _run_async(_get_parser_ver(api_url=api_url, correlation_id=corr_id))
                                if isinstance(vinfo, dict) and vinfo.get("ok"):
                                    vdata = vinfo.get("data") or {}
                                    parser_ver = (vdata.get("version") if isinstance(vdata, dict) else None)
                            except Exception:
                                pass
                            if provided_sha1 and api_sha1 and provided_sha1 != api_sha1:
                                response["error"] = {"code": 412, "message": "profiles_sha1 mismatch (precondition failed)"}
                            else:
                                # Prefer HTTP-only profiles seed fetched with ETag cache; fallback to bundled seed file
                                seed_http = _fetch_profiles_json_with_cache(api_url=api_url, correlation_id=corr_id, expected_etag=api_sha1)
                                seed = seed_http or _load_profiles_seed() or {}
                                prof = _extract_profile(seed, str(profile_id)) if profile_id else {}
                                prof_skills = _skills_from_profile(prof)
                                prof_keywords = _keywords_from_profile(prof)
                                cv_skills = [s.lower() for s in (parsed.get("skills") or []) if isinstance(s, str)]
                                summary = parsed.get("summary") or ""
                                tokens = set(_norm_tokens(summary))
                                # skills_match
                                match_count = sum(1 for s in prof_skills if s.lower() in set(cv_skills) or s.lower() in tokens)
                                denom = max(1, len(prof_skills))
                                skills_match = match_count / denom
                                # summary_keywords
                                kw_count = sum(1 for k in prof_keywords if k.lower() in tokens)
                                denom_kw = max(1, len(prof_keywords))
                                summary_keywords = kw_count / denom_kw
                                # contact_present
                                contact_present = 1.0 if ((parsed.get("email") or "") and (parsed.get("phone") or "")) else 0.0
                                breakdown = [
                                    {"criterion": "skills_match", "weight": 0.6, "score": round(skills_match, 4), "evidence": sorted(list(set(cv_skills).intersection(set(prof_skills))))},
                                    {"criterion": "summary_keywords", "weight": 0.3, "score": round(summary_keywords, 4), "evidence": [k for k in prof_keywords if k in tokens]},
                                    {"criterion": "contact_present", "weight": 0.1, "score": contact_present, "evidence": [k for k in ("email","phone") if parsed.get(k)]},
                                ]
                                final = int(round(100.0 * sum(b["weight"] * float(b["score"]) for b in breakdown)))
                                result = {
                                    "profile_id": profile_id,
                                    "score": final,
                                    "breakdown": breakdown,
                                    "version": {"profiles_sha1": provided_sha1 or api_sha1, "parser_version": parser_ver},
                                }
                                response["result"] = {"content": [{"type": "json", "json": result}]}
                        elif tool_name == "cv_anonymize":
                            parsed = tool_args.get("parsed") or {}
                            text_opt = tool_args.get("text_optional") or None
                            strategy = tool_args.get("strategy") or {"mask_name": True, "mask_email": True, "mask_phone": True, "mask_addresses": True}
                            masked = dict(parsed)
                            entities: List[Dict[str, str]] = []
                            # Mask fields
                            if strategy.get("mask_name") and masked.get("name"):
                                entities.append({"type": "NAME", "original": masked["name"], "masked": "██"})
                                masked["name"] = "██"
                            if strategy.get("mask_email") and masked.get("email"):
                                entities.append({"type": "EMAIL", "original": masked["email"], "masked": "██"})
                                masked["email"] = "██"
                            if strategy.get("mask_phone") and masked.get("phone"):
                                entities.append({"type": "PHONE", "original": masked["phone"], "masked": "██"})
                                masked["phone"] = "██"
                            # Mask summary/text
                            text_masked = None
                            if text_opt is not None:
                                text_masked, ents = _mask_text(str(text_opt), strategy)
                                entities.extend(ents)
                            else:
                                if masked.get("summary"):
                                    sm, ents = _mask_text(str(masked.get("summary")), strategy)
                                    masked["summary"] = sm
                                    entities.extend(ents)
                            result = {"parsed_masked": masked, "text_masked": text_masked, "entities": entities}
                            response["result"] = {"content": [{"type": "json", "json": result}]}
                        elif tool_name == "cv_index":
                            doc_id = tool_args.get("doc_id") or str(uuid.uuid4())
                            content = tool_args.get("content") or {}
                            meta = tool_args.get("metadata") or {}
                            upsert = bool(tool_args.get("upsert", True))
                            if not upsert and doc_id in _cv_index_store:
                                response["error"] = {"code": 409, "message": "document already exists"}
                            else:
                                text_parts: List[str] = []
                                if content.get("summary"): text_parts.append(str(content.get("summary")))
                                if content.get("skills"): text_parts.append(" ".join(map(str, content.get("skills"))))
                                if content.get("text_optional"): text_parts.append(str(content.get("text_optional")))
                                full_text = "\n".join(text_parts)
                                # Embed
                                vec = _run_async(_embed_text(full_text))
                                _cv_index_store[doc_id] = {"doc_id": doc_id, "content": content, "metadata": meta}
                                _cv_index_vectors[doc_id] = list(vec or [])
                                chunks = [{"chunk_id": f"{doc_id}#0", "tokens": min(512, len(full_text.split())), "vector_dim": len(_cv_index_vectors[doc_id])}]
                                result = {"doc_id": doc_id, "upserted": True, "chunks": chunks}
                                response["result"] = {"content": [{"type": "json", "json": result}]}
                        elif tool_name == "cv_search":
                            query = tool_args.get("query") or ""
                            top_k = int(tool_args.get("top_k") or 5)
                            filters = tool_args.get("filters") or {}
                            qvec = _run_async(_embed_text(query))
                            scored: List[Dict[str, Any]] = []
                            for did, vec in _cv_index_vectors.items():
                                score = _cosine(vec, qvec or [])
                                item = _cv_index_store.get(did) or {}
                                md = item.get("metadata") or {}
                                # Apply simple filters
                                if filters.get("profile_id") and md.get("profile_id") != filters.get("profile_id"):
                                    continue
                                if filters.get("min_score") is not None and (md.get("score") or 0) < int(filters.get("min_score")):
                                    continue
                                scored.append({"doc_id": did, "score": score, "metadata": md, "snippets": [str((item.get("content") or {}).get("summary") or "")]})
                            scored.sort(key=lambda x: x["score"], reverse=True)
                            result = {"matches": scored[:top_k]}
                            response["result"] = {"content": [{"type": "json", "json": result}]}
                        elif tool_name == "cv_export":
                            items = tool_args.get("items") or []
                            fmt = (tool_args.get("format") or "jsonl").lower()
                            dest = tool_args.get("destination") or {}
                            if fmt not in ("jsonl", "csv"):
                                response["error"] = {"code": 400, "message": "unsupported export format"}
                            else:
                                dtype = (dest.get("type") or "local").lower()
                                if dtype != "local":
                                    response["error"] = {"code": 403, "message": f"destination type '{dtype}' requires credentials"}
                                else:
                                    path = dest.get("path") or ""
                                    if not path.startswith("file://"):
                                        response["error"] = {"code": 400, "message": "local destination must start with file://"}
                                    else:
                                        local_path = path[len("file://"):]
                                        export_id = str(uuid.uuid4())
                                        count = 0
                                        if fmt == "jsonl":
                                            with open(local_path, "w", encoding="utf-8") as f:
                                                for it in items:
                                                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
                                                    count += 1
                                        else:
                                            # csv
                                            # flatten minimal fields
                                            fieldnames = ["doc_id", "name", "email", "phone", "summary", "skills", "score", "profile_id", "profiles_sha1"]
                                            with open(local_path, "w", encoding="utf-8", newline="") as f:
                                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                                writer.writeheader()
                                                for it in items:
                                                    row = {
                                                        "doc_id": it.get("doc_id"),
                                                        "name": ((it.get("parsed") or {}).get("name")),
                                                        "email": ((it.get("parsed") or {}).get("email")),
                                                        "phone": ((it.get("parsed") or {}).get("phone")),
                                                        "summary": ((it.get("parsed") or {}).get("summary")),
                                                        "skills": ",".join((it.get("parsed") or {}).get("skills") or []),
                                                        "score": it.get("score"),
                                                        "profile_id": it.get("profile_id"),
                                                        "profiles_sha1": it.get("profiles_sha1"),
                                                    }
                                                    writer.writerow(row)
                                                    count += 1
                                        result = {"export_id": export_id, "format": fmt, "count": count, "location": path}
                                        response["result"] = {"content": [{"type": "json", "json": result}]}
                        # SSE finished
                        if "result" in response and not response.get("error"):
                            try:
                                _dt = int((_time.time() - _t0) * 1000)
                                _mcp_response_queue.put_nowait({
                                    "jsonrpc": "2.0",
                                    "method": "tool_progress",
                                    "params": {"status": "finished", "tool": tool_name, "duration_ms": _dt},
                                })
                            except Exception:
                                pass
                        if "error" in response:
                            # fall through to error case below
                            pass
                    except Exception as e:
                        response["error"] = {"code": -32050, "message": f"{tool_name} failed: {e}"}
                        try:
                            _dt = int((_time.time() - _t0) * 1000)
                            _mcp_response_queue.put_nowait({
                                "jsonrpc": "2.0",
                                "method": "tool_progress",
                                "params": {"status": "failed", "tool": tool_name, "duration_ms": _dt, "error": str(e)},
                            })
                        except Exception:
                            pass
            elif tool_name in {"run_agentic_flow_archon_meta_agents_creator", "run_agent"}:
                # Validate arguments
                if not isinstance(tool_args, dict):
                    response["error"] = {"code": -32602, "message": "Invalid arguments for run_agent"}
                else:
                    tid = tool_args.get("thread_id")
                    user_input = tool_args.get("user_input")
                    profile_name = tool_args.get("profile_name")
                    user_config = tool_args.get("config") if isinstance(tool_args.get("config"), dict) else None
                    if not tid or not user_input:
                        response["error"] = {"code": -32602, "message": "'thread_id' and 'user_input' are required"}
                    else:
                        # Guardrail: refuse using run_agent for specialized flows
                        flow_req = None
                        try:
                            ucfg = user_config.get("configurable", user_config) if isinstance(user_config, dict) else None
                            if isinstance(ucfg, dict):
                                flow_req = ucfg.get("flow")
                        except Exception:
                            flow_req = None
                        if flow_req in {"ContentRestructurer", "DocsMaintainer", "ScriptsRestructurer"}:
                            response["error"] = {
                                "code": -32007,
                                "message": "Flow requires dedicated tool. Use run_content_restructurer or run_docs_maintainer instead of run_agent.",
                            }
                        else:
                            # Ensure thread bucket exists
                            if tid not in active_threads:
                                active_threads[tid] = []
                            # Enforce general profiles for run_agent (avoid specialized flows via ambient profile)
                            try:
                                effective_profile = profile_name or (get_current_profile() if _profiles_available else None)
                            except Exception:
                                effective_profile = profile_name
                            specialized_profiles = {"DocsMaintainer", "ContentRestructurer", "ScriptsRestructurer"}
                            if effective_profile in specialized_profiles:
                                response["error"] = {
                                    "code": -32008,
                                    "message": "run_agent requires a general profile (e.g., openai_default, ollama_default, openrouter_default). Use dedicated tools for DocsMaintainer/ContentRestructurer/ScriptsRestructurer or pass a general profile_name."
                                }
                            else:
                                # Merge caller config but ensure thread_id inside configurable
                                base_cfg = {"configurable": {"thread_id": tid}}
                                if user_config and isinstance(user_config, dict):
                                    # Merge shallowly; configurable merged separately if provided
                                    cfg = base_cfg.get("configurable", {}).copy()
                                    ucfg = user_config.get("configurable", user_config)
                                    if isinstance(ucfg, dict):
                                        cfg.update(ucfg)
                                    config = {"configurable": cfg}
                                else:
                                    config = base_cfg
                                # Sanitize: remove any flow leakage in run_agent
                                try:
                                    cfg_conf = (config.get("configurable", {}) or {})
                                    cfg_conf.pop("flow", None)
                                    config["configurable"] = cfg_conf
                                except Exception:
                                    pass
                                # Emit start progress event
                                try:
                                    _mcp_response_queue.put_nowait({
                                        "jsonrpc": "2.0",
                                        "method": "tool_progress",
                                        "params": {
                                            "status": "started",
                                            "tool": tool_name,
                                            "thread_id": tid,
                                            "flow": (config.get("configurable", {}) or {}).get("flow")
                                        }
                                    })
                                except Exception:
                                    pass
                                # Perform work and emit finished/failed
                                import time as _time
                                _t0 = _time.time()
                                try:
                                    logger.info(
                                        "[MCP] start",
                                        extra={
                                            "tool": tool_name,
                                            "thread_id": tid,
                                            "flow": (config.get("configurable", {}) or {}).get("flow"),
                                            "profile": profile_name or get_current_profile() if _profiles_available else profile_name,
                                            "prompt": (user_input or "")[:500],
                                            "config": {
                                                "source_dir": (config.get("configurable", {}) or {}).get("source_dir"),
                                                "dry_run": (config.get("configurable", {}) or {}).get("dry_run"),
                                                "apply_moves": (config.get("configurable", {}) or {}).get("apply_moves"),
                                                "backups_root": (config.get("configurable", {}) or {}).get("backups_root"),
                                                "filters": (config.get("configurable", {}) or {}).get("filters"),
                                            },
                                        },
                                    )
                                except Exception:
                                    pass
                                try:
                                    result = _make_request(tid, user_input, config, profile_name)
                                    active_threads[tid].append(user_input)
                                    # Relay per-step details if provided by Graph
                                    try:
                                        steps = result.get("steps") if isinstance(result, dict) else None
                                    except Exception:
                                        steps = None
                                    if isinstance(steps, list):
                                        for _st in steps:
                                            try:
                                                logger.info(
                                                    "[MCP] step",
                                                    extra={
                                                        "tool": tool_name,
                                                        "thread_id": tid,
                                                        "flow": (config.get("configurable", {}) or {}).get("flow"),
                                                        "profile": profile_name or get_current_profile() if _profiles_available else profile_name,
                                                        "agent": (_st or {}).get("agent"),
                                                        "model": (_st or {}).get("model"),
                                                        "status": (_st or {}).get("status"),
                                                        "duration_ms": (_st or {}).get("duration_ms"),
                                                        "node": (_st or {}).get("node"),
                                                    },
                                                )
                                            except Exception:
                                                pass
                                            try:
                                                _mcp_response_queue.put_nowait({
                                                    "jsonrpc": "2.0",
                                                    "method": "tool_progress",
                                                    "params": {
                                                        "status": "step",
                                                        "tool": tool_name,
                                                        "thread_id": tid,
                                                        "flow": (config.get("configurable", {}) or {}).get("flow"),
                                                        "step": _st,
                                                    }
                                                })
                                            except Exception:
                                                pass
                                    # Expecting {'response': str}
                                    text_resp = result.get("response") if isinstance(result, dict) else str(result)
                                    response["result"] = {"content": [{"type": "text", "text": text_resp}]}
                                    try:
                                        _dt = int(( _time.time() - _t0) * 1000)
                                        _preview = (text_resp or "")[:500]
                                        logger.info(
                                            "[MCP] finished",
                                            extra={
                                                "tool": tool_name,
                                                "thread_id": tid,
                                                "flow": (config.get("configurable", {}) or {}).get("flow"),
                                                "profile": profile_name or get_current_profile() if _profiles_available else profile_name,
                                                "duration_ms": _dt,
                                                "summary": (text_resp or "")[:240],
                                                "prompt": (user_input or "")[:500],
                                                "preview": _preview,
                                                "config": {
                                                    "source_dir": (config.get("configurable", {}) or {}).get("source_dir"),
                                                    "dry_run": (config.get("configurable", {}) or {}).get("dry_run"),
                                                    "apply_moves": (config.get("configurable", {}) or {}).get("apply_moves"),
                                                    "backups_root": (config.get("configurable", {}) or {}).get("backups_root"),
                                                    "filters": (config.get("configurable", {}) or {}).get("filters"),
                                                },
                                            },
                                        )
                                    except Exception:
                                        pass
                                    try:
                                        _mcp_response_queue.put_nowait({
                                            "jsonrpc": "2.0",
                                            "method": "tool_progress",
                                            "params": {
                                                "status": "finished",
                                                "tool": tool_name,
                                                "thread_id": tid,
                                                "flow": (config.get("configurable", {}) or {}).get("flow"),
                                                "summary": text_resp[:400],
                                                "prompt": (user_input or "")[:500],
                                                "preview": (text_resp or "")[:500],
                                                "config": {
                                                    "source_dir": (config.get("configurable", {}) or {}).get("source_dir"),
                                                    "dry_run": (config.get("configurable", {}) or {}).get("dry_run"),
                                                    "apply_moves": (config.get("configurable", {}) or {}).get("apply_moves"),
                                                    "backups_root": (config.get("configurable", {}) or {}).get("backups_root"),
                                                    "filters": (config.get("configurable", {}) or {}).get("filters"),
                                                }
                                            }
                                        })
                                    except Exception:
                                        pass
                                except Exception as e:
                                    response["error"] = {"code": -32003, "message": f"run_agent failed: {e}"}
                                    try:
                                        _dt = int(( _time.time() - _t0) * 1000)
                                        logger.info(
                                            "[MCP] failed",
                                            extra={
                                                "tool": tool_name,
                                                "thread_id": tid,
                                                "flow": (config.get("configurable", {}) or {}).get("flow"),
                                                "profile": profile_name or get_current_profile() if _profiles_available else profile_name,
                                                "duration_ms": _dt,
                                                "error": str(e),
                                            },
                                        )
                                    except Exception:
                                        pass
                                    try:
                                        _mcp_response_queue.put_nowait({
                                            "jsonrpc": "2.0",
                                            "method": "tool_progress",
                                            "params": {
                                                "status": "failed",
                                                "tool": tool_name,
                                                "thread_id": tid,
                                                "flow": (config.get("configurable", {}) or {}).get("flow"),
                                                "error": str(e)
                                            }
                                        })
                                    except Exception:
                                        pass
            elif tool_name in {"run_agentic_flow_docs_maintainer", "run_docs_maintainer"}:
                # Dedicated flow: DocsMaintainer
                if not isinstance(tool_args, dict):
                    response["error"] = {"code": -32602, "message": "Invalid arguments for run_docs_maintainer"}
                else:
                    tid = tool_args.get("thread_id") or str(uuid.uuid4())
                    user_input = tool_args.get("user_input") or ""
                    profile_name = tool_args.get("profile_name") or "DocsMaintainer"
                    user_config = tool_args.get("config") if isinstance(tool_args.get("config"), dict) else None
                    if tid not in active_threads:
                        active_threads[tid] = []
                    # Build config with enforced flow
                    base_cfg = {"configurable": {"thread_id": tid, "flow": "DocsMaintainer"}}
                    if user_config and isinstance(user_config, dict):
                        cfg = base_cfg.get("configurable", {}).copy()
                        ucfg = user_config.get("configurable", user_config)
                        if isinstance(ucfg, dict):
                            cfg.update(ucfg)
                        config = {"configurable": cfg}
                    else:
                        config = base_cfg
                    # Progress events
                    import time as _time
                    _t0 = _time.time()
                    try:
                        logger.info(
                            "[MCP] start",
                            extra={
                                "tool": tool_name,
                                "thread_id": tid,
                                "flow": "DocsMaintainer",
                                "profile": profile_name,
                                "prompt": (user_input or "")[:500],
                                "config": {
                                    "source_dir": (config.get("configurable", {}) or {}).get("source_dir"),
                                    "dry_run": (config.get("configurable", {}) or {}).get("dry_run"),
                                    "apply_moves": (config.get("configurable", {}) or {}).get("apply_moves"),
                                    "backups_root": (config.get("configurable", {}) or {}).get("backups_root"),
                                    "filters": (config.get("configurable", {}) or {}).get("filters"),
                                },
                            },
                        )
                    except Exception:
                        pass
                    try:
                        _mcp_response_queue.put_nowait({
                            "jsonrpc": "2.0",
                            "method": "tool_progress",
                            "params": {
                                "status": "started",
                                "tool": tool_name,
                                "thread_id": tid,
                                "flow": "DocsMaintainer",
                                "prompt": (user_input or "")[:500],
                                "config": {
                                    "source_dir": (config.get("configurable", {}) or {}).get("source_dir"),
                                    "dry_run": (config.get("configurable", {}) or {}).get("dry_run"),
                                    "apply_moves": (config.get("configurable", {}) or {}).get("apply_moves"),
                                    "backups_root": (config.get("configurable", {}) or {}).get("backups_root"),
                                    "filters": (config.get("configurable", {}) or {}).get("filters"),
                                }
                            }
                        })
                    except Exception:
                        pass
                    try:
                        result = _make_request(tid, user_input, config, profile_name)
                        active_threads[tid].append(user_input)
                        # Relay per-step details if provided by Graph
                        try:
                            steps = result.get("steps") if isinstance(result, dict) else None
                        except Exception:
                            steps = None
                        if isinstance(steps, list):
                            for _st in steps:
                                try:
                                    logger.info(
                                        "[MCP] step",
                                        extra={
                                            "tool": tool_name,
                                            "thread_id": tid,
                                            "flow": "DocsMaintainer",
                                            "profile": profile_name,
                                            "agent": (_st or {}).get("agent"),
                                            "model": (_st or {}).get("model"),
                                            "status": (_st or {}).get("status"),
                                            "duration_ms": (_st or {}).get("duration_ms"),
                                            "node": (_st or {}).get("node"),
                                        },
                                    )
                                except Exception:
                                    pass
                                try:
                                    _mcp_response_queue.put_nowait({
                                        "jsonrpc": "2.0",
                                        "method": "tool_progress",
                                        "params": {"status": "step", "tool": tool_name, "thread_id": tid, "flow": "DocsMaintainer", "step": _st}
                                    })
                                except Exception:
                                    pass
                        text_resp = result.get("response") if isinstance(result, dict) else str(result)
                        response["result"] = {"content": [{"type": "text", "text": text_resp}]}
                        try:
                            _dt = int(( _time.time() - _t0) * 1000)
                            _preview = (text_resp or "")[:500]
                            logger.info(
                                "[MCP] finished",
                                extra={
                                    "tool": tool_name,
                                    "thread_id": tid,
                                    "flow": "DocsMaintainer",
                                    "profile": profile_name,
                                    "duration_ms": _dt,
                                    "summary": (text_resp or "")[:240],
                                    "prompt": (user_input or "")[:500],
                                    "preview": _preview,
                                    "config": {
                                        "source_dir": (config.get("configurable", {}) or {}).get("source_dir"),
                                        "dry_run": (config.get("configurable", {}) or {}).get("dry_run"),
                                        "apply_moves": (config.get("configurable", {}) or {}).get("apply_moves"),
                                        "backups_root": (config.get("configurable", {}) or {}).get("backups_root"),
                                        "filters": (config.get("configurable", {}) or {}).get("filters"),
                                    },
                                },
                            )
                        except Exception:
                            pass
                        try:
                            _mcp_response_queue.put_nowait({
                                "jsonrpc": "2.0",
                                "method": "tool_progress",
                                "params": {
                                    "status": "finished",
                                    "tool": tool_name,
                                    "thread_id": tid,
                                    "flow": "DocsMaintainer",
                                    "summary": text_resp[:400],
                                    "prompt": (user_input or "")[:500],
                                    "preview": (text_resp or "")[:500],
                                    "config": {
                                        "source_dir": (config.get("configurable", {}) or {}).get("source_dir"),
                                        "dry_run": (config.get("configurable", {}) or {}).get("dry_run"),
                                        "apply_moves": (config.get("configurable", {}) or {}).get("apply_moves"),
                                        "backups_root": (config.get("configurable", {}) or {}).get("backups_root"),
                                        "filters": (config.get("configurable", {}) or {}).get("filters"),
                                    }
                                }
                            })
                        except Exception:
                            pass
                    except Exception as e:
                        response["error"] = {"code": -32009, "message": f"run_docs_maintainer failed: {e}"}
                        try:
                            _dt = int(( _time.time() - _t0) * 1000)
                            logger.info(
                                "[MCP] failed",
                                extra={
                                    "tool": tool_name,
                                    "thread_id": tid,
                                    "flow": "DocsMaintainer",
                                    "profile": profile_name,
                                    "duration_ms": _dt,
                                    "error": str(e),
                                },
                            )
                        except Exception:
                            pass
                        try:
                            _mcp_response_queue.put_nowait({
                                "jsonrpc": "2.0",
                                "method": "tool_progress",
                                "params": {"status": "failed", "tool": tool_name, "thread_id": tid, "flow": "DocsMaintainer", "error": str(e)}
                            })
                        except Exception:
                            pass
            elif tool_name in {"run_agentic_flow_content_restructurer", "run_content_restructurer"}:
                # Dedicated flow: ContentRestructurer
                if not isinstance(tool_args, dict):
                    response["error"] = {"code": -32602, "message": "Invalid arguments for run_content_restructurer"}
                else:
                    tid = tool_args.get("thread_id") or str(uuid.uuid4())
                    user_input = tool_args.get("user_input") or ""
                    profile_name = tool_args.get("profile_name") or "ContentRestructurer"
                    user_config = tool_args.get("config") if isinstance(tool_args.get("config"), dict) else None
                    if tid not in active_threads:
                        active_threads[tid] = []
                    # Build config with enforced flow
                    base_cfg = {"configurable": {"thread_id": tid, "flow": "ContentRestructurer"}}
                    if user_config and isinstance(user_config, dict):
                        cfg = base_cfg.get("configurable", {}).copy()
                        ucfg = user_config.get("configurable", user_config)
                        if isinstance(ucfg, dict):
                            cfg.update(ucfg)
                        config = {"configurable": cfg}
                    else:
                        config = base_cfg
                    # Progress events
                    try:
                        _mcp_response_queue.put_nowait({
                            "jsonrpc": "2.0",
                            "method": "tool_progress",
                            "params": {
                                "status": "started",
                                "tool": tool_name,
                                "thread_id": tid,
                                "flow": "ContentRestructurer",
                                "prompt": (user_input or "")[:500],
                                "config": {
                                    "source_dir": (config.get("configurable", {}) or {}).get("source_dir"),
                                    "dry_run": (config.get("configurable", {}) or {}).get("dry_run"),
                                    "apply_moves": (config.get("configurable", {}) or {}).get("apply_moves"),
                                    "backups_root": (config.get("configurable", {}) or {}).get("backups_root"),
                                    "filters": (config.get("configurable", {}) or {}).get("filters"),
                                }
                            }
                        })
                    except Exception:
                        pass
                    try:
                        result = _make_request(tid, user_input, config, profile_name)
                        active_threads[tid].append(user_input)
                        text_resp = result.get("response") if isinstance(result, dict) else str(result)
                        response["result"] = {"content": [{"type": "text", "text": text_resp}]}
                        try:
                            _dt = int(( _time.time() - _t0) * 1000)
                            _preview = (text_resp or "")[:500]
                            logger.info(
                                "[MCP] finished",
                                extra={
                                    "tool": tool_name,
                                    "thread_id": tid,
                                    "flow": "ContentRestructurer",
                                    "profile": profile_name,
                                    "duration_ms": _dt,
                                    "summary": (text_resp or "")[:240],
                                    "prompt": (user_input or "")[:500],
                                    "preview": _preview,
                                    "config": {
                                        "source_dir": (config.get("configurable", {}) or {}).get("source_dir"),
                                        "dry_run": (config.get("configurable", {}) or {}).get("dry_run"),
                                        "apply_moves": (config.get("configurable", {}) or {}).get("apply_moves"),
                                        "backups_root": (config.get("configurable", {}) or {}).get("backups_root"),
                                        "filters": (config.get("configurable", {}) or {}).get("filters"),
                                    },
                                },
                            )
                        except Exception:
                            pass
                        try:
                            _mcp_response_queue.put_nowait({
                                "jsonrpc": "2.0",
                                "method": "tool_progress",
                                "params": {
                                    "status": "finished",
                                    "tool": tool_name,
                                    "thread_id": tid,
                                    "flow": "ContentRestructurer",
                                    "summary": text_resp[:400],
                                    "prompt": (user_input or "")[:500],
                                    "preview": (text_resp or "")[:500],
                                    "config": {
                                        "source_dir": (config.get("configurable", {}) or {}).get("source_dir"),
                                        "dry_run": (config.get("configurable", {}) or {}).get("dry_run"),
                                        "apply_moves": (config.get("configurable", {}) or {}).get("apply_moves"),
                                        "backups_root": (config.get("configurable", {}) or {}).get("backups_root"),
                                        "filters": (config.get("configurable", {}) or {}).get("filters"),
                                    }
                                }
                            })
                        except Exception:
                            pass
                    except Exception as e:
                        response["error"] = {"code": -32010, "message": f"run_content_restructurer failed: {e}"}
                        try:
                            _dt = int(( _time.time() - _t0) * 1000)
                            logger.info(
                                "[MCP] failed",
                                extra={
                                    "tool": tool_name,
                                    "thread_id": tid,
                                    "flow": "ContentRestructurer",
                                    "profile": profile_name,
                                    "duration_ms": _dt,
                                    "error": str(e),
                                },
                            )
                        except Exception:
                            pass
                        try:
                            _mcp_response_queue.put_nowait({
                                "jsonrpc": "2.0",
                                "method": "tool_progress",
                                "params": {"status": "failed", "tool": tool_name, "thread_id": tid, "flow": "ContentRestructurer", "error": str(e)}
                            })
                        except Exception:
                            pass
            elif tool_name == "crawl_docs":
                # Launch a configurable crawl using crawler_core with SSE progress
                if not _crawler_available:
                    response["error"] = {"code": -32010, "message": "crawler_core unavailable"}
                else:
                    try:
                        args = data["params"].get("arguments", {}) if data.get("params") else {}
                        cfg_dict = args.get("config")
                        # Ensure environment is hydrated from active profile before building config/clients
                        try:
                            _ensure_env_from_profile()
                        except Exception:
                            pass
                        profile_cfg = None
                        if cfg_dict is None:
                            # Provide a conservative default config to allow wrapper calls without explicit config
                            # This will perform a no-op crawl if no seeds/sitemaps are provided by environment.
                            cfg_dict = {
                                "name": "default",
                                "source_tag": "default",
                                "seeds": [],
                                "sitemaps": [],
                                "allow_domains": [],
                                "table": "site_pages",
                                "max_pages": 0,
                            }
                        # Apply sane defaults from active profile env if caller omitted them
                        try:
                            # Do not mutate original dict
                            _cfg_dict = dict(cfg_dict)
                            # Remove transient keys not part of CrawlerConfig schema
                            if "store_embeddings" in _cfg_dict:
                                _cfg_dict.pop("store_embeddings", None)
                            # Embedding model
                            if not _cfg_dict.get("embedding_model"):
                                env_emb_model = os.environ.get("EMBEDDING_MODEL")
                                if env_emb_model:
                                    _cfg_dict["embedding_model"] = env_emb_model
                            # Embedding base URL (normalize /api -> /v1, and ensure /v1 suffix)
                            if not _cfg_dict.get("embedding_provider_base_url"):
                                env_emb_base = os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("EMBEDDING_PROVIDER_BASE_URL")
                                if env_emb_base:
                                    base = env_emb_base
                                    if base.endswith("/api"):
                                        base = base[:-4] + "/v1"
                                    if not base.rstrip("/").endswith("/v1"):
                                        base = base + ("v1" if base.endswith("/") else "/v1")
                                    _cfg_dict["embedding_provider_base_url"] = base
                            # LLM model for enrichment
                            if not _cfg_dict.get("llm_model"):
                                env_llm_model = os.environ.get("PRIMARY_MODEL")
                                if env_llm_model:
                                    _cfg_dict["llm_model"] = env_llm_model
                            cfg = CrawlerConfig(**_cfg_dict)
                        except Exception:
                            # Fallback to original if any mapping failed
                            cfg = CrawlerConfig(**cfg_dict)
                        # Fail-fast validation when enrichment is strictly required
                        try:
                            require_enrich = bool(getattr(cfg, "require_llm_enrichment", False))
                            llm_titles = bool(getattr(cfg, "llm_title_summary", False))
                        except Exception:
                            require_enrich = False
                            llm_titles = False
                        if require_enrich or llm_titles:
                            # Check presence of LLM credentials and model/endpoint
                            llm_key = os.environ.get("LLM_API_KEY")
                            llm_base = os.environ.get("LLM_BASE_URL") or os.environ.get("BASE_URL")
                            llm_model = getattr(cfg, "llm_model", None) or os.environ.get("PRIMARY_MODEL")
                            if require_enrich and (not llm_key or not llm_base or not llm_model):
                                missing = []
                                if not llm_key:
                                    missing.append("LLM_API_KEY")
                                if not llm_base:
                                    missing.append("LLM_BASE_URL/BASE_URL")
                                if not llm_model:
                                    missing.append("llm_model/PRIMARY_MODEL")
                                response["error"] = {
                                    "code": -32011,
                                    "message": f"crawl_docs failed: LLM enrichment required but missing config: {', '.join(missing)}"
                                }
                                return response
                        # Fail-fast embedding validation when caller explicitly requests storing embeddings
                        try:
                            store_embeddings = bool((cfg_dict or {}).get("store_embeddings", False))
                        except Exception:
                            store_embeddings = False
                        if store_embeddings:
                            emb_base = os.environ.get("EMBEDDING_BASE_URL") or os.environ.get("EMBEDDING_PROVIDER_BASE_URL")
                            emb_model = getattr(cfg, "embedding_model", None) or os.environ.get("EMBEDDING_MODEL")
                            emb_key = os.environ.get("EMBEDDING_API_KEY")
                            missing_emb = []
                            if not emb_base:
                                missing_emb.append("EMBEDDING_BASE_URL/EMBEDDING_PROVIDER_BASE_URL")
                            if not emb_model:
                                missing_emb.append("EMBEDDING_MODEL (or embedding_model in config)")
                            # Only require API key for known hosted providers (OpenAI/OpenRouter); local routers often don't need it
                            try:
                                _check_base = (emb_base or "").lower()
                            except Exception:
                                _check_base = ""
                            if ("openai" in _check_base or "openrouter" in _check_base) and not emb_key:
                                missing_emb.append("EMBEDDING_API_KEY")
                            if missing_emb:
                                response["error"] = {
                                    "code": -32014,
                                    "message": f"crawl_docs failed: store_embeddings=True but missing embedding config: {', '.join(missing_emb)}"
                                }
                                return response
                        clear_existing = bool(args.get("clear_existing", False))
                        if clear_existing:
                            try:
                                deleted = _clear_source_records(cfg.source_tag, cfg.table)
                                try:
                                    _mcp_response_queue.put_nowait({
                                        "jsonrpc": "2.0",
                                        "method": "tool_progress",
                                        "params": {"status": "running", "tool": tool_name, "message": f"Cleared {deleted} records for source={cfg.source_tag}"}
                                    })
                                except Exception:
                                    pass
                            except Exception as e:
                                logger.warning(f"clear_existing failed: {e}")
                        # Progress callback → SSE
                        _crawl_id_holder: Dict[str, Any] = {"id": None}
                        def _progress_cb(ev: Dict[str, Any]):
                            try:
                                _mcp_response_queue.put_nowait({
                                    "jsonrpc": "2.0",
                                    "method": "tool_progress",
                                    "params": {"status": "running", "tool": tool_name, "progress": ev},
                                })
                            except Exception:
                                pass
                            # Update in-memory status map
                            try:
                                cid = _crawl_id_holder.get("id")
                                if cid:
                                    st = _crawl_status_map.setdefault(cid, {})
                                    st["last_event"] = ev
                                    st["updated_at"] = time.time()
                            except Exception:
                                pass
                        # Final assurance: hydrate env again just before client init in crawler_core
                        try:
                            _ensure_env_from_profile()
                        except Exception:
                            pass
                        crawl_id = run_crawl(cfg, progress_cb=_progress_cb)
                        # Initialize status entry
                        try:
                            _crawl_id_holder["id"] = crawl_id
                            _crawl_status_map[crawl_id] = {
                                "status": "started",
                                "name": getattr(cfg, "name", None),
                                "source_tag": getattr(cfg, "source_tag", None),
                                "started_at": time.time(),
                            }
                        except Exception:
                            pass
                        # Normalize output to standard MCP content wrapper
                        response["result"] = {
                            "content": [
                                {"type": "text", "text": json.dumps({"status": "started", "crawl_id": crawl_id})}
                            ]
                        }
                    except Exception as e:
                        response["error"] = {"code": -32011, "message": f"crawl_docs failed: {e}"}

            elif tool_name == "rag_query":
                # Dense vector search via Supabase RPC (expects existing DB function)
                try:
                    args = data["params"].get("arguments", {}) if data.get("params") else {}
                    query = args.get("query")
                    if not query or not isinstance(query, str):
                        raise ValueError("query is required")
                    top_k = int(args.get("top_k", 8))
                    source_tags = args.get("source_tags") or []
                    if source_tags and not isinstance(source_tags, list):
                        raise ValueError("source_tags must be a list of strings")
                    qvec = _run_async(_embed_text(query))
                    sb = _get_supabase()
                    # Prefer a multi-source RPC if available
                    rows = None
                    try:
                        rows = sb.rpc("match_site_pages_multi", {"query_embedding": qvec, "match_count": top_k, "source_tags": source_tags}).execute().data
                        # If multi returns empty but tags were provided, try per-tag single RPCs
                        if (not rows) and source_tags:
                            agg: List[Dict[str, Any]] = []
                            for tag in source_tags:
                                try:
                                    datares = sb.rpc("match_site_pages", {"query_embedding": qvec, "match_count": top_k, "source_tag": tag}).execute().data
                                    if datares:
                                        agg.extend(datares)
                                except Exception:
                                    continue
                            rows = agg
                    except Exception:
                        # Fallback to single-source RPC (call per source)
                        if source_tags:
                            agg: List[Dict[str, Any]] = []
                            for tag in source_tags:
                                try:
                                    datares = sb.rpc("match_site_pages", {"query_embedding": qvec, "match_count": top_k, "source_tag": tag}).execute().data
                                    if datares:
                                        agg.extend(datares)
                                except Exception:
                                    continue
                            rows = agg
                        else:
                            # Last resort: try single without tag if DB function supports it
                            rows = sb.rpc("match_site_pages", {"query_embedding": qvec, "match_count": top_k}).execute().data
                    if rows is None:
                        raise RuntimeError("No RPC available. Please create match_site_pages(_multi) in DB.")
                    # Normalize output
                    results: List[Dict[str, Any]] = []
                    for r in rows:
                        results.append({
                            "url": r.get("url"),
                            "title": r.get("title"),
                            "summary": r.get("summary"),
                            "content": r.get("content"),
                            "metadata": r.get("metadata"),
                            "similarity": r.get("similarity")
                        })
                    response["result"] = {"content": [{"type": "text", "text": json.dumps(results)}]}
                except Exception as e:
                    response["error"] = {"code": -32012, "message": f"rag_query failed: {e}"}
            elif tool_name == "init_site_pages":
                # Check if site_pages exists; if not, return SQL and instructions to create it
                try:
                    sb = _get_supabase()
                    exists = True
                    try:
                        # Try a lightweight select to detect table presence
                        _ = sb.table("site_pages").select("id", count="exact").limit(1).execute()
                    except Exception:
                        exists = False
                    if exists:
                        response["result"] = {"content": [{"type": "text", "text": "site_pages exists"}]}
                    else:
                        sql_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils", "site_pages.sql")
                        sql_text = None
                        try:
                            with open(sql_path, "r", encoding="utf-8") as f:
                                sql_text = f.read()
                        except Exception as _e:
                            sql_text = "-- SQL file not found in container. Please apply site_pages.sql from repo: archon/utils/site_pages.sql"
                        instructions = (
                            "Apply this SQL in your Supabase project (SQL editor or psql). "
                            "It creates the site_pages table, indexes, and RPCs (match_site_pages)."
                        )
                        response["result"] = {
                            "content": [
                                {"type": "text", "text": "site_pages missing"},
                                {"type": "text", "text": instructions},
                                {"type": "text", "text": sql_text}
                            ]
                        }
                except Exception as e:
                    response["error"] = {"code": -32013, "message": f"init_site_pages failed: {e}"}

            elif tool_name == "site_pages_overview":
                # Lightweight overview: total rows, distinct source tags (from sample), and a small sample
                try:
                    sb = _get_supabase()
                    total = None
                    try:
                        res = sb.table("site_pages").select("id", count="exact").limit(1).execute()
                        # Supabase-py attaches count on response
                        total = getattr(res, "count", None)
                    except Exception:
                        total = None
                    # Sample for distinct sources
                    try:
                        rows = sb.table("site_pages").select("url,title,summary,metadata").limit(50).execute().data
                    except Exception:
                        rows = []
                    sources = []
                    try:
                        seen = set()
                        for r in rows or []:
                            src = ((r or {}).get("metadata") or {}).get("source")
                            if src and src not in seen:
                                seen.add(src)
                                sources.append(src)
                    except Exception:
                        sources = []
                    sample = []
                    for r in (rows or [])[:5]:
                        sample.append({
                            "url": r.get("url"),
                            "title": r.get("title"),
                            "summary": r.get("summary"),
                            "source": ((r or {}).get("metadata") or {}).get("source")
                        })
                    out = {"total_rows": total, "distinct_sources_sample": sources, "sample": sample}
                    response["result"] = {"content": [{"type": "text", "text": json.dumps(out)}]}
                except Exception as e:
                    response["error"] = {"code": -32014, "message": f"site_pages_overview failed: {e}"}

            elif tool_name == "site_pages_stats":
                # Detailed stats with optional filtering by source_tags
                try:
                    args = data.get("params", {}).get("arguments", {}) if data.get("params") else {}
                    source_tags = args.get("source_tags") or []
                    top_n_domains = int(args.get("top_n_domains", 10))
                    if source_tags and not isinstance(source_tags, list):
                        raise ValueError("source_tags must be an array of strings")
                    sb = _get_supabase()
                    # Total rows
                    total = None
                    try:
                        res = sb.table("site_pages").select("id", count="exact").limit(1).execute()
                        total = getattr(res, "count", None)
                    except Exception:
                        total = None
                    # Fetch a reasonable page to compute per-source stats (avoid heavy scans)
                    try:
                        # If filter is provided, rough client-side filter afterwards
                        data_rows = sb.table("site_pages").select("url,metadata").limit(1000).execute().data
                    except Exception:
                        data_rows = []
                    stats: Dict[str, Dict[str, Any]] = {}
                    # temp holders for unique urls and domain counts
                    per_source_urls: Dict[str, set] = {}
                    per_source_domains: Dict[str, Dict[str, int]] = {}
                    for r in data_rows or []:
                        meta = (r or {}).get("metadata") or {}
                        src = meta.get("source") or "unknown"
                        if source_tags and src not in source_tags:
                            continue
                        s = stats.setdefault(src, {"chunks": 0, "latest_indexed_at": None})
                        s["chunks"] += 1
                        ts = meta.get("indexed_at")
                        if ts:
                            try:
                                if s["latest_indexed_at"] is None or str(ts) > str(s["latest_indexed_at"]):
                                    s["latest_indexed_at"] = ts
                            except Exception:
                                pass
                        # accumulate unique urls and domain counts
                        try:
                            url = (r or {}).get("url")
                            if url:
                                per_source_urls.setdefault(src, set()).add(url)
                                host = urlparse(url).netloc or ""
                                if host:
                                    per_source_domains.setdefault(src, {}).setdefault(host, 0)
                                    per_source_domains[src][host] += 1
                        except Exception:
                            pass

                    # finalize pages/domains counts and top domains
                    for src, s in stats.items():
                        urls_set = per_source_urls.get(src, set())
                        domain_counts = per_source_domains.get(src, {})
                        s["pages"] = len(urls_set)
                        s["domains_total"] = len(domain_counts)
                        # sorted top domains
                        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
                        s["top_domains"] = [
                            {"domain": d, "chunks": c} for d, c in sorted_domains[:max(0, top_n_domains)]
                        ]
                    out = {"total_rows": total, "sources": stats, "filtered": bool(source_tags)}
                    response["result"] = {"content": [{"type": "text", "text": json.dumps(out)}]}
                except Exception as e:
                    response["error"] = {"code": -32015, "message": f"site_pages_stats failed: {e}"}

            elif tool_name == "crawl_status":
                try:
                    args = data.get("params", {}).get("arguments", {}) if data.get("params") else {}
                    cid = args.get("crawl_id")
                    if not cid or not isinstance(cid, str):
                        raise ValueError("crawl_id is required")
                    status_obj = _crawl_status_map.get(cid) or {"status": "unknown", "crawl_id": cid}
                    out = dict(status_obj)
                    out["crawl_id"] = cid
                    response["result"] = {"content": [{"type": "text", "text": json.dumps(out)}]}
                except Exception as e:
                    response["error"] = {"code": -32016, "message": f"crawl_status failed: {e}"}

            elif tool_name == "list_profiles":
                if not _profiles_available:
                    response["error"] = {"code": -32001, "message": "Profile utilities unavailable"}
                else:
                    try:
                        profiles = get_all_profiles()
                        response["result"] = {"content": [{"type": "text", "text": json.dumps(profiles)}]}
                    except Exception as e:
                        response["error"] = {"code": -32004, "message": f"list_profiles failed: {e}"}
            elif tool_name == "get_active_profile":
                if not _profiles_available:
                    response["error"] = {"code": -32001, "message": "Profile utilities unavailable"}
                else:
                    try:
                        ap = get_current_profile()
                        response["result"] = {"content": [{"type": "text", "text": ap or ""}]}
                    except Exception as e:
                        response["error"] = {"code": -32005, "message": f"get_active_profile failed: {e}"}
            else:
                response["result"] = {"status": "success", "method": method}

    return response

# Endpoint pour la communication MCP (legacy direct HTTP)
@app.post("/mcp")
async def handle_mcp(request: Request):
    try:
        data = await request.json()
        logger.info(f"Received MCP request: {json.dumps(data, indent=2)}")
        return _process_mcp_payload(data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling MCP request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# MCP SSE messages endpoint: client sends JSON-RPC here; responses are emitted on SSE
@app.post("/messages")
async def mcp_messages(request: Request):
    try:
        raw = await request.body()
        if not raw:
            raise ValueError("Empty request body")
        # Decode to text and normalize
        try:
            text = raw.decode("utf-8", errors="strict")
        except Exception:
            # Fallback with replacement to avoid hard crash
            text = raw.decode("utf-8", errors="replace")
        text = text.lstrip("\ufeff").strip()
        logger.debug(f"/messages raw body: {text}")
        try:
            data = json.loads(text)
        except Exception as je:
            logger.error(f"Invalid JSON body (text): {text!r} error={je}")
            raise
        logger.info(f"Received MCP message: {json.dumps(data, indent=2)}")
        resp = _process_mcp_payload(data)
        # Enqueue for SSE stream consumers
        await _mcp_response_queue.put(resp)
        # Also return the JSON-RPC response inline to satisfy clients expecting immediate body with id
        return JSONResponse(content=resp)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling MCP message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Compatibility: some clients POST JSON-RPC to /sse instead of /messages
@app.post("/sse")
async def sse_post(request: Request):
    # Delegate to the same handler as /messages
    return await mcp_messages(request)

# Si le fichier est exécuté directement
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting MCP server on http://0.0.0.0:8100")
    uvicorn.run(
        "mcp_server:app",
        host="0.0.0.0",
        port=8100,
        reload=True,
        log_level="info",
        # Important pour le support SSE
        proxy_headers=True,
        forwarded_allow_ips='*'
    )

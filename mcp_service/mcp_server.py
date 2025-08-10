from fastapi import FastAPI, HTTPException, Request, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, AsyncGenerator
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

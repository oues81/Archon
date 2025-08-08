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

# Profile and provider utilities
try:
    from archon.utils.utils import get_all_profiles, get_current_profile
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
            while True:
                # Envoyer un événement de pulsation toutes les 30 secondes
                yield {
                    "event": "heartbeat",
                    "data": json.dumps({"status": "alive", "timestamp": time.time()})
                }
                await asyncio.sleep(30)
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
        return {"status": "success", "active_profile": body.profile_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint pour la communication MCP
@app.post("/mcp")
async def handle_mcp(request: Request):
    try:
        data = await request.json()
        logger.info(f"Received MCP request: {json.dumps(data, indent=2)}")
        
        # Vérification de la requête
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Invalid request format")
        
        # Traitement de la requête
        response = {
            "jsonrpc": "2.0",
            "id": data.get("id")
        }
        
        method = data.get("method", "")
        
        if method == "initialize":
            response["result"] = {
                "capabilities": {
                    "workspace": {"workspaceFolders": True},
                    "textDocument": {
                        "synchronization": {"dynamicRegistration": True},
                        "completion": {"dynamicRegistration": True}
                    }
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
        else:
            response["result"] = {"status": "success", "method": method}
        
        return response
        
    except Exception as e:
        logger.error(f"Error handling MCP request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

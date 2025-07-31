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

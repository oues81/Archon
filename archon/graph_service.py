import traceback
import sys
import os
import json
import logging
import asyncio
import dataclasses
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)



# Appliquer le correctif pour TypedDict
try:
    from patch_typing import *
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Optional, Dict, Any, List

# Import du router des profils
try:
    from api.profiles import router as profiles_router
    profiles_available = True
except ImportError:
    profiles_router = None
    profiles_available = False
    logging.warning("Module api.profiles non disponible")

from archon.archon_graph import get_agentic_flow
from archon.utils.utils import write_to_log
from archon.archon.llm import LLMProvider

try:
    from langgraph.types import Command
except ImportError:
    # Fallback implementation if langgraph.types.Command is not available
    class Command:
        """Dummy implementation of Command for compatibility"""
        pass
    
app = FastAPI()

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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}    

@app.post("/invoke")
async def invoke_agent(request: InvokeRequest):
    """
    Process a message through the agentic flow. This endpoint can dynamically
    apply a configuration profile for the duration of the request.
    """
    try:
        # Initialize LLM provider with the requested profile or current profile
        provider = LLMProvider(profile_name=request.profile_name)
        if not provider.config:
            # Get the actual profile name for error message
            from archon.utils.utils import get_current_profile
            actual_profile = request.profile_name or get_current_profile()
            raise ValueError(f"Failed to load configuration for profile '{actual_profile}'")
        # Utiliser dataclasses.asdict() au lieu de .dict() qui n'existe pas
        llm_config_dict = dataclasses.asdict(provider.config)
        
        # Fix: Map the configuration keys to match what archon_graph.py expects (UPPERCASE)
        llm_config = {
            'LLM_PROVIDER': llm_config_dict.get('provider'),
            'REASONER_MODEL': llm_config_dict.get('reasoner_model'),
            'PRIMARY_MODEL': llm_config_dict.get('primary_model'),
            'CODER_MODEL': llm_config_dict.get('coder_model'),
            'ADVISOR_MODEL': llm_config_dict.get('advisor_model'),
            'OLLAMA_BASE_URL': llm_config_dict.get('base_url') if llm_config_dict.get('provider') == 'ollama' else None
        }
        
        # Fix: Add the API key with the expected name for compatibility with archon_graph.py
        if provider.config.api_key:
            llm_config['LLM_API_KEY'] = provider.config.api_key
            if provider.config.provider == 'openrouter':
                llm_config['OPENROUTER_API_KEY'] = provider.config.api_key
            elif provider.config.provider == 'openai':
                llm_config['OPENAI_API_KEY'] = provider.config.api_key
    except Exception as e:
        logger.error(f"Error loading profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load profile {request.profile_name}: {str(e)}")

    # Prepare configuration for the agentic flow
    config = {
        "configurable": {
            "thread_id": request.thread_id,
            "llm_config": llm_config
        }
    }

    logger.info(f"Starting invoke_agent for thread {request.thread_id} with profile {request.profile_name or 'default'}")
    
    response = ""
    final_state = None
    
    # This is the original, correct logic for handling the agent state
    if request.is_first_message:
        user_message = {
            "kind": "request",
            "parts": [{
                "part_kind": "user-prompt",
                "content": request.message,
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        message_bytes = json.dumps([user_message]).encode('utf-8')
        initial_state = {
            "latest_user_message": request.message,
            "next_user_message": "",
            "messages": [message_bytes],
            "scope": "", "advisor_output": "", "file_list": [],
            "refined_prompt": "", "refined_tools": "", "refined_agent": ""
        }
        input_for_flow = initial_state
    else:
        # For subsequent messages, we might only need to pass the new message
        input_for_flow = {"next_user_message": request.message}

    try:
        flow = get_agentic_flow()
        async for state_update in flow.astream(input_for_flow, config, stream_mode="values"):
            final_state = state_update

        if final_state:
            if final_state.get('generated_code'):
                result_obj = final_state['generated_code']
                response = result_obj.data if hasattr(result_obj, 'data') and result_obj.data else str(result_obj)
            elif final_state.get('advisor_output'):
                response = final_state['advisor_output']
            elif final_state.get('scope'):
                response = final_state['scope']
            else:
                logger.warning("No generated content found in final state.")
        
        return {"response": response}

    except Exception as e:
        logger.error(f"Exception in invoke_agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
@app.post('/test')
async def test_provider(request: InvokeRequest):
    """Endpoint dédié aux tests de fournisseurs"""
    try:
        # Logique de test simplifiee
        test_prompt = f"Test de connexion avec le fournisseur: {request.message}"
        
        # Configuration complète pour le graphe
        inputs = {
            "messages": [{"content": test_prompt, "type": "human"}],
            "thread_id": request.thread_id,
            "configurable": {
                "thread_id": request.thread_id,
                "checkpoint_ns": "test",
                "checkpoint_id": f"test-{datetime.now().isoformat()}"
            }
        }
        
        # Exécution du test
        result = await agentic_flow.ainvoke(inputs)
        
        # Sérialisation correcte
        return {
            "status": "success",
            "provider": "openrouter" if "openrouter" in os.environ.get("LLM_PROVIDER", "").lower() else "ollama",
            "response": str(result)
        }
    except Exception as e:
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

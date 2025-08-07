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
from archon.llm import get_llm_provider

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
            'OLLAMA_BASE_URL': provider.config.base_url
        }

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
        flow = get_agentic_flow()
        logger.info(f"🚨 DEBUG: About to start flow.astream with input: {input_for_flow}")
        logger.info(f"🚨 DEBUG: Config: {config}")
        
        iteration_count = 0
        async for state_update in flow.astream(input_for_flow, config, stream_mode="values"):
            iteration_count += 1
            logger.info(f"🚨 DEBUG: Iteration {iteration_count}, state_update: {state_update}")
            final_state = state_update
        
        logger.info(f"🚨 DEBUG: Flow completed after {iteration_count} iterations")

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

        return {"response": response}

    except Exception as e:
        logger.error(f"Exception in invoke_agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/test')
async def test_provider(request: InvokeRequest):
    """Endpoint dédié aux tests de fournisseurs"""
    try:
        # Récupérer le graphe agentique
        agentic_flow = get_agentic_flow()

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
        
        # La configuration doit être passée comme un deuxième argument positionnel
        config = {"configurable": inputs["configurable"]}
        
        # Exécution du test
        result = await agentic_flow.ainvoke(inputs, config)
        
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

"""
Archon Graph - Agent Workflow Management
Utilizes a unified LLM provider to support multiple backends (Ollama, OpenAI, OpenRouter)
"""
# -*- coding: utf-8 -*-
import os
# Set UTF-8 encoding environment variables
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('LC_ALL', 'C.UTF-8')
os.environ.setdefault('LANG', 'C.UTF-8')

import asyncio
import json
import logging
import os
import sys
import time
import threading
import atexit
import httpx
import logfire
from typing import Dict, Any, Optional, Union, List, Tuple
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class LoggingHTTPClient(httpx.AsyncClient):
    """HTTP client with request/response logging"""
    
    async def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
        # Log request
        start_time = time.time()
        request_id = str(id(request))
        
        # Log request details (excluding sensitive headers)
        headers = dict(request.headers)
        safe_headers = {
            k: (v if k.lower() not in ['authorization', 'api-key'] else '***REDACTED***')
            for k, v in headers.items()
        }
        
        try:
            # Log request
            logger.debug(
                f"🔵 HTTP Request [{request_id}]: {request.method} {request.url}\n"
                f"Headers: {json.dumps(safe_headers, indent=2)}\n"
                f"Content type: {headers.get('content-type', 'N/A')}"
            )
            
            # Send the request
            response = await super().send(request, **kwargs)
            
            # Calculate duration
            duration = (time.time() - start_time) * 1000  # in ms
            
            # Log response
            logger.debug(
                f"🟢 HTTP Response [{request_id}]: {response.status_code} in {duration:.2f}ms\n"
                f"URL: {response.url}\n"
                f"Headers: {json.dumps(dict(response.headers), indent=2)}"
            )
            
            return response
            
        except Exception as e:
            # Log error
            duration = (time.time() - start_time) * 1000  # in ms
            logger.error(
                f"🔴 HTTP Error [{request_id}] after {duration:.2f}ms: {str(e)}\n"
                f"URL: {getattr(request, 'url', 'N/A')}\n"
                f"Method: {getattr(request, 'method', 'N/A')}"
            )
            raise

try:
    from api.profiles import router as profiles_router
except ImportError:
    # Fallback si le module n'est pas disponible
    profiles_router = None
from datetime import datetime
from typing import Annotated, List, Any, Optional, Dict, Union
from typing_extensions import TypedDict

# Simple OpenRouter Configuration
logging.info("🔧 Simple OpenRouter Configuration")

# Logger Configuration
logger = logging.getLogger(__name__)

# Configure Logfire
try:
    # Simplified Logfire configuration without unsupported options
    logfire.configure(service_name="archon")
    logger.info("✅ Logfire configured successfully")
except Exception as e:
    # In case of error, continue execution without Logfire
    logger.warning(f"⚠️ Unable to configure Logfire: {e}")
    # If Logfire is not configured correctly, disable logging to avoid warnings
    os.environ.setdefault("LOGFIRE_DISABLE", "1")

# Agent Definitions
# These are initialized later based on the loaded provider
reasoner_agent: Optional['AIAgent'] = None
advisor: Optional['AIAgent'] = None
coder: Optional['AIAgent'] = None

# Unified LLM Provider Import
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'archon'))
from llm import LLMProvider, LLMConfig

# Pydantic AI Compatibility Imports
from pydantic_ai import RunContext, Agent as PydanticAgent, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings
import httpx

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    SystemPromptPart,
    UserPromptPart
)



def get_llm_instance(provider: str, model_name: str, config: Dict[str, Any]):
    """Creates and returns an LLM instance based on the provided configuration."""
    provider = (provider or "openrouter").lower()
    logger.info(f"Configuring LLM instance for provider: {provider} with model: {model_name}")

    try:
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIModel

        http_client = LoggingHTTPClient(timeout=60.0, limits=httpx.Limits(max_connections=100, max_keepalive_connections=20))
        client_kwargs = {"http_client": http_client}

        if provider == "ollama":
            client_kwargs.update({
                "api_key": "ollama",
                "base_url": config.get("OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL"),
            })
        elif provider == "openrouter":
            api_key = config.get("LLM_API_KEY") or config.get("OPENROUTER_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY or LLM_API_KEY not found in profile configuration or environment")
            client_kwargs.update({
                "api_key": api_key,
                "base_url": "https://openrouter.ai/api/v1",
                "default_headers": {
                    "HTTP-Referer": config.get("OPENROUTER_REFERRER", os.getenv("OPENROUTER_REFERRER", "http://localhost:8110")),
                    "X-Title": config.get("OPENROUTER_X_TITLE", os.getenv("OPENROUTER_X_TITLE", "Archon"))
                }
            })
        elif provider == "openai":
            api_key = config.get("LLM_API_KEY") or config.get("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY or LLM_API_KEY not found in profile configuration or environment")
            client_kwargs.update({
                "api_key": api_key,
                "base_url": "https://api.openai.com/v1"
            })
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Ensure the http_client is always added to the list for cleanup
        http_clients.append(http_client)
        logger.debug(f"✅ Registered HTTP client for cleanup: {http_client}")

        # Create the client and model instances
        openai_client = AsyncOpenAI(**client_kwargs)
        model = OpenAIModel(model_name=model_name, openai_client=openai_client)
        
        logger.info(f"✅ Successfully initialized LLM for {provider} with model {model_name}")
        return model

    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM for {provider}: {str(e)}", exc_info=True)
        raise


# LangGraph Imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.state import START
    from langgraph.checkpoint.memory import MemorySaver
except ImportError as e:
    logger.warning(f"Could not import LangGraph: {e}")
    class StateGraph:
        def __init__(self, *args, **kwargs): pass
    END = None
    START = None
    MemorySaver = object

# Prompt Imports
try:
    from archon.agent_prompts import (
        prompt_refiner_agent_prompt, advisor_prompt, coder_prompt_with_examples, reasoner_prompt
    )
except ImportError:
    # Fallback si le module n'est pas disponible
    prompt_refiner_agent_prompt = ""
    advisor_prompt = ""
    coder_prompt_with_examples = ""
    reasoner_prompt = ""

# Log models on startup


# Agent State Definition
class AgentState(TypedDict):
    latest_user_message: str
    next_user_message: str
    messages: List[Any]
    scope: str
    advisor_output: str
    file_list: List[str]
    refined_prompt: str
    refined_tools: str
    refined_agent: str
    generated_code: Optional[str] = None
    error: Optional[str] = None

def ensure_event_loop():
    """Ensures that an asyncio event loop is available in the current thread"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in this thread, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def run_async_in_sync(coro):
    """Execute a coroutine synchronously while ensuring the existence of an event loop"""
    loop = ensure_event_loop()
    return loop.run_until_complete(coro)

def define_scope_with_reasoner(state: AgentState, config: dict) -> AgentState:
    """Defines the project scope using a reasoner agent based on the active profile."""
    logger.info("---STEP: Defining scope with reasoner agent---")
    try:
        llm_config = config.get("configurable", {}).get("llm_config", {})
        model_name = llm_config.get("REASONER_MODEL")
        provider = llm_config.get("LLM_PROVIDER")
        logger.info(f"🧠 REASONER - Provider: {provider} | Model: {model_name}")
        logger.info(f"🧠 REASONER - User message: {state['latest_user_message']}")

        reasoner = PydanticAgent(
            get_llm_instance(provider, model_name, llm_config),
            system_prompt=reasoner_prompt
        )

        logger.info("🔍 REASONER - Sending request...")
        result = run_async_in_sync(reasoner.run(state['latest_user_message']))
        scope_text = result.data if hasattr(result, 'data') else str(result)

        logger.info(f"🔍 REASONER - Complete response received: {scope_text[:200]}...")
        state['scope'] = scope_text
        return state
        
    except Exception as e:
        error_msg = f"Error in define_scope_with_reasoner: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state['error'] = error_msg
        state['scope'] = f"Error: {error_msg}"
        return state

def advisor_with_examples(state: AgentState, config: dict) -> AgentState:
    """Generates advice and examples using the advisor agent based on the active profile."""
    logger.info("---STEP: Generating advice with advisor agent---")
    try:
        llm_config = config.get("configurable", {}).get("llm_config", {})
        model_name = llm_config.get("PRIMARY_MODEL")
        provider = llm_config.get("LLM_PROVIDER")
        logger.info(f"💡 ADVISOR - Provider: {provider} | Model: {model_name}")

        advisor = PydanticAgent(
            get_llm_instance(provider, model_name, llm_config),
            system_prompt=advisor_prompt
        )
        
        logger.info("💡 ADVISOR - Sending request...")
        result = run_async_in_sync(advisor.run(state['scope']))
        advisor_text = result.data if hasattr(result, 'data') else str(result)

        logger.info(f"💡 ADVISOR - Complete response received: {advisor_text[:200]}...")
        state['advisor_output'] = advisor_text
        return state

    except Exception as e:
        error_msg = f"Error in advisor_with_examples: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state['error'] = error_msg
        state['advisor_output'] = f"Error: {error_msg}"
        return state

def coder_agent(state: AgentState, config: dict) -> AgentState:
    """Generates the final code using the coder agent based on the active profile."""
    logger.info("---STEP: Generating code with coder agent---")
    try:
        llm_config = config.get("configurable", {}).get("llm_config", {})
        model_name = llm_config.get("CODER_MODEL")
        provider = llm_config.get("LLM_PROVIDER")
        logger.info(f"⚡ CODER - Provider: {provider} | Model: {model_name}")

        scope = state.get('scope', '')
        advisor_output = state.get('advisor_output', '')
        logger.info(f"⚡ CODER - Scope: {scope[:200]}...")
        logger.info(f"⚡ CODER - Advisor Output: {advisor_output[:200]}...")

        coder = PydanticAgent(
            get_llm_instance(provider, model_name, llm_config),
            system_prompt=coder_prompt_with_examples
        )
        
        logger.info("⚡ CODER - Sending request...")
        instruction = f"Scope: {scope}\n\nAdvisor Output: {advisor_output}"
        result = run_async_in_sync(coder.run(instruction))
        code_text = result.data if hasattr(result, 'data') else str(result)

        logger.info(f"⚡ CODER - Complete response received: {code_text[:200]}...")
        state['generated_code'] = code_text
        return state
        
    except Exception as e:
        error_msg = f"Error in coder_agent: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state['error'] = error_msg
        state['generated_code'] = f"Error: {error_msg}"
        return state

# Global instances and state
llm_instance = None
http_clients = []

async def cleanup_http_clients():
    """Clean up all HTTP clients on application exit"""
    if not http_clients:
        return
        
    logger.info("🧹 Cleaning up HTTP clients...")
    
    for client in http_clients[:]:  # Create a copy of the list
        try:
            if client and not client.is_closed:
                await client.aclose()
                logger.debug(f"✅ Closed HTTP client: {client}")
            http_clients.remove(client)  # Remove from the list after closing
        except Exception as e:
            logger.error(f"❌ Error closing HTTP client: {e}", exc_info=True)

# Register cleanup on exit
def cleanup():
    """Synchronous cleanup wrapper"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(cleanup_http_clients())
        else:
            loop.run_until_complete(cleanup_http_clients())
    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)

atexit.register(cleanup)

agentic_flow = None

def get_agentic_flow():
    global agentic_flow
    if agentic_flow is None:
        # Initialize the StateGraph and build the workflow
        builder = StateGraph(AgentState)
        builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
        builder.add_node("advisor_with_examples", advisor_with_examples)
        builder.add_node("coder_agent", coder_agent)
        builder.set_entry_point("define_scope_with_reasoner")
        builder.add_edge("define_scope_with_reasoner", "advisor_with_examples")
        builder.add_edge("advisor_with_examples", "coder_agent")
        builder.add_edge("coder_agent", END)
        memory = MemorySaver()
        agentic_flow = builder.compile(checkpointer=memory)
    return agentic_flow

if __name__ == '__main__':
    try:
        initial_state = {
            'latest_user_message': 'Hello, can you help me create an AI agent?',
            'next_user_message': '',
            'messages': [],
            'scope': '',
            'advisor_output': '',
            'file_list': [],
            'refined_prompt': '',
            'refined_tools': '',
            'refined_agent': ''
        }
        
        print("Starting agentic flow execution...")
        result = agentic_flow.invoke(initial_state)
        
        print("\nExecution Result:")
        print(f"- Latest Message: {result.get('latest_user_message', '')}")
        print(f"- Scope Defined: {bool(result.get('scope', ''))}")
        print(f"- Code Generated: {bool(result.get('generated_code', ''))}")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {str(e)}")
        print(f"[ERROR] Type: {type(e).__name__}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")

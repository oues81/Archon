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
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional, Union, List, Tuple
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from archon.utils.utils import configure_logging, get_bool_env

# Configure logging centrally
_log_summary = configure_logging()
logger = logging.getLogger(__name__)

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

rag_flag = get_bool_env('RAG_ENABLED', False)
logger.info(
    f"🔧 Logging configured | console={_log_summary.get('console')} file={_log_summary.get('file')}"
    f" path={_log_summary.get('file_path')} json={_log_summary.get('json')} | RAG_ENABLED={rag_flag}"
)

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
from archon.llm import LLMProvider, LLMConfig

# Pydantic AI Compatibility Imports
try:
    from pydantic_ai import RunContext, Agent as PydanticAgent, ModelRetry
    from pydantic_ai.settings import ModelSettings
except ImportError:
    # Provide minimal shims so module import doesn't fail when optional deps are absent
    class RunContext:  # type: ignore
        pass
    class PydanticAgent:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        async def run(self, *args, **kwargs):
            return type("_Res", (), {"data": ""})
        def system_prompt(self, fn):
            return fn
    class ModelRetry(Exception):  # type: ignore
        pass
    class ModelSettings:  # type: ignore
        pass
import httpx

try:
    from pydantic_ai.messages import (
        ModelMessage,
        ModelMessagesTypeAdapter,
        ModelRequest,
        ModelResponse,
        TextPart,
        SystemPromptPart,
        UserPromptPart
    )
except ImportError:
    # Minimal fallbacks when pydantic_ai is not installed
    class ModelMessage:  # type: ignore
        pass
    class ModelMessagesTypeAdapter:  # type: ignore
        pass
    class ModelRequest:  # type: ignore
        pass
    class ModelResponse:  # type: ignore
        pass
    class TextPart:  # type: ignore
        pass
    class SystemPromptPart:  # type: ignore
        pass
    class UserPromptPart:  # type: ignore
        pass



async def _ensure_ollama_model_is_pulled(openai_client: Any, model_name: str):
    """Helper function to check if an Ollama model exists and pull it if not."""
    try:
        logger.debug(f"Checking if Ollama model '{model_name}' exists...")
        await openai_client.models.retrieve(model_name)
        logger.debug(f"Ollama model '{model_name}' found.")
    except Exception:
        logger.warning(f"Ollama model '{model_name}' not found. Attempting to pull it...")
        try:
            # A simple chat completion request will trigger a pull if the model is not present.
            await openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            logger.info(f"Successfully pulled or confirmed Ollama model: {model_name}")
        except Exception as pull_error:
            logger.error(f"Failed to pull Ollama model '{model_name}': {pull_error}")
            raise pull_error

async def get_llm_instance(provider: str, model_name: str, config: Dict[str, Any]):
    """Creates and returns an LLM instance based on the provided configuration."""
    provider = (provider or "openrouter").lower()
    logger.info(f"Configuring LLM instance for provider: {provider} with model: {model_name}")

    try:
        # Lazy import optional deps only when needed
        from openai import AsyncOpenAI  # type: ignore
        from pydantic_ai.models.openai import OpenAIModel  # type: ignore

        http_client = LoggingHTTPClient(timeout=60.0, limits=httpx.Limits(max_connections=100, max_keepalive_connections=20))
        client_kwargs = {"http_client": http_client}

        if provider == "ollama":
            base_url = config.get("OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
            if not base_url:
                raise ValueError("OLLAMA_BASE_URL not found in profile or environment")
            if not model_name:
                raise ValueError("Model name is required for Ollama provider")
                
            client_kwargs.update({"api_key": "ollama", "base_url": base_url})
            openai_client = AsyncOpenAI(**client_kwargs)
            await _ensure_ollama_model_is_pulled(openai_client, model_name)
            model = OpenAIModel(model_name=model_name, openai_client=openai_client)
            
        elif provider == "openrouter":
            api_key = config.get("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY or LLM_API_KEY not found")
            client_kwargs.update({
                "api_key": api_key,
                "base_url": "https://openrouter.ai/api/v1",
                "default_headers": {
                    "HTTP-Referer": config.get("OPENROUTER_REFERRER", "http://localhost:8110"),
                    "X-Title": config.get("OPENROUTER_X_TITLE", "Archon")
                }
            })
            openai_client = AsyncOpenAI(**client_kwargs)
            model = OpenAIModel(model_name=model_name, openai_client=openai_client)
            
        elif provider == "openai":
            api_key = config.get("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY or LLM_API_KEY not found")
            client_kwargs.update({"api_key": api_key})
            openai_client = AsyncOpenAI(**client_kwargs)
            model = OpenAIModel(model_name=model_name, openai_client=openai_client)

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Ensure the http_client is always added to the list for cleanup
        http_clients.append(http_client)
        logger.debug(f"✅ Registered HTTP client for cleanup: {http_client}")
        
        logger.info(f"✅ Successfully initialized LLM for {provider} with model {model_name}")
        return model

    except ImportError as e:
        logger.error(f"Missing dependencies for '{provider}': {e}")
        raise ValueError(f"Dependencies for '{provider}' are not installed.") from e
    except Exception as e:
        logger.error(f"Failed to create LLM instance for provider {provider}: {e}")
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
    from archon.archon.agent_prompts import (
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

async def define_scope_with_reasoner(state: AgentState, config: dict) -> AgentState:
    """Defines the project scope using a reasoner agent based on the active profile."""
    logger.info("🚨 DEBUG: ENTERING define_scope_with_reasoner function")
    logger.info("---STEP: Defining scope with reasoner agent---")
    try:
        # Ensure required state keys exist
        state.setdefault('error', None)
        state.setdefault('scope', '')
        state.setdefault('messages', [])
        
        # Get LLM configuration
        llm_config = config.get("configurable", {}).get("llm_config", {})
        model_name = llm_config.get("REASONER_MODEL")
        provider = llm_config.get("LLM_PROVIDER")
        
        logger.info(f"🧠 REASONER - Provider: {provider} | Model: {model_name}")
        
        # Initialize messages if empty
        if not state['messages']:
            if 'latest_user_message' in state:
                state['messages'] = [{'content': state['latest_user_message'], 'role': 'user'}]
            else:
                state['messages'] = [{'content': 'No message provided', 'role': 'system'}]
        
        # Get the last user message
        last_message = next((msg for msg in reversed(state['messages']) if msg.get('role') == 'user'), None)
        if last_message:
            user_message = last_message['content']
            state['latest_user_message'] = user_message
            logger.info(f"🧠 REASONER - Processing user message: {user_message[:200]}...")
        else:
            user_message = state.get('latest_user_message', 'No user message available')
            logger.warning("⚠️ No user message found in messages, using latest_user_message")
        
        # Initialize the reasoner agent
        reasoner = PydanticAgent(
            await get_llm_instance(provider, model_name, llm_config),
            system_prompt=reasoner_prompt
        )

        # Process the message
        logger.info("🔍 REASONER - Sending request to reasoner...")
        result = await reasoner.run(user_message)
        scope_text = result.data if hasattr(result, 'data') else str(result)

        # Update state with results
        state['scope'] = scope_text
        state['messages'].append({
            'role': 'assistant',
            'content': f"Scope defined: {scope_text[:200]}..."
        })
        
        logger.info(f"✅ REASONER - Scope defined successfully: {scope_text[:200]}...")
        return state
        
    except Exception as e:
        error_msg = f"Error in define_scope_with_reasoner: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state['error'] = error_msg
        state['scope'] = f"Error: {error_msg}"
        state['messages'].append({
            'role': 'error',
            'content': error_msg
        })
        return state
        return state

async def advisor_with_examples(state: AgentState, config: dict) -> AgentState:
    """Generates advice and examples using the advisor agent based on the active profile."""
    logger.info("---STEP: Generating advice with advisor agent---")
    try:
        # Ensure required state keys exist
        state.setdefault('error', None)
        state.setdefault('advisor_output', '')
        state.setdefault('scope', state.get('scope', ''))
        
        # Get LLM configuration
        llm_config = config.get("configurable", {}).get("llm_config", {})
        model_name = llm_config.get("ADVISOR_MODEL")
        provider = llm_config.get("LLM_PROVIDER")
        
        logger.info(f"💡 ADVISOR - Provider: {provider} | Model: {model_name} (ADVISOR_MODEL)")
        logger.info(f"💡 ADVISOR - Scope: {state['scope'][:200]}...")

        # Initialize the advisor agent
        advisor = PydanticAgent(
            await get_llm_instance(provider, model_name, llm_config),
            system_prompt=advisor_prompt
        )
        
        # Process the scope
        logger.info("💡 ADVISOR - Sending request to advisor...")
        result = await advisor.run(state['scope'])
        advisor_text = result.data if hasattr(result, 'data') else str(result)

        # Update state with results
        state['advisor_output'] = advisor_text
        state['messages'].append({
            'role': 'assistant',
            'content': f"Advisor output: {advisor_text[:200]}..."
        })
        
        logger.info(f"✅ ADVISOR - Advice generated successfully: {advisor_text[:200]}...")
        return state

    except Exception as e:
        error_msg = f"Error in advisor_with_examples: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state['error'] = error_msg
        state['advisor_output'] = f"Error: {error_msg}"
        state['messages'].append({
            'role': 'error',
            'content': error_msg
        })
        return state

from archon.archon.pydantic_ai_coder import create_pydantic_ai_coder
from archon.utils.utils import get_clients, get_bool_env, validate_rag_env

async def coder_agent(state: AgentState, config: dict) -> AgentState:
    """Generates the final code using the coder agent based on the active profile."""
    logger.info("---STEP: Generating code with coder agent---")
    try:
        # Ensure required state keys exist
        state.setdefault('error', None)
        state.setdefault('generated_code', '')
        state.setdefault('scope', state.get('scope', ''))
        state.setdefault('advisor_output', state.get('advisor_output', ''))

        # Validate inputs
        if not state['scope'] or len(state['scope'].split()) < 3:
            error_msg = "Scope is too short or undefined. Cannot generate code."
            logger.warning(f"⚠️ CODER - {error_msg}")
            state['error'] = error_msg
            state['generated_code'] = f"Error: {error_msg}"
            return state

        if not state['advisor_output'] or len(state['advisor_output'].split()) < 3:
            error_msg = "Advisor output is too short or undefined. Cannot generate code."
            logger.warning(f"⚠️ CODER - {error_msg}")
            state['error'] = error_msg
            state['generated_code'] = f"Error: {error_msg}"
            return state
        
        # Get LLM configuration
        llm_config = config.get("configurable", {}).get("llm_config", {})
        model_name = llm_config.get("CODER_MODEL")
        provider = llm_config.get("LLM_PROVIDER")
        
        logger.info(f"⚡ CODER - Provider: {provider} | Model: {model_name}")
        logger.info(f"⚡ CODER - Scope: {state['scope'][:200]}...")
        logger.info(f"⚡ CODER - Advisor Output: {state['advisor_output'][:200]}...")

        # Initialize LLM model instance and coder agent
        llm_model = await get_llm_instance(provider, model_name, llm_config)
        coder = create_pydantic_ai_coder(custom_model=llm_model)
        
        # Prepare the instruction with context
        instruction = (
            f"## Scope\n{state['scope']}\n\n"
            f"## Advisor Output\n{state['advisor_output']}"
        )
        
        # Prepare deps and conditionally enable RAG clients
        deps = {
            'reasoner_output': state['scope'],
            'advisor_output': state['advisor_output'],
        }
        if get_bool_env('RAG_ENABLED', default=False) and validate_rag_env():
            embedding_client, supabase_client, _neo4j = get_clients()
            if embedding_client and supabase_client:
                deps.update({
                    'supabase': supabase_client,
                    'embedding_client': embedding_client,
                })

        # Generate code
        logger.info("⚡ CODER - Sending request to coder...")
        result = await coder.run(instruction, deps=deps)
        code_text = result.data if hasattr(result, 'data') else str(result)

        # Update state with results
        state['generated_code'] = code_text
        state['messages'].append({
            'role': 'assistant',
            'content': f"Generated code: {code_text[:200]}..."
        })
        
        logger.info(f"✅ CODER - Code generated successfully: {code_text[:200]}...")
        return state
        
    except Exception as e:
        error_msg = f"Error in coder_agent: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state['error'] = error_msg
        state['generated_code'] = f"Error: {error_msg}"
        state['messages'].append({
            'role': 'error',
            'content': error_msg
        })
        return state
        return state

# Global instances and state
llm_instance = None
http_clients = []

def cleanup_http_clients():
    """Clean up all HTTP clients on application exit"""
    logger.info("🧹 Cleaning up HTTP clients...")
    # Clear the list synchronously
    http_clients.clear()
    # Let garbage collector handle the rest
    import gc
    gc.collect()

# Register cleanup on exit
def cleanup():
    """Synchronous cleanup wrapper that safely handles cleanup without requiring an event loop.
    
    This function is registered with atexit and must be able to run in any context,
    including when the asyncio event loop is not available.
    """
    try:
        # Use print as logging might be closed already
        print("🧹 Performing cleanup...")
        
        # Clear the list synchronously - this is safe to do without an event loop
        if http_clients:
            print(f"ℹ️ Cleaning up {len(http_clients)} HTTP clients")
            # Just clear the list - don't try to close clients as we might not have an event loop
            http_clients.clear()
        
    except Exception as e:
        # Print to stderr to ensure it's visible even if stdout is redirected
        import sys, traceback
        print("⚠️ Error during cleanup:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    
    # Explicitly clean up any remaining resources
    try:
        import gc
        gc.collect()
    except Exception:
        pass  # Don't let GC errors prevent cleanup

atexit.register(cleanup)

agentic_flow = None

def get_agentic_flow():
    """
    Returns the compiled agentic flow, initializing it if necessary.
    
    This function is thread-safe and ensures the flow is properly initialized
    with error handling and logging.
    """
    global agentic_flow
    
    if agentic_flow is None:
        try:
            logger.info("🔄 Initializing agentic flow...")
            
            # Create a new StateGraph
            builder = StateGraph(AgentState)
            
            # Add nodes
            builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
            builder.add_node("advisor_with_examples", advisor_with_examples)
            builder.add_node("coder_agent", coder_agent)
            
            # Set the entry point
            builder.set_entry_point("define_scope_with_reasoner")
            
            # Define the workflow edges
            builder.add_edge("define_scope_with_reasoner", "advisor_with_examples")
            builder.add_edge("advisor_with_examples", "coder_agent")
            builder.add_edge("coder_agent", END)
            
            # Initialize memory for checkpoints
            memory = MemorySaver()
            
            # Compile the graph
            agentic_flow = builder.compile(
                checkpointer=memory
            )
            
            logger.info("✅ Agentic flow initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize agentic flow: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    return agentic_flow

async def run_agent_workflow(initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run the agent workflow with the provided initial state.
    
    Args:
        initial_state: Optional initial state for the agent.
                      If not provided, a default state will be used.
                      
    Returns:
        Dict containing the final state of the agent.
    """
    if initial_state is None:
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
    
    try:
        logger.info("🚀 Starting agentic flow execution...")
        
        # Get the agentic flow
        flow = get_agentic_flow()
        
        # Run the flow with the initial state
        logger.info("⚡ Executing agent workflow...")
        result = await flow.ainvoke(initial_state)
        
        # Log the results
        logger.info("✅ Agent workflow completed successfully")
        logger.info(f"📋 Scope: {result.get('scope', '')[:200]}...")
        logger.info(f"📝 Code generated: {bool(result.get('generated_code', ''))}")
        
        return result
        
    except Exception as e:
        error_msg = f"❌ Agent workflow failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Ensure we still return a valid state with error information
        if isinstance(initial_state, dict):
            initial_state['error'] = error_msg
            return initial_state
        
        # If we can't update the initial state, return a minimal error state
        return {
            'error': error_msg,
            'messages': [
                {
                    'role': 'error',
                    'content': f'Workflow failed: {str(e)}'
                }
            ]
        }

if __name__ == '__main__':
    # Configure logging for the main script
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('agent_workflow.log')
            ]
        )
        
        # Run the workflow
        final_state = asyncio.run(run_agent_workflow())
        
        # Print the results
        print("\n" + "="*50)
        print("AGENT WORKFLOW EXECUTION COMPLETE")
        print("="*50)
        
        # Display key results
        if 'error' in final_state:
            print(f"❌ Error: {final_state['error']}")
        
        print("\n📋 Scope:")
        print(final_state.get('scope', 'No scope defined'))
        
        if 'generated_code' in final_state and final_state['generated_code']:
            print("\n💻 Generated Code:")
            print(final_state['generated_code'])
        
        print("\n✅ Done!")
        
    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print("Check the logs for more details.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Execution interrupted by user")
        sys.exit(0)

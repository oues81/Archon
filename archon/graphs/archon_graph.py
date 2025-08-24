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

from archon.utils.utils import configure_logging, get_bool_env, build_llm_config_from_active_profile

# Configure logging centrally
_log_summary = configure_logging()
logger = logging.getLogger(__name__)

async def _with_retries(coro_factory, tries: int = 3, base_delay: float = 0.5):
    """Run an async operation with simple exponential backoff.
    coro_factory: a zero-arg callable returning an awaitable.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, tries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            last_exc = e
            if attempt >= tries:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(f"Retrying after error (attempt {attempt}/{tries}) in LLM call: {e} | sleeping {delay:.2f}s")
            await asyncio.sleep(delay)

def _fallback_model(llm_config: Dict[str, Any]) -> str:
    """Pick an Ollama fallback model strictly from profile config.

    Raises if no fallback model is specified in the profile to avoid
    non-profile defaults or environment fallbacks.
    """
    model = llm_config.get("OLLAMA_MODEL")
    if not model:
        raise ValueError("OLLAMA_MODEL missing in profile for fallback usage")
    return model

def _ollama_allowed(llm_config: Dict[str, Any]) -> bool:
    """Whether Ollama usage is allowed by profile/config.

    Centralized guard to prevent accidental Ollama calls in container runs
    when user wants OpenRouter-only behavior.
    """
    try:
        return bool(llm_config.get("ALLOW_OLLAMA_FALLBACK", False))
    except Exception:
        return False

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
            if not _ollama_allowed(config or {}):
                raise ValueError("Ollama usage disabled by config: set ALLOW_OLLAMA_FALLBACK=True to enable")
            base_url = config.get("OLLAMA_BASE_URL")
            if not base_url:
                raise ValueError("OLLAMA_BASE_URL not found in profile")
            if not model_name:
                raise ValueError("Model name is required for Ollama provider")
            # Normalize to ensure OpenAI-compatible endpoints under /v1
            _bu = base_url.rstrip("/")
            if not _bu.endswith("/v1"):
                base_url = f"{_bu}/v1"
            else:
                base_url = _bu
            
            client_kwargs.update({"api_key": "ollama", "base_url": base_url})
            openai_client = AsyncOpenAI(**client_kwargs)
            await _ensure_ollama_model_is_pulled(openai_client, model_name)
            model = OpenAIModel(model_name=model_name, openai_client=openai_client)
            
        elif provider == "openrouter":
            api_key = config.get("LLM_API_KEY")
            if not api_key:
                raise ValueError("Missing OpenRouter API key: set 'LLM_API_KEY' in profile")
            base_url = (config.get("BASE_URL")).rstrip("/") if (config.get("BASE_URL")) else None
            if not base_url:
                raise ValueError("Missing OpenRouter BASE_URL: set 'BASE_URL' in profile")
            client_kwargs.update({
                "api_key": api_key,
                "base_url": base_url,
            })
            # Optional headers if provided in profile
            ref = config.get("OPENROUTER_REFERRER")
            xtitle = config.get("OPENROUTER_X_TITLE")
            if ref or xtitle:
                client_kwargs["default_headers"] = {}
                if ref:
                    client_kwargs["default_headers"]["HTTP-Referer"] = ref
                if xtitle:
                    client_kwargs["default_headers"]["X-Title"] = xtitle
            logger.info(f"🔗 OpenRouter base_url set to: {base_url}")
            openai_client = AsyncOpenAI(**client_kwargs)
            model = OpenAIModel(model_name=model_name, openai_client=openai_client)
            
        elif provider == "openai":
            api_key = config.get("LLM_API_KEY")
            if not api_key:
                raise ValueError("Missing OpenAI API key: set 'LLM_API_KEY' in profile")
            base_url = (config.get("BASE_URL"))
            if base_url:
                client_kwargs.update({"api_key": api_key, "base_url": base_url})
            else:
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
    from archon.archon.prompts.agent_prompts import (
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
        try:
            result = await _with_retries(lambda: reasoner.run(user_message))
        except Exception as e:
            if _ollama_allowed(llm_config):
                logger.warning(f"Reasoner failed, attempting fallback to Ollama: {e}")
                try:
                    fb_model = _fallback_model(llm_config)
                    fb_llm = await get_llm_instance("ollama", fb_model, llm_config)
                    reasoner_fb = PydanticAgent(fb_llm, system_prompt=reasoner_prompt)
                    result = await _with_retries(lambda: reasoner_fb.run(user_message))
                except Exception:
                    raise
            else:
                raise
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
        try:
            result = await _with_retries(lambda: advisor.run(state['scope']))
        except Exception as e:
            if _ollama_allowed(llm_config):
                logger.warning(f"Advisor failed, attempting fallback to Ollama: {e}")
                try:
                    fb_model = _fallback_model(llm_config)
                    fb_llm = await get_llm_instance("ollama", fb_model, llm_config)
                    advisor_fb = PydanticAgent(fb_llm, system_prompt=advisor_prompt)
                    result = await _with_retries(lambda: advisor_fb.run(state['scope']))
                except Exception:
                    raise
            else:
                raise
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
        try:
            result = await _with_retries(lambda: coder.run(instruction, deps=deps))
        except Exception as e:
            if _ollama_allowed(llm_config):
                logger.warning(f"Coder failed, attempting fallback to Ollama: {e}")
                try:
                    fb_model = _fallback_model(llm_config)
                    fb_llm = await get_llm_instance("ollama", fb_model, llm_config)
                    coder_fb = create_pydantic_ai_coder(custom_model=fb_llm)
                    result = await _with_retries(lambda: coder_fb.run(instruction, deps=deps))
                except Exception:
                    raise
            else:
                raise
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
        
        # Build strict llm_config from active profile and merge any overrides from state
        overrides = {}
        try:
            overrides = (initial_state or {}).get("llm_overrides") or {}
        except Exception:
            overrides = {}
        llm_config = build_llm_config_from_active_profile(overrides if isinstance(overrides, dict) else None)

        # Prepare flow and config
        flow = get_agentic_flow()
        corr = (initial_state or {}).get("correlation_id") or "agent-graph"
        config = {"configurable": {"thread_id": str(corr), "llm_config": llm_config}}
        logger.info(
            f"⚡ Executing agent workflow with provider='{llm_config.get('LLM_PROVIDER')}' "
            f"reasoner='{llm_config.get('REASONER_MODEL')}' advisor='{llm_config.get('ADVISOR_MODEL')}' coder='{llm_config.get('CODER_MODEL')}'"
        )

        # Run the flow with configured llm_config
        result = await flow.ainvoke(initial_state, config)
        
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

# --- Backward-compatibility proxy to canonical implementation ---
# Delegate public API to canonical graphs module to avoid duplication.
# This keeps import paths stable while the real implementation lives under
# `archon.archon.graphs.archon.app.graph`.
try:
    from archon.archon.graphs.archon.app.graph import (
        get_agentic_flow as _canonical_get_agentic_flow,
        run_agent_workflow as _canonical_run_agent_workflow,
    )
    get_agentic_flow = _canonical_get_agentic_flow  # type: ignore
    run_agent_workflow = _canonical_run_agent_workflow  # type: ignore
    __all__ = [
        'get_agentic_flow',
        'run_agent_workflow',
    ]
except Exception as _proxy_err:
    # If canonical module isn't importable, keep legacy definitions.
    # Log at debug level only to not spam in normal runs.
    try:
        logging.getLogger(__name__).debug(
            f"Proxy to canonical graphs.archon failed: {_proxy_err}"
        )
    except Exception:
        pass

from __future__ import annotations as _annotations

import os
import sys
from typing import List, Dict, Any

from dotenv import load_dotenv
import logfire
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
try:
    from pydantic_ai.models.anthropic import AnthropicModel  # optional
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    class AnthropicModel:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("pydantic_ai Anthropic extra not installed")
from archon.archon.ollama_model import OllamaModel

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from archon.utils.utils import get_env_var, get_bool_env, validate_rag_env
from archon.archon.agent_prompts import primary_coder_prompt
from archon.archon.agent_tools import (
    retrieve_relevant_documentation_tool,
    list_documentation_pages_tool,
    get_page_content_tool
)

load_dotenv()

# --- LLM and Agent Configuration ---
provider = get_env_var('LLM_PROVIDER') or 'OpenAI'
llm = get_env_var('PRIMARY_MODEL') or 'gpt-4o-mini'
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

if provider == "Anthropic":
    if _ANTHROPIC_AVAILABLE:
        model = AnthropicModel(llm, api_key=api_key)
    else:
        # Fallback to Ollama when Anthropic extra is not available
        model = OllamaModel(model_name=llm, base_url=base_url)
elif provider == "Ollama":
    model = OllamaModel(model_name=llm, base_url=base_url)
else:
    # Par défaut, utiliser Ollama avec les paramètres par défaut
    model = OllamaModel(model_name=llm, base_url=base_url)

logfire.configure(send_to_logfire='if-token-present')

# Define tools as standalone functions
async def retrieve_relevant_documentation(ctx: RunContext[Dict[str, Any]], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    """
    supabase_client = ctx.deps["supabase"]
    embedding_client = ctx.deps["embedding_client"]
    return await retrieve_relevant_documentation_tool(supabase_client, embedding_client, user_query)

async def list_documentation_pages(ctx: RunContext[Dict[str, Any]]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    """
    supabase_client = ctx.deps["supabase"]
    return await list_documentation_pages_tool(supabase_client)

async def get_page_content(ctx: RunContext[Dict[str, Any]], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page.
    """
    supabase_client = ctx.deps["supabase"]
    return await get_page_content_tool(supabase_client, url)

# Module-level default; actual tools are decided dynamically in create_pydantic_ai_coder()
coder_tools = []

# Export the model for reuse
coder_model = model

# Export the tools for reuse
pydantic_coder_tools = coder_tools

# Export the dynamic prompt function for reuse (but we won't use it directly in the agent constructor)
def dynamic_coder_prompt(ctx: RunContext[Dict[str, Any]]) -> str:
    """Appends reasoner and advisor outputs to the base prompt."""
    reasoner_output = ctx.deps.get("reasoner_output", "No reasoner output provided.")
    advisor_output = ctx.deps.get("advisor_output", "No advisor output provided.")
    return f"""{primary_coder_prompt}\n\n    Additional thoughts/instructions from the reasoner LLM. \n    This scope includes documentation pages for you to search as well: \n    {reasoner_output}\n\n    Recommended starting point from the advisor agent:\n    {advisor_output}\n    """

def create_pydantic_ai_coder(custom_model: Any | None = None):
    """Creates and returns a new instance of the pydantic_ai_coder Agent.

    Args:
        custom_model: Optional LLM model instance to use for the agent. If not provided,
                      falls back to the module-level configured `model`.

    Returns:
        Agent: Configured Pydantic AI agent with RAG tools enabled.
    """
    # Use a string for the base system prompt instead of a function
    base_system_prompt = (
        "You are an expert AI agent engineer specializing in building Pydantic AI agents. "
        "Help the user create and refine their agent."
    )

    # Choose model: prefer the injected one to comply with strict profile-based selection
    selected_model = custom_model if custom_model is not None else model

    # Dynamically enable RAG tools at creation time based on env
    if get_bool_env('RAG_ENABLED', default=False) and validate_rag_env():
        local_tools = [
            retrieve_relevant_documentation,
            list_documentation_pages,
            get_page_content,
        ]
    else:
        local_tools = []

    # Create agent with string system prompt and explicit deps_type to allow dict-based deps
    agent = Agent(
        model=selected_model,
        system_prompt=base_system_prompt,
        tools=local_tools,
        retries=2,
        deps_type=Dict[str, Any],
    )

    # Optionally, you can still add dynamic content by using the decorator
    @agent.system_prompt
    def append_dynamic_content(ctx: RunContext[Dict[str, Any]]) -> str:
        reasoner_output = getattr(ctx.deps, "reasoner_output", "No reasoner output provided.")
        advisor_output = getattr(ctx.deps, "advisor_output", "No advisor output provided.")
        return f"""
        Additional thoughts/instructions from the reasoner LLM:
        This scope includes documentation pages for you to search as well:
        {reasoner_output}

        Recommended starting point from the advisor agent:
        {advisor_output}
        """

    return agent
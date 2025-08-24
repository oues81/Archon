from __future__ import annotations

from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import logfire
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

from archon.utils.utils import get_env_var
from archon.archon.schemas import PydanticAIDeps
from archon.archon.prompts.agent_prompts import primary_coder_prompt
from archon.archon.agents.agent_tools import (
    retrieve_relevant_documentation_tool,
    list_documentation_pages_tool,
    get_page_content_tool,
)
from archon.archon.models import OllamaModel


# --- Helpers expected by unit tests ---
def get_bool_env(key: str, default: bool = False) -> bool:
    """Return a boolean from env via `get_env_var` with common truthy values.

    Tests monkeypatch this function; keep logic simple and deterministic.
    """
    val = (get_env_var(key) or "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "y", "on"}


def validate_rag_env() -> bool:
    """Placeholder validation for RAG-related env. Tests monkeypatch this."""
    return True


load_dotenv()
logfire.configure(send_to_logfire="if-token-present")


# --- Tool functions (async signatures kept for compatibility with real runtime) ---
async def retrieve_relevant_documentation(ctx, user_query: str) -> str:  # type: ignore[no-untyped-def]
    supabase_client = ctx.deps["supabase"]
    embedding_client = ctx.deps["embedding_client"]
    return await retrieve_relevant_documentation_tool(supabase_client, embedding_client, user_query)


async def list_documentation_pages(ctx) -> List[str]:  # type: ignore[no-untyped-def]
    supabase_client = ctx.deps["supabase"]
    return await list_documentation_pages_tool(supabase_client)


async def get_page_content(ctx, url: str) -> str:  # type: ignore[no-untyped-def]
    supabase_client = ctx.deps["supabase"]
    return await get_page_content_tool(supabase_client, url)


# --- Dynamic prompt ---
def dynamic_coder_prompt(ctx) -> str:  # type: ignore[no-untyped-def]
    reasoner_output = ctx.deps.get("reasoner_output", "No reasoner output provided.")
    advisor_output = ctx.deps.get("advisor_output", "No advisor output provided.")
    return (
        f"{primary_coder_prompt}\n\n"
        "    Additional thoughts/instructions from the reasoner LLM. \n"
        "    This scope includes documentation pages for you to search as well: \n"
        f"    {reasoner_output}\n\n"
        "    Recommended starting point from the advisor agent:\n"
        f"    {advisor_output}\n"
    )


# --- Factory ---
def _resolve_model() -> Any:
    provider = get_env_var("LLM_PROVIDER") or "OpenAI"
    llm = get_env_var("PRIMARY_MODEL") or "gpt-4o-mini"
    base_url = get_env_var("BASE_URL") or "http://localhost:11434"
    api_key = get_env_var("LLM_API_KEY") or ""
    if provider == "Anthropic":
        return AnthropicModel(llm, api_key=api_key)
    # Default: Ollama (local/dev friendly)
    return OllamaModel(model_name=llm, base_url=base_url)


def create_pydantic_ai_coder(custom_model: Optional[Any] = None):
    """Create and return the Coder Agent.

    Parameters
    - custom_model: optional override for the underlying model. Tests use a string.
    """
    # Toggle tools via env (tests monkeypatch get_bool_env/validate_rag_env)
    tools: List[Any] = []
    if get_bool_env("ENABLE_RAG", False) and validate_rag_env():
        tools = [
            retrieve_relevant_documentation,
            list_documentation_pages,
            get_page_content,
        ]

    model: Any = custom_model if custom_model is not None else _resolve_model()

    agent = Agent(
        model=model,
        system_prompt=primary_coder_prompt,  # static string for base prompt
        tools=tools,
        retries=2,
        deps_type=PydanticAIDeps,
    )
    # Register dynamic system prompt through decorator API too (unit tests assert this)
    agent.system_prompt(dynamic_coder_prompt)
    return agent
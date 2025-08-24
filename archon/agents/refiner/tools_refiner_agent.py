from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import sys
import json
import logging
from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models import Model  # Import direct depuis pydantic_ai.models
from archon.models.ollama_model import OllamaModel
from typing import Any, Dict, List, Optional, AsyncIterator, Union, cast, AsyncGenerator
from openai import AsyncOpenAI
import aiohttp
import json

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import get_env_var
from archon.archon.agent_prompts import tools_refiner_prompt
from archon.agent_tools import (
    retrieve_relevant_documentation_tool,
    list_documentation_pages_tool,
    get_page_content_tool,
    get_file_content_tool
)

# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

load_dotenv()

provider = get_env_var('LLM_PROVIDER') or 'Ollama'  # Par défaut sur Ollama
llm = get_env_var('LLM_MODEL') or 'phi3:mini'  # Modèle par défaut mis à jour
base_url = get_env_var('BASE_URL') or 'http://localhost:11434'  # Utilisation de localhost par défaut
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

# Configuration du modèle en fonction du fournisseur
provider_lower = provider.lower()
logger.info(f"Chargement du fournisseur: {provider_lower}, modèle: {llm}, base_url: {base_url}")

try:
    if provider_lower == "anthropic":
        logger.info(f"Initialisation du modèle Anthropic: {llm}")
        model = AnthropicModel(llm, api_key=api_key)
        
    elif provider_lower == "ollama":
        # Configuration pour Ollama
        model_name = llm.split(':')[0]  # Prendre juste le nom du modèle sans le tag
        logger.info(f"Initialisation du modèle Ollama: {model_name} (URL: {base_url})")
        model = OllamaModel(model_name=model_name, base_url=base_url)
        
    elif provider_lower == "openrouter":
        # Configuration pour OpenRouter
        logger.info(f"Initialisation du modèle OpenRouter: {llm}")
        
        # Configuration des en-têtes pour OpenRouter
        headers = {
            "HTTP-Referer": "https://github.com/oues/archon",
            "X-Title": "Archon AI Agent"
        }
        
        # Utilisation d'un modèle Ollama local comme modèle factice pour pydantic-ai
        # Nous allons intercepter les appels plus tard pour les rediriger vers OpenRouter
        logger.info("Utilisation d'un modèle Ollama local comme modèle factice")
        model = OllamaModel(model_name="phi3:mini", base_url="http://host.docker.internal:11434")
        
        # Enregistrement des informations OpenRouter pour une utilisation ultérieure
        model._openrouter_config = {
            'api_key': api_key,
            'model': llm,
            'headers': headers
        }
        
    elif provider_lower == "openai":
        # Configuration pour OpenAI standard
        openai_url = base_url if base_url else "https://api.openai.com/v1"
        logger.info(f"Initialisation du modèle OpenAI: {llm} (URL: {openai_url})")
        model = OpenAIModel(
            model_name=llm,
            api_key=api_key,
            base_url=openai_url
        )
        
    else:
        logger.warning(f"Fournisseur non reconnu: {provider_lower}. Utilisation du modèle brut.")
        model = llm
        
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation du modèle {llm} avec le fournisseur {provider_lower}: {str(e)}")
    raise

embedding_model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small'

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class ToolsRefinerDeps:
    supabase: Client
    embedding_client: AsyncOpenAI
    file_list: List[str]

tools_refiner_agent = Agent(
    model,
    system_prompt=tools_refiner_prompt,
    deps_type=ToolsRefinerDeps,
    retries=2
)


@tools_refiner_agent.system_prompt  
def add_file_list(ctx: RunContext[str]) -> str:
    joined_files = "\n".join(ctx.deps.file_list)
    return f"""
    
    Here is the list of all the files that you can pull the contents of with the
    'get_file_content' tool if the example/tool/MCP server is relevant to the
    agent the user is trying to build:

    {joined_files}
    """

@tools_refiner_agent.tool
async def retrieve_relevant_documentation(ctx: RunContext[ToolsRefinerDeps], query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    Make sure your searches always focus on implementing tools.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        query: Your query to retrieve relevant documentation for implementing tools
        
    Returns:
        A formatted string containing the top 4 most relevant documentation chunks
    """
    return await retrieve_relevant_documentation_tool(ctx.deps.supabase, ctx.deps.embedding_client, query)

@tools_refiner_agent.tool
async def list_documentation_pages(ctx: RunContext[ToolsRefinerDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    This will give you all pages available, but focus on the ones related to tools.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    return await list_documentation_pages_tool(ctx.deps.supabase)

@tools_refiner_agent.tool
async def get_page_content(ctx: RunContext[ToolsRefinerDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    Only use this tool to get pages related to using tools with Pydantic AI.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    return await get_page_content_tool(ctx.deps.supabase, url)

@tools_refiner_agent.tool_plain
def get_file_content(file_path: str) -> str:
    """
    Retrieves the content of a specific file. Use this to get the contents of an example, tool, config for an MCP server
    
    Args:
        file_path: The path to the file
        
    Returns:
        The raw contents of the file
    """
    return get_file_content_tool(file_path)    
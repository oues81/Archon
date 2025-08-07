from __future__ import annotations as _annotations

import logging
import os
import sys
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import List, Dict, Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel

from archon.archon.agent_prompts import advisor_prompt
from archon.archon.agent_tools import get_file_content_tool
from archon.utils import get_env_var

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AdvisorDeps:
    file_list: List[str]

# Define tool as standalone function
async def get_file_content(ctx: RunContext[AdvisorDeps], file_path: str) -> str:
    """
    Retrieves the content of a specific file. Use this to get the contents of an example, tool, config for an MCP server
    
    Args:
        file_path: The path to the file
        
    Returns:
        The raw contents of the file
    """
    return get_file_content_tool(file_path)

# Define the list of tools for the advisor agent
advisor_tools = [get_file_content]

class AdvisorAgent:
    def __init__(self, model_config: dict):
        # Make config reading robust: prefer 'LLM', fallback to 'PRIMARY_MODEL', then to a default.
        llm = model_config.get('LLM') or model_config.get('PRIMARY_MODEL') or 'phi3:mini'
        base_url = model_config.get('BASE_URL')
        api_key = model_config.get('LLM_API_KEY')

        logger.info(f"Initializing AdvisorAgent with model: {llm}")

        # Treat all models as OpenAI-compatible. This works for OpenRouter and Ollama's API.
        # For Ollama, provide a dummy API key since it's not needed
        api_key = api_key or "dummy-key"
        model = OpenAIModel(llm, base_url=base_url, api_key=api_key)

        # Create agent with string system prompt
        self.agent = Agent(
            model=model,
            system_prompt=advisor_prompt,  # Use base prompt string
            tools=advisor_tools,
            retries=2
        )
        
        # Add dynamic content via decorator
        @self.agent.system_prompt
        def add_file_list(ctx: RunContext[AdvisorDeps]) -> str:
            joined_files = "\n".join(ctx.deps.file_list)
            return f"""
            Here is the list of all the files that you can pull the contents of with the
            'get_file_content' tool if the example/tool/MCP server is relevant to the
            agent the user is trying to build:

            {joined_files}
            """
    
    async def run(self, message: str, deps: AdvisorDeps):
        """Run the advisor agent with the given message and dependencies."""
        return await self.agent.run(message, deps=deps)

# Factory function for backward compatibility
def get_default_agent() -> AdvisorAgent:
    load_dotenv()
    config = {
        'LLM': get_env_var('LLM') or get_env_var('PRIMARY_MODEL'),
        'BASE_URL': get_env_var('BASE_URL'),
        'LLM_API_KEY': get_env_var('LLM_API_KEY'),
    }
    # Filter out None values so defaults in __init__ can apply
    config = {k: v for k, v in config.items() if v is not None}
    return AdvisorAgent(config)

# This global instance will be used by other parts of the application that are not tests.
# It will be created on first import.
if 'pytest' not in sys.modules:
    advisor_agent = get_default_agent()

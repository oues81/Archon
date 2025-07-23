from __future__ import annotations as _annotations

import logfire
import os
import sys
from dotenv import load_dotenv
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from archon.models.ollama_model import OllamaModel

# Add the parent directory to sys.path to allow importing from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var
from archon.agent_prompts import prompt_refiner_agent_prompt

# Chargement des variables d'environnement
load_dotenv()

provider = (get_env_var('LLM_PROVID') or 'ollama').lower()  # Par défaut sur Ollama
llm = get_env_var('LLM_MODEL') or 'phi3:mini'  # Modèle par défaut mis à jour
base_url = get_env_var('BASE_URL') or 'http://host.docker.internal:11434'  # Utilisation de l'hôte Docker par défaut
api_key = get_env_var('API_KEY') or get_env_var('OPENROUTER_API_KEY') or 'no-api-key-provided'

# Configuration du modèle en fonction du fournisseur
if provider == "anthropic":
    model = AnthropicModel(llm, api_key=api_key)
elif provider == "openrouter":
    # Pour OpenRouter, on utilise OpenAIModel avec l'URL de base d'OpenRouter
    model = OpenAIModel(
        model=llm,
        api_key=api_key,
        base_url='https://openrouter.ai/api/v1',
        http_headers={
            "HTTP-Referer": "https://github.com/oues/archon",
            "X-Title": "Archon AI Agent"
        }
    )
elif provider == "ollama":
    # Pour Ollama, on utilise OllamaModel avec l'URL de base
    model_name = llm.split(':')[0]  # Prendre juste le nom du modèle sans le tag
    model = OllamaModel(model_name=model_name, base_url=base_url)
else:
    # Par défaut, on utilise le modèle tel quel (pour compatibilité)
    model = llm

logfire.configure(send_to_logfire='if-token-present')

prompt_refiner_agent = Agent(
    model,
    system_prompt=prompt_refiner_agent_prompt
)
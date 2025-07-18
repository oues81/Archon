from __future__ import annotations as _annotations

import logging
import os
import sys
import traceback

# Configuration du logger AVANT toute autre importation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ajout du répertoire parent au path pour permettre les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importations après la configuration du logger
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
from typing import List, Optional, Any, Dict, Union, AsyncIterator
import asyncio
import httpx
import json
from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent, ModelRetry, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models import Model as BaseModel
from supabase import Client
from archon.models.ollama_model import OllamaClient, OllamaModel
from utils.utils import get_env_var
from archon.agent_prompts import advisor_prompt
from archon.agent_tools import get_file_content_tool

# Chargement des variables d'environnement
load_dotenv()

provider = get_env_var('LLM_PROVIDER') or 'Ollama'  # Par défaut sur Ollama
llm = get_env_var('LLM_MODEL') or 'phi3:mini'  # Modèle par défaut mis à jour
base_url = get_env_var('BASE_URL') or 'http://172.26.224.1:11434'  # IP de l'hôte WSL2 par défaut
api_key = get_env_var('LLM_API_KEY') or 'no-llm-api-key-provided'

class OllamaModelWrapper:
    """Wrapper pour le modèle qui implémente l'interface attendue par Pydantic AI."""
    
    def __init__(self, model_name: str, base_url: str = None):
        self._model_name = model_name
        self._client = OllamaClient(base_url=base_url if base_url else 'http://localhost:11434')
        # Ajout des attributs nécessaires pour la compatibilité avec pydantic-ai 0.0.22
        self.requests = self  # Pour la rétrocompatibilité
        self.usage = type('Usage', (), {'requests': 0, 'tokens': 0})()
    
    def name(self) -> str:
        return self._model_name
    
    async def run(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Exécute le modèle avec les messages donnés et retourne la réponse."""
        try:
            print(f"[DEBUG] ModelWrapper.run called with messages: {messages}")
            response = await self._client.chat(messages, self._model_name, **kwargs)
            content = response.get("message", {}).get("content", "")
            print(f"[DEBUG] ModelWrapper.run response: {content[:200]}...")
            return content
        except Exception as e:
            print(f"[ERROR] ModelWrapper.run error: {str(e)}")
            print("[ERROR] Traceback:", traceback.format_exc())
            raise
    
    async def run_stream(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        """Exécute le modèle en streaming avec les messages donnés."""
        try:
            print(f"[DEBUG] ModelWrapper.run_stream called with messages: {messages}")
            async for chunk in self._client.chat_stream(messages, self._model_name, **kwargs):
                content = chunk.get("message", {}).get("content", "")
                if content:
                    print(f"[DEBUG] Streaming chunk: {content[:100]}...")
                    yield content
        except Exception as e:
            print(f"[ERROR] ModelWrapper.run_stream error: {str(e)}")
            print("[ERROR] Traceback:", traceback.format_exc())
            raise

# Initialize the appropriate model based on the provider
if provider == 'Ollama':
    # Utiliser directement le nom du modèle pour pydantic_ai
    model_name = llm
    
    # Utiliser la même URL de base pour tous les appels
    ollama_base_url = os.getenv('BASE_URL', 'http://localhost:11434')
    
    # Créer une instance du client Ollama pour une utilisation ultérieure
    ollama_client = OllamaClient(base_url=ollama_base_url)
    
    # Configuration pour Ollama
    model_config = {
        'model': model_name,
        'model_kwargs': {
            'base_url': ollama_base_url
        }
    }
    
    logger.info(f"Configuration du modèle Ollama - URL: {ollama_base_url}, Modèle: {model_name}")
else:
    # Pour OpenAI, on utilise directement le modèle OpenAI
    from pydantic_ai.models.openai import OpenAIModel
    model_name = llm
    model = OpenAIModel(llm, base_url=base_url, api_key=api_key)
    model_config = {
        'model': model_name,
        'model_kwargs': {
            'base_url': base_url,
            'api_key': api_key
        }
    }

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class AdvisorDeps:
    file_list: List[str]

# Création de l'agent conseiller
if provider == 'Ollama':
    try:
        # Créer le modèle Ollama avec les paramètres corrects
        model = OllamaModel(
            model_name=llm,
            base_url=base_url,
            temperature=0.7,
            max_tokens=2048
        )
        
        # Créer un wrapper simple pour le modèle qui respecte l'interface de Pydantic AI
        class CustomModelWrapper(BaseModel):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self._model_name = getattr(model, '_model_name', 'ollama-model')
                # Ajouter les attributs nécessaires pour la compatibilité
                self.requests = self
                self.usage = type('Usage', (), {'requests': 0, 'tokens': 0})()
            
            @property
            def model_name(self) -> str:
                return self._model_name
                
            @property
            def system(self) -> str:
                return "Ollama"
                
            @property
            def name(self) -> str:
                return self._model_name
                
            async def agent_model(self, function_tools=None, **kwargs):
                return self
                
            async def request(self, messages, model_settings=None, function_tools=None, **kwargs):
                try:
                    # Convertir les messages si nécessaire
                    formatted_messages = []
                    for msg in messages:
                        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                            formatted_messages.append(msg)
                        else:
                            formatted_messages.append({"role": "user", "content": str(msg)})
                    
                    # Fusionner les paramètres du modèle
                    request_kwargs = {}
                    if model_settings and isinstance(model_settings, dict):
                        request_kwargs.update(model_settings)
                    request_kwargs.update(kwargs)
                    
                    # Appeler le modèle avec les messages formatés
                    response = await self.model.request(formatted_messages, **request_kwargs)
                    
                    # Mettre à jour le compteur d'utilisation
                    self.usage.requests += 1
                    if hasattr(response, 'get') and 'usage' in response:
                        self.usage.tokens += response['usage'].get('total_tokens', 0)
                    
                    # Créer un état valide avec tous les champs requis
                    state_update = {
                        'latest_user_message': messages[-1]['content'] if messages and isinstance(messages[-1], dict) and 'content' in messages[-1] else '',
                        'messages': [msg if isinstance(msg, bytes) else str(msg).encode() for msg in messages],
                        'scope': '',
                        'advisor_output': response.get('content', '') if hasattr(response, 'get') else str(response),
                        'file_list': [],
                        'refined_prompt': '',
                        'refined_tools': '',
                        'refined_agent': '',
                        'usage': self.usage  # Inclure l'utilisation dans l'état
                    }
                    
                    # Retourner la réponse et l'état mis à jour
                    return state_update, {}
                except Exception as e:
                    logger.error(f"Erreur dans ModelWrapper.request: {str(e)}", exc_info=True)
                    raise
        
        # Créer l'agent avec le modèle configuré
        advisor_agent = PydanticAgent(
            model=CustomModelWrapper(model),
            system_prompt=advisor_prompt,
            deps_type=AdvisorDeps,
            retries=2
        )
        
        logger.info(f"Agent conseiller initialisé avec le modèle: {llm}")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du modèle Ollama: {str(e)}")
        raise
else:
    # Pour les autres fournisseurs (OpenAI, etc.), utiliser la configuration standard
    try:
        advisor_agent = PydanticAgent(
            model=model_name,
            system_prompt=advisor_prompt,
            deps_type=AdvisorDeps,
            retries=2,
            **model_config
        )
        logger.info(f"Agent conseiller initialisé avec le modèle: {model_name}")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du modèle {provider}: {str(e)}")
        raise

@advisor_agent.system_prompt  
def add_file_list(ctx: RunContext[str]) -> str:
    joined_files = "\n".join(ctx.deps.file_list)
    return f"""
    
    Here is the list of all the files that you can pull the contents of with the
    'get_file_content' tool if the example/tool/MCP server is relevant to the
    agent the user is trying to build:

    {joined_files}
    """

@advisor_agent.tool_plain
def get_file_content(file_path: str) -> str:
    """
    Retrieves the content of a specific file. Use this to get the contents of an example, tool, config for an MCP server
    
    Args:
        file_path: The path to the file
        
    Returns:
        The raw contents of the file
    """
    return get_file_content_tool(file_path)
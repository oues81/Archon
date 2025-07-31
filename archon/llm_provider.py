"""
Fournisseur LLM unifié pour Archon AI
Gère les différents fournisseurs de modèles de langage (Ollama, OpenAI, OpenRouter)
"""
import os
import sys
import traceback
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import httpx
from dotenv import load_dotenv

# Configurer le logger
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# Chemin vers le répertoire de travail
WORKBENCH_DIR = Path(__file__).parent.parent / "workbench"
ENV_VARS_FILE = WORKBENCH_DIR / "env_vars.json"

def load_profile_config():
    """Charge la configuration du profil actif"""
    try:
        if not ENV_VARS_FILE.exists():
            logger.warning(f"Fichier de configuration introuvable: {ENV_VARS_FILE}")
            return {}
            
        with open(ENV_VARS_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        current_profile = config.get('current_profile', 'default')
        profiles = config.get('profiles', {})
        
        if current_profile not in profiles:
            logger.warning(f"Le profil actif '{current_profile}' n'existe pas dans la configuration.")
            return {}
            
        return profiles[current_profile]
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration du profil: {e}")
        return {}

def get_env_from_profile(var_name: str, default: str = None) -> str:
    """Récupère une variable d'environnement depuis le profil actif"""
    profile_config = load_profile_config()
    if var_name in profile_config and profile_config[var_name] not in (None, ''):
        return str(profile_config[var_name])
    return os.getenv(var_name, default)

@dataclass
class LLMConfig:
    """Configuration pour le fournisseur LLM"""
    provider: str = field(default_factory=lambda: get_env_from_profile("LLM_PROVIDER", "ollama"))
    model: str = field(default_factory=lambda: get_env_from_profile("LLM_MODEL", "llama3"))
    reasoner_model: str = field(default_factory=lambda: get_env_from_profile("REASONER_MODEL", "mistralai/mistral-7b-instruct"))
    primary_model: str = field(default_factory=lambda: get_env_from_profile("PRIMARY_MODEL", "mistralai/mistral-7b-instruct"))
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
@dataclass
class LLMProvider:
    """Classe unifiée pour interagir avec différents fournisseurs LLM"""
    config: LLMConfig = field(default_factory=LLMConfig)
    client: Any = None
    pydantic_ai_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialise le client en fonction du fournisseur"""
        try:
            provider_lower = self.config.provider.lower()
            if provider_lower == "openrouter":
                self._setup_openrouter()
            elif provider_lower == "openai":
                self._setup_openai()
            elif provider_lower == "ollama":
                self._setup_ollama()
            else:
                raise ValueError(f"Fournisseur LLM non supporté: {self.config.provider}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du LLMProvider: {e}")
            logger.error(traceback.format_exc())

    def _setup_openrouter(self):
        """Configure le client pour OpenRouter"""
        logger.info("Configuration pour OpenRouter...")
        self.config.api_key = get_env_from_profile("OPENROUTER_API_KEY")
        self.config.base_url = get_env_from_profile("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if not self.config.api_key:
            raise ValueError("La clé API OpenRouter (OPENROUTER_API_KEY) n'est pas définie.")
        self.pydantic_ai_config = {
            "provider": "openrouter",
            "api_key": self.config.api_key,
            "base_url": self.config.base_url
        }
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        )
        logger.info("Configuration OpenRouter terminée.")

    def _setup_openai(self):
        """Configure le client pour OpenAI"""
        logger.info("Configuration pour OpenAI...")
        self.config.api_key = get_env_from_profile("OPENAI_API_KEY")
        self.config.base_url = get_env_from_profile("OPENAI_BASE_URL")
        if not self.config.api_key:
            raise ValueError("La clé API OpenAI (OPENAI_API_KEY) n'est pas définie.")
        self.pydantic_ai_config = {
            "provider": "openai",
            "api_key": self.config.api_key
        }
        if self.config.base_url:
            self.pydantic_ai_config['base_url'] = self.config.base_url
        self.client = None 
        logger.info("Configuration OpenAI terminée.")

    def _setup_ollama(self):
        """Configure le client pour Ollama"""
        logger.info("Configuration pour Ollama...")
        self.config.base_url = get_env_from_profile("OLLAMA_BASE_URL", "http://localhost:11434")
        self.pydantic_ai_config = {}
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=120.0,
            headers={"Content-Type": "application/json"}
        )
        logger.info(f"Configuration Ollama terminée. URL de base: {self.config.base_url}")

    async def generate(self, messages: List[Dict[str, str]], model: Optional[str] = None, stream: bool = False, **kwargs) -> Union[Dict[str, Any], str]:
        """Génère une réponse à partir d'une liste de messages"""
        if not self.client and self.config.provider.lower() != 'openai':
            raise Exception("Le client LLM n'est pas initialisé. Vérifiez la configuration.")
        target_model = model or self.config.model
        if self.config.provider.lower() == "ollama":
            return await self._generate_with_ollama(messages, target_model, stream, **kwargs)
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        try:
            response = await self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            if stream:
                return "Streaming non implémenté pour cet appel direct."
            else:
                return response.json()
        except httpx.HTTPStatusError as e:
            error_content = e.response.text
            error_msg = f"Erreur HTTP {e.response.status_code} de l'API: {error_content}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Erreur de requête vers l'API: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Erreur inattendue: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise Exception(error_msg) from e

    def _format_messages_for_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Formate les messages pour l'API Ollama"""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if role == "system":
                formatted.append(f"SYSTEM: {content}")
            elif role == "assistant":
                formatted.append(f"ASSISTANT: {content}")
            else:  # user
                formatted.append(f"USER: {content}")
        return "\n".join(formatted) + "\nASSISTANT: "

    async def _generate_with_ollama(self, messages: List[Dict[str, str]], model: str, stream: bool, **kwargs) -> Dict[str, Any]:
        """Génère une réponse avec Ollama en utilisant un appel direct"""
        # Vérifier que le modèle est disponible
        try:
            models_response = await self.client.get("/api/tags")
            models_response.raise_for_status()
            available_models = {m["name"]: m for m in models_response.json().get("models", [])}
            
            if model not in available_models:
                logger.warning(f"Modèle {model} non trouvé. Modèles disponibles: {', '.join(available_models.keys())}")
                # Essayer avec le modèle par défaut
                model = "phi4-mini:latest"
        except Exception as e:
            logger.warning(f"Impossible de vérifier les modèles disponibles: {e}")
            # En cas d'erreur, on continue avec le modèle demandé
        
        # Préparer le payload
        payload = {
            "model": model,
            "prompt": self._format_messages_for_ollama(messages),
            "stream": False,  # Désactiver le streaming pour simplifier
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "num_ctx": 4096,
                **kwargs.get("options", {})
            }
        }
        
        # Ajouter max_tokens si spécifié
        if "max_tokens" in kwargs:
            payload["options"]["num_predict"] = kwargs["max_tokens"]
        
        logger.debug(f"Envoi de la requête à Ollama: {json.dumps(payload, indent=2)}")
        
        try:
            # Envoyer la requête avec un timeout plus long
            async with self.client.stream(
                "POST",
                "/api/generate",
                json=payload,
                timeout=180.0  # 3 minutes de timeout
            ) as response:
                response.raise_for_status()
                
                # Traiter la réponse
                full_response = ""
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                full_response += chunk["response"]
                            if chunk.get("done", False):
                                # Dernier chunk, retourner la réponse complète
                                return {
                                    "id": f"ollama-{chunk.get('created_at', '')}",
                                    "object": "chat.completion",
                                    "created": chunk.get("created_at", int(time.time())),
                                    "model": model,
                                    "choices": [{
                                        "index": 0,
                                        "message": {
                                            "role": "assistant",
                                            "content": full_response.strip()
                                        },
                                        "finish_reason": chunk.get("done_reason", "stop")
                                    }],
                                    "usage": {
                                        "prompt_tokens": chunk.get("eval_count", 0),
                                        "completion_tokens": chunk.get("prompt_eval_count", 0),
                                        "total_tokens": chunk.get("eval_count", 0) + chunk.get("prompt_eval_count", 0)
                                    }
                                }
                        except json.JSONDecodeError:
                            logger.warning(f"Impossible de parser le chunk JSON: {line}")
                
                # Si on arrive ici sans avoir fini, retourner ce qu'on a
                return {
                    "id": f"ollama-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_response.strip()
                        },
                        "finish_reason": "length"
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
                
        except httpx.HTTPStatusError as e:
            error_content = e.response.text if e.response else "Pas de réponse"
            error_msg = f"Erreur HTTP {getattr(e.response, 'status_code', 'inconnu')} d'Ollama: {error_content}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
            
        except httpx.RequestError as e:
            error_msg = f"Erreur de requête vers Ollama: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
            
        except Exception as e:
            error_msg = f"Erreur inattendue avec Ollama: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Erreur inattendue lors de l'appel à Ollama: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise Exception(error_msg) from e
    
    def _format_messages_for_ollama(self, messages: List[Dict[str, str]]) -> str:
        """
        Formate les messages pour Ollama
        Ollama utilise un format simple de prompt, nous combinons donc les messages
        """
        formatted_prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted_prompt += f"<system>\n{content}\n</system>\n\n"
            elif role == "user":
                formatted_prompt += f"<user>\n{content}\n</user>\n\n"
            elif role == "assistant":
                formatted_prompt += f"<assistant>\n{content}\n</assistant>\n\n"
            else:
                formatted_prompt += f"{content}\n\n"
        formatted_prompt += "<assistant>\n"
        return formatted_prompt

# Singleton pour une utilisation facile dans tout le projet
llm_provider = LLMProvider()

# Exemple d'utilisation
async def example_usage():
    """Exemple d'utilisation du fournisseur LLM"""
    provider = LLMProvider()
    response = await provider.generate(
        messages=[
            {"role": "system", "content": "Vous êtes un assistant utile."},
            {"role": "user", "content": "Bonjour, comment ça va ?"}
        ]
    )
    return response

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())

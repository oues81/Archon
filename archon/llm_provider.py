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
    # D'abord essayer de charger depuis le profil
    profile_config = load_profile_config()
    if var_name in profile_config and profile_config[var_name] not in (None, ''):
        return str(profile_config[var_name])
    
    # Sinon, utiliser la valeur par défaut
    return os.getenv(var_name, default)

@dataclass
class LLMConfig:
    """Configuration pour le fournisseur LLM"""
    provider: str = field(default_factory=lambda: get_env_from_profile("LLM_PROVIDER", "Ollama").lower())
    model: str = field(default_factory=lambda: get_env_from_profile("LLM_MODEL", "qwen2.5:latest"))
    base_url: str = field(default_factory=lambda: get_env_from_profile("OLLAMA_API_BASE", "http://localhost:11434"))
    # Utiliser OPENROUTER_API_KEY pour OpenRouter, sinon OPENAI_API_KEY
    api_key: str = field(default_factory=lambda: get_env_from_profile(
        "OPENROUTER_API_KEY", 
        os.getenv("OPENAI_API_KEY", "")
    ))
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 300  # Timeout augmenté à 5 minutes pour les modèles lents sur CPU
    
    def __post_init__(self):
        """Met à jour la configuration à partir du profil actif"""
        self._update_from_profile()
    
    def _update_from_profile(self):
        """Met à jour la configuration à partir du profil actif"""
        profile_config = load_profile_config()
        if not profile_config:
            return
            
        # Mettre à jour les attributs avec les valeurs du profil
        for key, value in profile_config.items():
            if hasattr(self, key) and value not in (None, ''):
                # Convertir les types si nécessaire
                if key in ['temperature']:
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        continue
                elif key in ['max_tokens', 'timeout']:
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        continue
                        
                setattr(self, key, value)
        
        # Mettre à jour le fournisseur en minuscules
        if hasattr(self, 'provider'):
            self.provider = self.provider.lower()

class LLMProvider:
    """Fournisseur unifié pour différents modèles de langage"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._setup_provider()
    
    def _setup_provider(self):
        """Configure le fournisseur en fonction de la configuration"""
        self.provider = self.config.provider.lower()
        logger.info(f"Configuration du fournisseur: {self.provider}")
        
        try:
            if self.provider == "openrouter":
                # Vérifier que la clé API est définie
                if not self.config.api_key:
                    error_msg = "Clé API OpenRouter non configurée. Utilisation d'Ollama par défaut."
                    logger.warning(error_msg)
                    self.provider = "ollama"
                    self._setup_ollama()
                    return
                
                try:
                    # Importer et créer une nouvelle instance du client OpenRouter
                    from openrouter_client import OpenRouterClient, OpenRouterConfig
                    
                    # Créer une nouvelle configuration avec les paramètres actuels
                    config = OpenRouterConfig(
                        api_key=self.config.api_key,
                        default_model=self.config.model,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                    
                    # Créer une nouvelle instance du client avec la configuration
                    self.client = OpenRouterClient(config=config)
                    logger.info(f"Client OpenRouter configuré avec succès pour le modèle {self.config.model}")
                except Exception as e:
                    error_msg = f"Erreur lors de la configuration d'OpenRouter: {str(e)}. Utilisation d'Ollama par défaut."
                    logger.error(error_msg)
                    self.provider = "ollama"
                    self._setup_ollama()
                
            elif self.provider == "openai":
                # Configuration pour OpenAI v1.0.0+
                if not self.config.api_key:
                    logger.warning("Aucune clé API OpenAI configurée. Utilisation d'Ollama par défaut.")
                    self.provider = "ollama"
                    self._setup_ollama()
                else:
                    try:
                        from openai import OpenAI
                        self.client = OpenAI(api_key=self.config.api_key)
                        logger.info("Client OpenAI configuré avec succès")
                    except Exception as e:
                        error_msg = f"Erreur lors de la configuration d'OpenAI: {str(e)}. Utilisation d'Ollama par défaut."
                        logger.error(error_msg)
                        self.provider = "ollama"
                        self._setup_ollama()
            else:  # Ollama par défaut
                self._setup_ollama()
                
        except Exception as e:
            logger.error(f"Erreur lors de la configuration du fournisseur {self.provider}: {e}")
            # Basculer vers Ollama en cas d'erreur
            logger.info("Basculer vers Ollama comme fournisseur par défaut")
            self.provider = "ollama"
            self._setup_ollama()
    
    def _setup_ollama(self):
        """Configure le client Ollama"""
        self.client = None  # Ollama utilise directement des requêtes HTTP
        logger.info("Configuration d'Ollama comme fournisseur par défaut")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Génère une réponse à partir d'un ensemble de messages
        
        Args:
            messages: Liste des messages du chat
            model: Modèle à utiliser (optionnel)
            temperature: Température pour la génération
            max_tokens: Nombre maximum de tokens à générer
            **kwargs: Arguments supplémentaires pour le fournisseur
            
        Returns:
            Réponse du modèle avec le contenu généré et les métadonnées
        """
        # Mettre à jour la configuration à partir du profil actif
        self.config._update_from_profile()
        
        # Utiliser les paramètres fournis ou les valeurs par défaut de la configuration
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        logger.info(f"Génération avec le fournisseur {self.config.provider}, modèle {model}")
        
        try:
            if self.config.provider == "openrouter":
                return await self._generate_openrouter(messages, model, temperature, max_tokens, **kwargs)
            elif self.config.provider == "openai":
                return await self._generate_openai(messages, model, temperature, max_tokens, **kwargs)
            else:  # Ollama par défaut
                return await self._generate_ollama(messages, model, temperature, max_tokens, **kwargs)
        except Exception as e:
            logger.error(f"Erreur lors de la génération: {str(e)}")
            # Retourner une réponse par défaut en cas d'erreur
            return {
                "content": f"Désolé, une erreur s'est produite lors de la génération: {str(e)}",
                "model": model,
                "usage": {},
                "error": str(e)
            }
    
    async def _generate_openrouter(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Génère une réponse via l'API OpenRouter avec basculement automatique sur Ollama en cas d'échec"""
        try:
            # Vérifier que le client est correctement configuré
            if not hasattr(self, 'client') or not self.client:
                raise ValueError("Client OpenRouter non initialisé")
            
            # Vérifier que la clé API est définie
            if not self.config.api_key or self.config.api_key == "your_openrouter_api_key_here":
                raise ValueError("Clé API OpenRouter non configurée")
            
            # Effectuer l'appel à l'API
            response = await self.client.chat_completion(
                messages=messages,
                model=model or self.config.model,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                **kwargs
            )
            
            # Vérifier que la réponse contient les champs attendus
            if not response or "choices" not in response or not response["choices"]:
                error_msg = f"Réponse inattendue de l'API OpenRouter: {response}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Extraire le contenu de la réponse
            choice = response["choices"][0]
            if "message" not in choice:
                error_msg = f"Format de réponse inattenu: {choice}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            # Si on arrive ici, tout s'est bien passé
            return {
                "content": choice["message"].get("content", ""),
                "model": response.get("model", model or self.config.model),
                "usage": response.get("usage", {}),
                "raw_response": response
            }
            
        except Exception as e:
            error_msg = f"Erreur lors de l'appel à l'API OpenRouter: {str(e)}"
            logger.error(error_msg)
            
            # Essayer de basculer sur Ollama en cas d'erreur
            try:
                logger.warning("Tentative de basculement vers Ollama suite à une erreur OpenRouter...")
                self.provider = "ollama"
                self._setup_ollama()
                return await self._generate_ollama(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception as fallback_error:
                logger.error(f"Échec du basculement vers Ollama: {str(fallback_error)}")
                return {
                    "content": f"{error_msg}. Échec du basculement vers Ollama: {str(fallback_error)}",
                    "model": model or self.config.model,
                    "usage": {},
                    "error": str(e)
                }
    
    async def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Génère une réponse via l'API OpenAI v1.0.0+"""
        try:
            # Vérifier que le client est correctement configuré
            if not hasattr(self, 'client') or not self.client:
                raise ValueError("Client OpenAI non initialisé")
                
            # Utiliser asyncio pour exécuter la requête de manière asynchrone
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=model or self.config.model,
                    messages=messages,
                    temperature=temperature or self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                    **kwargs
                )
            )
            
            # Extraire la réponse (format OpenAI v1.0.0+)
            content = response.choices[0].message.content
            
            return {
                "content": content,
                "model": model or self.config.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if hasattr(response, 'usage') else {},
                "raw_response": response.model_dump() if hasattr(response, 'model_dump') else str(response)
            }
        except Exception as e:
            error_msg = f"Erreur lors de l'appel à l'API OpenAI: {str(e)}"
            logger.error(error_msg)
            
            # Essayer de basculer sur Ollama en cas d'erreur
            try:
                logger.warning("Tentative de basculement vers Ollama suite à une erreur OpenAI...")
                self.provider = "ollama"
                self._setup_ollama()
                return await self._generate_ollama(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception as fallback_error:
                logger.error(f"Échec du basculement vers Ollama: {str(fallback_error)}")
                return {
                    "content": f"{error_msg}. Échec du basculement vers Ollama: {str(fallback_error)}",
                    "model": model or self.config.model,
                    "usage": {},
                    "error": str(e)
                }
    
    async def _generate_ollama(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Génère une réponse via l'API Ollama avec gestion améliorée des erreurs"""
        model_to_use = model or self.config.model
        logger.info(f"Appel à Ollama avec le modèle {model_to_use}")
        
        try:
            host = os.getenv("OLLAMA_HOST", "host.docker.internal")
            ollama_url = f"http://{host}:11434"
            
            # Vérifier d'abord si le modèle est disponible
            async with httpx.AsyncClient() as client:
                # Vérifier si le serveur Ollama est en cours d'exécution
                try:
                    health_check = await client.get(f"{ollama_url}/api/tags", timeout=10.0)
                    health_check.raise_for_status()
                    
                    # Vérifier si le modèle est disponible
                    models_response = await client.get(f"{ollama_url}/api/tags")
                    models_response.raise_for_status()
                    available_models = [m["name"] for m in models_response.json().get("models", [])]
                    
                    if model_to_use not in available_models:
                        logger.warning(f"Modèle {model_to_use} non trouvé. Modèles disponibles: {', '.join(available_models)}")
                        # Essayer d'utiliser un modèle par défaut
                        fallback_model = "llama3:latest" if "llama3:latest" in available_models else available_models[0] if available_models else None
                        if fallback_model:
                            logger.info(f"Utilisation du modèle de secours: {fallback_model}")
                            model_to_use = fallback_model
                        else:
                            error_msg = f"Aucun modèle Ollama disponible. Veuillez en télécharger un avec 'ollama pull <modèle>'"
                            logger.error(error_msg)
                            return {
                                "content": error_msg,
                                "model": model_to_use,
                                "usage": {},
                                "error": error_msg
                            }
                except Exception as e:
                    error_msg = f"Impossible de se connecter au serveur Ollama à {ollama_url}: {str(e)}"
                    logger.error(error_msg)
                    return {
                        "content": error_msg,
                        "model": model_to_use,
                        "usage": {},
                        "error": error_msg
                    }
            
            # Préparer le payload avec gestion des messages
            formatted_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ["user", "assistant", "system"]:
                    formatted_messages.append({"role": role, "content": content})
            
            # Utiliser l'API chat plus moderne si disponible
            payload = {
                "model": model_to_use,
                "messages": formatted_messages,
                "stream": False,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "num_predict": max_tokens or self.config.max_tokens
                }
            }
            
            # Envoyer la requête avec un timeout raisonnable
            timeout = 300  # 5 minutes
            async with httpx.AsyncClient(timeout=timeout) as client:
                # Essayer d'abord avec l'API chat plus récente
                try:
                    response = await client.post(
                        f"{ollama_url}/api/chat",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    response.raise_for_status()
                    result = response.json()
                    
                    # Formater la réponse selon le format attendu
                    response_content = result.get("message", {}).get("content", "")
                    logger.info(f"Réponse d'Ollama reçue: {len(response_content)} caractères")
                    
                    return {
                        "content": response_content,
                        "model": result.get("model", model_to_use),
                        "usage": {
                            "total_tokens": len(response_content.split()),
                            "prompt_tokens": 0,  # Non fourni par l'API Ollama
                            "completion_tokens": 0  # Non fourni par l'API Ollama
                        },
                        "raw_response": result
                    }
                    
                except (httpx.HTTPStatusError, json.JSONDecodeError):
                    # En cas d'échec, essayer avec l'ancienne API generate
                    try:
                        logger.info("L'API chat a échoué, tentative avec l'API generate...")
                        old_payload = {
                            "model": model_to_use,
                            "prompt": self._format_messages_for_ollama(messages),
                            "stream": False,
                            "options": {
                                "temperature": temperature or self.config.temperature,
                                "num_predict": max_tokens or self.config.max_tokens
                            }
                        }
                        
                        response = await client.post(
                            f"{ollama_url}/api/generate",
                            json=old_payload,
                            headers={"Content-Type": "application/json"}
                        )
                        response.raise_for_status()
                        result = response.json()
                        
                        response_content = result.get("response", "")
                        logger.info(f"Réponse d'Ollama (legacy) reçue: {len(response_content)} caractères")
                        
                        return {
                            "content": response_content,
                            "model": result.get("model", model_to_use),
                            "usage": {"total_tokens": len(response_content.split())},
                            "raw_response": result
                        }
                        
                    except Exception as inner_e:
                        error_msg = f"Erreur avec l'API generate: {str(inner_e)}"
                        logger.error(error_msg)
                        raise Exception(f"Échec des deux API Ollama (chat et generate): {error_msg}")
            
        except httpx.HTTPStatusError as e:
            error_msg = f"Erreur HTTP {e.response.status_code} lors de l'appel à Ollama"
            try:
                error_data = e.response.json()
                error_msg = f"{error_msg}: {error_data.get('error', 'Détails non disponibles')}"
            except:
                pass
            logger.error(error_msg)
            raise Exception(error_msg) from e
            
        except httpx.RequestError as e:
            error_msg = f"Erreur de requête vers Ollama: {str(e)}"
            logger.error(error_msg)
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
        
        # Ajouter un marqueur pour la réponse de l'assistant
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

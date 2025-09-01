"""
Fabrique pour les modèles LLM.
Permet de créer des instances de modèles en fonction du fournisseur configuré.
"""
from typing import Dict, Any, Optional, Union, List
import os
import json
import httpx
from dataclasses import dataclass

# Import des modèles Ollama depuis le bon paquet
try:
    # Import via paquet exposé (grâce à l'alias dans archon.archon.models.__init__)
    from k.core.models.ollama_model import OllamaModel
except Exception:
    # Fallback relatif direct si nécessaire
    from ..archon.ollama_model import OllamaModel  # type: ignore

# Import ModelConfig depuis l'emplacement réel
try:
    from k.config.model_config import ModelConfig
except Exception:
    from ..archon.config.model_config import ModelConfig  # type: ignore

@dataclass
class OpenRouterResponse:
    """Classe pour représenter la réponse d'OpenRouter"""
    content: str
    model: str
    usage: Dict[str, int]
    raw_response: Dict[str, Any]

class OpenRouterClient:
    """Client léger pour interagir avec l'API OpenRouter"""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/oues/archon",
            "X-Title": "Archon AI Agent",
            "Content-Type": "application/json"
        }
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> OpenRouterResponse:
        """Génère une réponse à partir d'une liste de messages"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            return OpenRouterResponse(
                content=data["choices"][0]["message"]["content"],
                model=data.get("model", self.model),
                usage=data.get("usage", {}),
                raw_response=data
            )

class ModelFactory:
    """Fabrique pour créer des instances de modèles LLM."""
    
    @staticmethod
    def create_model(config: ModelConfig):
        """Crée une instance de modèle en fonction de la configuration.
        
        Pour OpenRouter, le model_name doit être au format 'provider/model' (sans le préfixe 'openrouter:')
        La clé API OpenRouter doit être définie dans la variable d'environnement OPENROUTER_API_KEY.
        """
        if config.provider == 'openrouter':
            # Vérifier que la clé API est définie
            if not config.api_key:
                raise ValueError("La clé API OpenRouter n'est pas définie")
                
            # Nettoyer le nom du modèle
            model_name = config.model_name
            if model_name.startswith('openrouter:'):
                model_name = model_name[len('openrouter:'):]
                
            # Créer et retourner une instance du client OpenRouter
            return OpenRouterClient(
                api_key=config.api_key,
                model=model_name
            )
            
        else:  # Ollama par défaut
            return OllamaModel(
                model_name=config.model_name,
                base_url=config.base_url,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
    
    @classmethod
    def from_env(cls, provider: str = None):
        """Crée un modèle à partir du profil courant défini dans env_vars.json.

        Sélectionner un modèle via des variables d'environnement directes est
        déprécié. Nous lisons le profil courant depuis la configuration et
        construisons la configuration modèle à partir de ce profil.
        """
        from k.config.config import load_config
        cfg = load_config() or {}
        current_profile = cfg.get('current_profile')
        if not current_profile:
            raise ValueError("Aucun 'current_profile' défini dans env_vars.json")
        config = ModelConfig.from_profile(current_profile)
        return cls.create_model(config)

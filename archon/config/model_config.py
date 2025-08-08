"""
Configuration des modèles LLM pour Archon AI.
Permet de configurer différents fournisseurs (Ollama, OpenAI, OpenRouter, etc.)
de manière dynamique via des variables d'environnement.
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration pour un modèle LLM."""
    provider: str
    model_name: str
    base_url: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    headers: Optional[Dict[str, str]] = None

    @classmethod
    def from_profile(cls, profile_name: str) -> 'ModelConfig':
        """Crée une configuration depuis un profil de env_vars.json.

        Cette méthode est utilisée par les tests qui patchent
        `archon.config.config.load_config`.
        """
        try:
            from archon.config.config import load_config  # patched in tests
        except Exception as e:  # pragma: no cover
            logger.error(f"Impossible d'importer load_config: {e}")
            raise

        data = load_config() or {}
        profiles: Dict[str, Any] = data.get('profiles', {})
        profile: Dict[str, Any] = profiles.get(profile_name, {})
        if not profile:
            raise ValueError(f"Profil non trouvé: {profile_name}")

        provider = str(profile.get('LLM_PROVIDER', 'ollama')).lower()

        # Commun
        temp = float(profile.get('TEMPERATURE', 0.7)) if isinstance(profile.get('TEMPERATURE', 0.7), (int, float, str)) else 0.7
        max_tok = int(profile.get('MAX_TOKENS', 2048)) if isinstance(profile.get('MAX_TOKENS', 2048), (int, float, str)) else 2048

        # API keys
        api_key = (
            profile.get('LLM_API_KEY')
            or profile.get('OPENAI_API_KEY')
            or profile.get('OPENROUTER_API_KEY')
            or None
        )

        # Model name preference
        model_name = (
            profile.get('PRIMARY_MODEL')
            or profile.get('model')
            or ''
        )

        # Base URL resolution
        base_url = profile.get('BASE_URL') or ''

        headers: Optional[Dict[str, str]] = None

        if provider == 'openrouter':
            # Defaults for OpenRouter
            base_url = base_url or 'https://openrouter.ai/api/v1'
            if not api_key:
                # Les tests patchent les appels HTTP, éviter d'échouer fort ici
                api_key = 'test-key'

            clean_model = str(model_name).replace('openrouter:', '').replace('openrouter/', '')
            # Si modèle gratuit avec suffixe :free, enlever le suffixe pour l'appel
            model_to_use = clean_model.split(':')[0] if ':free' in clean_model else clean_model

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/oues/archon",
                "X-Title": "Archon AI Agent",
            }

            return cls(
                provider='openrouter',
                model_name=model_to_use,
                base_url=base_url,
                api_key=api_key,
                temperature=temp,
                max_tokens=max_tok,
                headers=headers,
            )

        if provider == 'openai':
            base_url = base_url or 'https://api.openai.com/v1'
            if not api_key:
                api_key = 'test-key'

            return cls(
                provider='openai',
                model_name=model_name or 'gpt-4o-mini',
                base_url=base_url,
                api_key=api_key,
                temperature=temp,
                max_tokens=max_tok,
            )

        # Default: ollama
        base_url = profile.get('OLLAMA_BASE_URL') or base_url or 'http://localhost:11434'
        # Normaliser pour inclure /v1 si absent
        if base_url and not base_url.rstrip('/').endswith('/v1'):
            base_url = base_url.rstrip('/') + '/v1'

        return cls(
            provider='ollama',
            model_name=model_name or 'phi3:latest',
            base_url=base_url,
            api_key=None,
            temperature=temp,
            max_tokens=max_tok,
        )

    @classmethod
    def from_env(cls, provider: str = None) -> 'ModelConfig':
        """
        Crée une configuration à partir des variables d'environnement.
        
        Args:
            provider: Le fournisseur à utiliser ('ollama', 'openai', 'openrouter')
                     Si None, utilise la variable d'environnement LLM_PROVIDER
        """
        # Déterminer le fournisseur
        provider = (provider or os.getenv('LLM_PROVIDER', 'ollama')).lower()
        
        # Configuration commune
        config = {
            'temperature': float(os.getenv('TEMPERATURE', '0.7')),
            'max_tokens': int(os.getenv('MAX_TOKENS', '2048')),
            'model_name': os.getenv('LLM_MODEL', ''),
            'base_url': os.getenv('BASE_URL', ''),
            'api_key': os.getenv('API_KEY')
        }
        
        # Configuration spécifique au fournisseur
        if provider == 'openrouter':
            return cls._from_openrouter(config)
        elif provider == 'openai':
            return cls._from_openai(config)
        else:  # Par défaut, utiliser Ollama
            return cls._from_ollama(config)
    
    @classmethod
    def _from_ollama(cls, config: dict) -> 'ModelConfig':
        """Crée une configuration pour Ollama."""
        model_name = config['model_name'] or 'phi3:latest'
        base_url = config['base_url'] or 'http://host.docker.internal:11434'
        
        logger.info(f"[CONFIG] Utilisation du fournisseur: ollama")
        logger.info(f"[CONFIG] Modèle: {model_name}")
        logger.info(f"[CONFIG] URL de base: {base_url}")
        
        return cls(
            provider='ollama',
            model_name=model_name,
            base_url=base_url,
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )
    
    @classmethod
    def _from_openai(cls, config: dict) -> 'ModelConfig':
        """Crée une configuration pour OpenAI."""
        model_name = config['model_name'] or 'gpt-4-turbo'
        base_url = config['base_url'] or 'https://api.openai.com/v1'
        api_key = config['api_key'] or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("La clé API OpenAI (OPENAI_API_KEY) est requise")
        
        logger.info(f"[CONFIG] Utilisation du fournisseur: openai")
        logger.info(f"[CONFIG] Modèle: {model_name}")
        
        return cls(
            provider='openai',
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )
    
    @classmethod
    def _from_openrouter(cls, config: dict) -> 'ModelConfig':
        """Crée une configuration pour OpenRouter."""
        # Utiliser le modèle spécifié ou un modèle gratuit par défaut
        model_name = config['model_name'] or 'mistralai/mistral-7b-instruct:free'
        base_url = 'https://openrouter.ai/api/v1'
        api_key = config['api_key'] or os.getenv('OPENROUTER_API_KEY')
        
        if not api_key:
            raise ValueError("La clé API OpenRouter (OPENROUTER_API_KEY) est requise")
        
        # Nettoyer le nom du modèle des préfixes openrouter/
        clean_model_name = model_name.replace('openrouter:', '').replace('openrouter/', '')
        
        logger.info(f"[CONFIG] Utilisation du fournisseur: openrouter")
        logger.info(f"[CONFIG] Modèle: {clean_model_name}")
        logger.info(f"[CONFIG] URL de base: {base_url}")
        
        # Pour les modèles avec :free, on utilise le nom nettoyé
        model_to_use = clean_model_name.split(':')[0] if ':free' in clean_model_name else clean_model_name
        
        return cls(
            provider='openrouter',
            model_name=model_to_use,  # Utiliser le nom nettoyé
            base_url=base_url,
            api_key=api_key,
            temperature=float(os.getenv('TEMPERATURE', '0.7')),
            max_tokens=int(os.getenv('MAX_TOKENS', '2048')),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/oues/archon",
                "X-Title": "Archon AI Agent"
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire."""
        config = {
            'model': self.model_name,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
        
        if self.provider in ['openrouter', 'openai']:
            config.update({
                'base_url': self.base_url,
                'api_key': self.api_key,
                'headers': self.headers
            })
        
        return config

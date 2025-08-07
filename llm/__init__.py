"""
Unified LLM Provider for Archon
Supports multiple LLM providers with dynamic profile switching
"""
import logging
import os
import sys
import importlib
from typing import Optional, Dict, Any, Type, TypeVar
from dataclasses import dataclass, asdict
from pathlib import Path
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variable for provider classes
T = TypeVar('T', bound='LLMProvider')

class ProfileConfig:
    """Configuration pour un profil spécifique"""
    def __init__(self, profile_data: Dict[str, Any]):
        self.provider = profile_data.get('provider', 'openai')
        self.api_key = profile_data.get('api_key')
        self.base_url = profile_data.get('base_url')
        self.model = profile_data.get('model', 'gpt-4')
        self.timeout = int(profile_data.get('timeout', 30))
        
        # Configurations spécifiques au fournisseur
        self.provider_config = profile_data.get('config', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire"""
        return {
            'provider': self.provider,
            'api_key': self.api_key,
            'base_url': self.base_url,
            'model': self.model,
            'timeout': self.timeout,
            'config': self.provider_config
        }

def get_config_path() -> Path:
    """Retourne le chemin du fichier de configuration des profils"""
    # Chemin par défaut si la variable d'environnement n'est pas définie
    default_path = Path(__file__).parent.parent / 'workbench' / 'env_vars.json'
    return Path(os.getenv('ARCHON_CONFIG', str(default_path)))

def load_config() -> Dict[str, Any]:
    """Charge la configuration complète"""
    config_path = get_config_path()
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
        return {"profiles": {}, "current_profile": None}

def load_profile(profile_name: str) -> Optional[Dict[str, Any]]:
    """Charge la configuration d'un profil spécifique"""
    config = load_config()
    return config.get('profiles', {}).get(profile_name)

@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: str = "openai"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o-mini"
    timeout: int = 30
    
    # Modèles spécifiques
    reasoner_model: str = "gpt-4o-mini"
    primary_model: str = "gpt-4o-mini"
    coder_model: Optional[str] = "gpt-4o-nano"
    advisor_model: Optional[str] = "gpt-4o-mini"
    
    @classmethod
    def from_profile(cls, profile_name: str) -> 'LLMConfig':
        """Crée une configuration à partir d'un profil"""
        profile_data = load_profile(profile_name)
        if not profile_data:
            raise ValueError(f"Profil non trouvé: {profile_name}")
        
        # Mappage des champs de l'ancienne structure vers la nouvelle
        provider = profile_data.get('LLM_PROVIDER', 'openai').lower()
        api_key = profile_data.get('LLM_API_KEY') or profile_data.get('OPENAI_API_KEY')
        base_url = profile_data.get('BASE_URL')
        
        # Si c'est Ollama et qu'on a une URL de base spécifique
        if provider == 'ollama' and 'OLLAMA_BASE_URL' in profile_data:
            base_url = profile_data['OLLAMA_BASE_URL']
            # S'assurer que l'URL se termine par /v1 pour la compatibilité
            if base_url and not base_url.endswith('/v1'):
                base_url = base_url.rstrip('/') + '/v1'
        
        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=profile_data.get('PRIMARY_MODEL', 'gpt-4'),
            reasoner_model=profile_data.get('REASONER_MODEL', profile_data.get('PRIMARY_MODEL', 'gpt-4')),
            primary_model=profile_data.get('PRIMARY_MODEL', 'gpt-4'),
            coder_model=profile_data.get('CODER_MODEL'),
            advisor_model=profile_data.get('ADVISOR_MODEL')
        )

class LLMProvider:
    """Unified LLM Provider that supports multiple backends with profile switching"""
    _instance = None
    _initialized = False
    
    def __new__(cls: Type[T]) -> T:
        """Implémente le pattern Singleton"""
        if cls._instance is None:
            cls._instance = super(LLMProvider, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialise le fournisseur LLM"""
        if not self._initialized:
            self.config: Optional[LLMConfig] = None
            self._current_profile: Optional[str] = None
            self._initialize_from_profile()
            self._initialized = True
    
    def _initialize_from_profile(self, profile_name: Optional[str] = None):
        """Initialize provider configuration from specified or current profile"""
        try:
            if not profile_name:
                # Essayer de charger le profil depuis les variables d'environnement
                profile_name = os.getenv('ARCHON_PROFILE', 'default')
                
                # Vérifier si le profil existe dans la configuration
                config_path = get_config_path()
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        if 'current_profile' in config and config['current_profile']:
                            profile_name = config['current_profile']
            
            # Mettre à jour le profil courant
            self._current_profile = profile_name
            logger.info(f"Initializing LLM Provider with profile: {profile_name}")
            
            # Charger la configuration du profil
            self.config = LLMConfig.from_profile(profile_name)
            
            # Initialiser le fournisseur spécifique
            provider_method = getattr(self, f"_initialize_{self.config.provider.lower()}", None)
            if provider_method and callable(provider_method):
                provider_method()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
            
            logger.info(f"Successfully initialized {self.config.provider} provider")
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            # Fallback à une configuration par défaut
            self.config = LLMConfig()
            logger.warning("Falling back to default configuration")
            
    def reload_profile(self, profile_name: Optional[str] = None) -> bool:
        """Recharge la configuration à partir d'un profil"""
        try:
            self._initialize_from_profile(profile_name)
            return True
        except Exception as e:
            logger.error(f"Failed to reload profile: {e}")
            return False
    
    def _initialize_ollama(self):
        """Initialize Ollama configuration"""
        if not self.config.base_url:
            self.config.base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            
        # Définir les modèles par défaut si non spécifiés
        if not self.config.model:
            self.config.model = "llama2"
        if not self.config.reasoner_model:
            self.config.reasoner_model = self.config.model
        if not self.config.primary_model:
            self.config.primary_model = self.config.model
            
        logger.info(f"Initialized Ollama provider with model: {self.config.model}")
        
        # Vérifier la connectivité
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Vérifie la connexion au serveur Ollama"""
        try:
            import requests
            # Vérifier si l'URL contient déjà /v1 ou se termine par un slash
            base_url = self.config.base_url.rstrip('/')
            if '/v1' in base_url:
                # Si l'URL contient déjà /v1, on l'enlève pour éviter les doublons
                base_url = base_url.replace('/v1', '')
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.debug("Successfully connected to Ollama server")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama server: {e}")
            return False
            
    def _initialize_openrouter(self):
        """Initialize OpenRouter configuration"""
        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENROUTER_API_KEY")
            
        if not self.config.api_key:
            raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or in profile config.")
            
        if not self.config.base_url:
            self.config.base_url = "https://openrouter.ai/api/v1"
            
        # Définir les modèles par défaut si non spécifiés
        if not self.config.model:
            self.config.model = "anthropic/claude-2"
        if not self.config.reasoner_model:
            self.config.reasoner_model = self.config.model
        if not self.config.primary_model:
            self.config.primary_model = self.config.model
            
        logger.info(f"Initialized OpenRouter provider with model: {self.config.model}")
    
    def _initialize_openai(self):
        """Initialize OpenAI configuration"""
        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENAI_API_KEY")
            
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or in profile config.")
            
        if not self.config.base_url:
            self.config.base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            
        # Définir les modèles par défaut si non spécifiés
        if not self.config.model:
            self.config.model = "gpt-4"
        if not self.config.reasoner_model:
            self.config.reasoner_model = self.config.model
        if not self.config.primary_model:
            self.config.primary_model = self.config.model
            
        logger.info(f"Initialized OpenAI provider with model: {self.config.model}")
        
        # Masquer la clé API dans les logs
        if self.config.api_key:
            masked_key = self.config.api_key[:4] + "*****" + self.config.api_key[-4:]
            logger.debug(f"Using API key: {masked_key}")
    
    def reload_config(self, profile_name: Optional[str] = None):
        """
        Recharge la configuration à partir du profil spécifié ou du profil courant
        
        Args:
            profile_name: Nom du profil à charger. Si None, recharge le profil courant.
        """
        logger.info(f"Reloading LLM configuration for profile: {profile_name or 'current'}")
        self._initialize_from_profile(profile_name)
        
    def get_current_profile(self) -> Optional[str]:
        """Retourne le nom du profil actuellement chargé"""
        return self._current_profile
        
    def get_available_models(self) -> Dict[str, Any]:
        """Retourne la liste des modèles disponibles pour le fournisseur actuel"""
        if not self.config:
            return {}
            
        models = {
            'default': self.config.model,
            'reasoner': self.config.reasoner_model,
            'primary': self.config.primary_model
        }
        
        if self.config.coder_model:
            models['coder'] = self.config.coder_model
        if self.config.advisor_model:
            models['advisor'] = self.config.advisor_model
            
        return models
    
    def get_model_for_agent(self, agent_type: str) -> str:
        """Get the appropriate model for a specific agent type"""
        if not self.config:
            raise ValueError("LLM Provider not initialized")
        
        if agent_type == "reasoner":
            return self.config.reasoner_model
        elif agent_type == "coder":
            return self.config.coder_model or self.config.primary_model
        elif agent_type == "advisor":
            return self.config.advisor_model or self.config.primary_model
        else:
            return self.config.primary_model
            
    def _generate(self, messages: list, model: Optional[str] = None, **kwargs) -> str:
        """
        Generate text using the configured LLM provider
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            model: Optional model to use (overrides the configured model)
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            Generated text response
            
        Raises:
            ValueError: If the provider is not properly configured
            Exception: For any errors during generation
        """
        if not self.config:
            raise ValueError("LLM Provider not initialized")
            
        # Use specified model or fall back to configured model
        model_to_use = model or self.config.model
        
        try:
            if self.config.provider == 'ollama':
                return self._generate_with_ollama(messages, model_to_use, **kwargs)
            elif self.config.provider == 'openai':
                return self._generate_with_openai(messages, model_to_use, **kwargs)
            elif self.config.provider == 'openrouter':
                return self._generate_with_openrouter(messages, model_to_use, **kwargs)
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
        except Exception as e:
            logger.error(f"Error generating text with {self.config.provider}: {str(e)}")
            raise
    
    def _generate_with_ollama(self, messages: list, model: str, **kwargs) -> str:
        """Generate text using Ollama API"""
        import requests
        import json
        
        if not self.config.base_url:
            raise ValueError("Ollama base URL is not configured")
            
        url = f"{self.config.base_url}/chat/completions"
        
        # Format messages for Ollama API
        formatted_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]
        
        payload = {
            "model": model,
            "messages": formatted_messages,
            "stream": False
        }
        
        # Add any additional parameters
        payload.update(kwargs)
        
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract the generated text from the response
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise ValueError("Unexpected response format from Ollama API")
    
    def _generate_with_openai(self, messages: list, model: str, **kwargs) -> str:
        """Generate text using OpenAI API"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package is required. Install with 'pip install openai'")
        
        if not self.config.api_key:
            raise ValueError("OpenAI API key is not configured")
            
        client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or "https://api.openai.com/v1"
        )
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    def _generate_with_openrouter(self, messages: list, model: str, **kwargs) -> str:
        """Generate text using OpenRouter API"""
        import requests
        
        if not self.config.api_key:
            raise ValueError("OpenRouter API key is not configured")
            
        url = self.config.base_url or "https://openrouter.ai/api/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        # Add any additional parameters
        payload.update(kwargs)
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.config.timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            raise ValueError("Unexpected response format from OpenRouter API")

def initialize_llm_provider() -> LLMProvider:
    """Initialize and return a new LLM provider instance"""
    return LLMProvider()

# Global instance
llm_provider: Optional[LLMProvider] = None

def get_llm_provider() -> LLMProvider:
    """Get or create the global LLM provider instance"""
    global llm_provider
    if llm_provider is None:
        llm_provider = initialize_llm_provider()
    return llm_provider

# Initialize on import
try:
    llm_provider = initialize_llm_provider()
    if llm_provider and llm_provider.config:
        logger.info(f"🚀 LLM Provider initialized with {llm_provider.config.provider}")
    else:
        logger.warning("⚠️ LLM Provider initialized but config is None")
except Exception as e:
    logger.error(f"❌ Failed to initialize LLM provider on import: {e}")
    llm_provider = None

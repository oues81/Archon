"""
Unified LLM Provider for Archon
Supports multiple LLM providers: Ollama, OpenRouter, and OpenAI
"""
import logging
import os
import sys
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# Initialize logger FIRST before any usage
logger = logging.getLogger(__name__)

# Add utils directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))  # /app/src/archon/archon/llm
archon_dir = os.path.dirname(current_dir)                  # /app/src/archon/archon
src_archon_dir = os.path.dirname(archon_dir)              # /app/src/archon
utils_dir = os.path.join(src_archon_dir, "utils")         # /app/src/archon/utils
sys.path.insert(0, utils_dir)

try:
    from utils import get_env_var, get_current_profile, write_to_log
    logger.info("✅ Successfully imported utils functions")
except ImportError as e:
    logger.error(f"❌ Failed to import utils functions: {e}")
    # Fallback if import fails
    def get_env_var(var_name: str, profile: Optional[str] = None) -> Optional[str]:
        return os.environ.get(var_name)
    
    def get_current_profile() -> str:
        # CORRECTION: Read from env_vars.json at correct location
        try:
            workbench_dir = "/app/src/archon/workbench"
            env_file_path = os.path.join(workbench_dir, "env_vars.json")
            if os.path.exists(env_file_path):
                import json
                with open(env_file_path, "r") as f:
                    env_vars = json.load(f)
                    return env_vars.get("current_profile", "ollama_default")
        except Exception as e:
            print(f"[FALLBACK ERROR] Failed to read profile: {e}")
        return "ollama_default"
    
    def write_to_log(message: str):
        print(f"[LOG] {message}")

@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    reasoner_model: str = "tinyllama:1.1b"
    primary_model: str = "tinyllama:1.1b"
    coder_model: Optional[str] = None
    advisor_model: Optional[str] = None
    http_headers: Dict[str, str] = field(default_factory=dict)

class LLMProvider:
    """Unified LLM Provider that supports multiple backends"""
    
    def __init__(self, profile_name: Optional[str] = None):
        self.config: Optional[LLMConfig] = None
        self._initialize_from_profile(profile_name)
    
    def _initialize_from_profile(self, profile_name: Optional[str] = None):
        """Initialize provider configuration from current profile"""
        try:
            logger.info(f"🔧 Starting LLM Provider initialization with profile_name: {profile_name}")
            profile_to_use = profile_name or get_current_profile()
            logger.info(f"🔧 Using profile: {profile_to_use}")
            write_to_log(f"🔧 Initializing LLM Provider with profile: {profile_to_use}")
            
            # Get provider type
            provider = get_env_var("LLM_PROVIDER", profile_to_use)
            logger.info(f"🔍 Retrieved LLM_PROVIDER: {provider}")
            if not provider:
                # Try alternative names
                provider = get_env_var("PROVIDER", profile_to_use)
                logger.info(f"🔍 Retrieved PROVIDER (fallback): {provider}")
            
            if not provider:
                logger.warning("❌ No LLM provider specified, defaulting to OpenRouter")
                provider = "OpenRouter"
            
            # Normalize provider name
            provider = provider.lower()
            
            if provider == "ollama":
                self._initialize_ollama()
            elif provider == "openrouter":
                self._initialize_openrouter()
            elif provider == "openai":
                self._initialize_openai()
            else:
                logger.error(f"❌ Unsupported provider: {provider}")
                raise ValueError(f"Unsupported LLM provider: {provider}")
            
            logger.info(f"✅ LLM Provider '{provider}' initialized successfully")
            write_to_log(f"✅ LLM Provider '{provider}' initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM provider: {e}")
            write_to_log(f"❌ Failed to initialize LLM provider: {e}")
            # Create a fallback config to prevent crashes
            self.config = LLMConfig(
                provider="openrouter",
                reasoner_model="deepseek/deepseek-chat-v3-0324:free",
                primary_model="moonshotai/kimi-k2:free"
            )
    
    def _initialize_ollama(self):
        """Initialize Ollama configuration"""
        base_url = get_env_var("OLLAMA_BASE_URL") or "http://ollama:11434"
        reasoner_model = get_env_var("REASONER_MODEL") or "phi3:latest"
        primary_model = get_env_var("PRIMARY_MODEL") or "phi3:latest"
        
        self.config = LLMConfig(
            provider="ollama",
            base_url=base_url,
            reasoner_model=reasoner_model,
            primary_model=primary_model
        )
        
        logger.info(f"🦙 Ollama configured: {base_url}")
        logger.info(f"🧠 Reasoner model: {reasoner_model}")
        logger.info(f"⚡ Primary model: {primary_model}")
    
    def _normalize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Normalize HTTP headers by ensuring all header names and values are ASCII-only.
        
        Args:
            headers: Dictionary of headers to normalize
            
        Returns:
            Dictionary with normalized headers (ASCII-only)
        """
        if not headers:
            return {}
            
        normalized = {}
        for k, v in headers.items():
            if not k or not isinstance(k, str) or not isinstance(v, (str, bytes, int, float)):
                logger.warning(f"⚠️ Skipping invalid header: {k}={v}")
                continue
                
            try:
                # Ensure key is a string
                k_str = str(k) if not isinstance(k, str) else k
                
                # Normalize key - convert to ASCII and replace non-ASCII with '_'
                nk = k_str.encode('ascii', errors='replace').decode('ascii')
                nk = ''.join(c if c.isascii() and c.isprintable() else '_' for c in nk)
                
                # Ensure value is a string
                v_str = str(v) if not isinstance(v, str) else v
                
                # Normalize value - convert to ASCII and replace non-ASCII with '?'
                nv = v_str.encode('ascii', errors='replace').decode('ascii')
                nv = ''.join(c if c.isascii() and c.isprintable() else '?' for c in nv)
                
                # Log if normalization changed the values
                if nk != k_str or nv != v_str:
                    logger.debug(f"Normalized header: '{k_str}' -> '{nk}', '{v_str}' -> '{nv}'")
                    
                normalized[nk] = nv
                
            except Exception as e:
                logger.error(f"❌ Failed to normalize header {k}={v}: {str(e)}", exc_info=True)
                continue
                
        return normalized

    def _initialize_openrouter(self):
        """Initialize OpenRouter configuration"""
        api_key = get_env_var("OPENROUTER_API_KEY")
        if not api_key:
            logger.warning("⚠️ OPENROUTER_API_KEY not found, trying LLM_API_KEY")
            api_key = get_env_var("LLM_API_KEY")
        
        if not api_key:
            raise ValueError("❌ OpenRouter API key is required but not found")
        
        reasoner_model = get_env_var("REASONER_MODEL") or "deepseek/deepseek-chat-v3-0324:free"
        primary_model = get_env_var("PRIMARY_MODEL") or "moonshotai/kimi-k2:free"
        coder_model = get_env_var("CODER_MODEL") or "qwen/qwen3-coder:free"
        advisor_model = get_env_var("ADVISOR_MODEL") or "meta-llama/llama-3.1-8b-instruct:free"
        
        # Create normalized headers
        headers = self._normalize_headers({
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Archon"
        })
        
        self.config = LLMConfig(
            provider="openrouter",
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            reasoner_model=reasoner_model,
            primary_model=primary_model,
            coder_model=coder_model,
            advisor_model=advisor_model,
            http_headers=headers
        )
        
        # Mask API key for logging
        masked_key = api_key[:6] + "*****" + api_key[-4:] if len(api_key) > 10 else "***"
        logger.info(f"🔑 OpenRouter configured with key: {masked_key}")
        logger.info(f"🧠 Reasoner model: {reasoner_model}")
        logger.info(f"⚡ Primary model: {primary_model}")
    
    def _initialize_openai(self):
        """Initialize OpenAI configuration"""
        api_key = get_env_var("OPENAI_API_KEY")
        if not api_key:
            logger.warning("⚠️ OPENAI_API_KEY not found, trying LLM_API_KEY")
            api_key = get_env_var("LLM_API_KEY")
        
        if not api_key:
            raise ValueError("❌ OpenAI API key is required but not found")
        
        reasoner_model = get_env_var("REASONER_MODEL") or "gpt-4o-mini"
        primary_model = get_env_var("PRIMARY_MODEL") or "gpt-4o-mini"
        coder_model = get_env_var("CODER_MODEL") or "gpt-4o-mini"
        advisor_model = get_env_var("ADVISOR_MODEL") or "gpt-4o-mini"
        
        # Create normalized headers
        headers = self._normalize_headers({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        
        self.config = LLMConfig(
            provider="openai",
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            reasoner_model=reasoner_model,
            primary_model=primary_model,
            coder_model=coder_model,
            advisor_model=advisor_model,
            http_headers=headers
        )
        
        # Mask API key for logging
        masked_key = api_key[:6] + "*****" + api_key[-4:] if len(api_key) > 10 else "***"
        logger.info(f"🔑 OpenAI configured with key: {masked_key}")
        logger.info(f"🧠 Reasoner model: {reasoner_model}")
        logger.info(f"⚡ Primary model: {primary_model}")
    
    def reload_config(self):
        """Reload configuration from current profile"""
        self._initialize_from_profile()
    
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

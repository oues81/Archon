"""
Utilities for Archon project
Handles environment variables, profile management, and logging
"""

import json
import logging
import os
import re
import importlib
import webbrowser
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
# Optional dependency: streamlit (not required in server-only containers)
try:
    import streamlit as st
except Exception:
    st = None

# Imports nécessaires pour la fonction get_clients (optionnels, chargés paresseusement)
try:
    from openai import AsyncOpenAI  # Peut être absent selon l'environnement
except ImportError:
    AsyncOpenAI = None

try:
    # Modern supabase-py v2
    from supabase import create_client as _supabase_create_client  # type: ignore
except ImportError:
    _supabase_create_client = None  # type: ignore

try:
    # Legacy style (may not exist in v2)
    from supabase import Client as _SupabaseClient  # type: ignore
except ImportError:
    _SupabaseClient = None  # type: ignore

try:
    from k.core.utils.neo4j_client import Neo4jClient
except ImportError:
    # Fallback si Neo4jClient n'existe pas
    class Neo4jClient:
        def __init__(self, uri, username, password, database):
            self.uri = uri
            self.username = username
            self.password = password
            self.database = database
        def close(self):
            pass

logger = logging.getLogger(__name__)

def write_to_log(message: str, level: str = 'info'):
    """Writes a message to the appropriate log level."""
    if level == 'info':
        logging.info(message)
    elif level == 'warning':
        logging.warning(message)
    elif level == 'error':
        logging.error(message)
    elif level == 'debug':
        logging.debug(message)
    else:
        logging.info(message)

def get_bool_env(var_name: str, default: bool = False) -> bool:
    """Return a boolean environment/profile value with common truthy parsing.

    Accepts values like 1/0, true/false, yes/no (case-insensitive).
    """
    val = get_env_var(var_name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

def validate_rag_env() -> bool:
    """Validate required environment for RAG when enabled. Returns True if OK.

    We require at minimum: SUPABASE_URL, SUPABASE_SERVICE_KEY, EMBEDDING_PROVIDER, EMBEDDING_API_KEY.
    """
    required = [
        ("SUPABASE_URL", get_env_var("SUPABASE_URL")),
        ("SUPABASE_SERVICE_KEY", get_env_var("SUPABASE_SERVICE_KEY")),
        ("EMBEDDING_PROVIDER", get_env_var("EMBEDDING_PROVIDER")),
        ("EMBEDDING_API_KEY", get_env_var("EMBEDDING_API_KEY")),
    ]
    missing = [k for k, v in required if not v]
    if missing:
        logger.warning(f"RAG enabled but missing required vars: {missing}")
        return False
    return True

def configure_logging() -> dict:
    """Configure application logging once, with optional file and JSON formatting.

    Env vars:
    - LOG_LEVEL: logging level (default INFO)
    - LOG_TO_FILE: enable file logging (true/false)
    - LOG_FILE_PATH: file path for logs (default ./logs/archon.log)
    - LOG_JSON: output logs in JSON (true/false)

    Returns a summary dict of active sinks.
    """
    import sys
    import json as _json
    from logging import StreamHandler, Formatter
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger()
    summary = {"console": True, "file": False, "file_path": None, "json": False}

    # Idempotent guard
    if getattr(root, "_archon_logging_configured", False):
        return getattr(root, "_archon_logging_summary", summary)

    # Level
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    try:
        level = getattr(logging, level_name)
    except AttributeError:
        level = logging.INFO
    root.setLevel(level)

    # Formatter
    json_enabled = str(os.getenv("LOG_JSON", "false")).lower() in {"1", "true", "yes", "on"}
    summary["json"] = json_enabled

    class JsonFormatter(Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "ts": getattr(record, "created", None),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            return _json.dumps(payload, ensure_ascii=False)

    if json_enabled:
        formatter = JsonFormatter()
    else:
        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    console = StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler
    file_enabled = str(os.getenv("LOG_TO_FILE", "false")).lower() in {"1", "true", "yes", "on"}
    if file_enabled:
        file_path = os.getenv("LOG_FILE_PATH", os.path.join(os.getcwd(), "logs", "archon.log"))
        # Rotation settings can be tuned via env
        try:
            max_bytes = int(os.getenv("LOG_FILE_MAX_BYTES", "1000000"))
        except Exception:
            max_bytes = 1_000_000
        try:
            backup_count = int(os.getenv("LOG_BACKUP_COUNT", "3"))
        except Exception:
            backup_count = 3
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            fhandler = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backup_count)
            fhandler.setFormatter(formatter)
            root.addHandler(fhandler)
            summary["file"] = True
            summary["file_path"] = file_path
        except Exception as e:
            logging.getLogger(__name__).warning(f"Could not enable file logging: {e}")

    # Mark configured
    setattr(root, "_archon_logging_configured", True)
    setattr(root, "_archon_logging_summary", summary)
    return summary

def get_env_vars_file_path() -> str:
    """Return the path for env_vars.json compatible with host and containers.

    Resolution order:
    1) ARCHON_CONFIG env var if set
    2) Module-relative default: <repo>/src/archon/workbench/env_vars.json
    """
    cfg = os.environ.get("ARCHON_CONFIG")
    if cfg:
        return cfg
    default_path = Path(__file__).parent.parent / 'workbench' / 'env_vars.json'
    return str(default_path)

def load_env_vars() -> Dict[str, Any]:
    """Load environment variables from env_vars.json"""
    env_file_path = get_env_vars_file_path()
    
    try:
        with open(env_file_path, 'r') as f:
            env_vars = json.load(f)
        logger.debug(f"✅ Loaded env_vars from {env_file_path}")
        return env_vars
    except FileNotFoundError:
        logger.error(f"❌ env_vars.json not found at {env_file_path}")
        return {"current_profile": "openrouter_default", "profiles": {}}
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in env_vars.json: {e}")
        return {"current_profile": "openrouter_default", "profiles": {}}
    except Exception as e:
        logger.error(f"❌ Error loading env_vars.json: {e}")
        return {"current_profile": "openrouter_default", "profiles": {}}

def substitute_env_vars(value: str) -> str:
    """
    Substitute environment variables in a string.
    Replaces ${VAR_NAME} with the value from environment variables.
    """
    if not isinstance(value, str):
        return value
    
    # Pattern to match ${VAR_NAME}
    pattern = r'\$\{([^}]+)\}'
    
    def replace_var(match):
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is not None:
            logger.debug(f"🔄 Substituted ${{{var_name}}} -> {env_value}")
            return env_value
        else:
            logger.warning(f"⚠️ Environment variable ${{{var_name}}} not found, keeping original")
            return match.group(0)  # Return original ${VAR_NAME} if not found
    
    result = re.sub(pattern, replace_var, value)
    return result

def get_current_profile() -> str:
    """Get the current active profile name"""
    try:
        env_vars = load_env_vars()
        current_profile = env_vars.get("current_profile", "openrouter_default")
        logger.debug(f"📋 Current profile: {current_profile}")
        return current_profile
    except Exception as e:
        logger.error(f"❌ Error getting current profile: {e}")
        return "openrouter_default"

def get_profile_config(profile_name: Optional[str] = None) -> Dict[str, Any]:
    """Get configuration for a specific profile"""
    try:
        env_vars = load_env_vars()
        
        if profile_name is None:
            profile_name = get_current_profile()
        
        profiles = env_vars.get("profiles", {})
        if profile_name not in profiles:
            logger.warning(f"⚠️ Profile '{profile_name}' not found, available: {list(profiles.keys())}")
            # Return first available profile or empty dict
            if profiles:
                first_profile = next(iter(profiles.keys()))
                logger.info(f"🔄 Using first available profile: {first_profile}")
                return profiles[first_profile]
            return {}
        
        profile_config = profiles[profile_name]
        logger.debug(f"📋 Loaded config for profile '{profile_name}'")
        return profile_config
    
    except Exception as e:
        logger.error(f"❌ Error getting profile config for '{profile_name}': {e}")
        return {}

def get_env_var(var_name: str, profile_name: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable value from profile or environment.
    First checks the profile configuration, then environment variables.
    Performs variable substitution on profile values.
    """
    try:
        # First try to get from profile
        profile_config = get_profile_config(profile_name)
        
        if var_name in profile_config:
            raw_value = profile_config[var_name]
            if isinstance(raw_value, str):
                # Substitute environment variables
                substituted_value = substitute_env_vars(raw_value)
                logger.debug(f"🔍 Profile var {var_name}: {raw_value} -> {substituted_value}")
                return substituted_value
            else:
                return str(raw_value)
        
        # If not in profile, try environment variables
        env_value = os.environ.get(var_name)
        if env_value is not None:
            logger.debug(f"🌍 Environment var {var_name}: {env_value}")
            return env_value
        
        logger.debug(f"❓ Variable '{var_name}' not found in profile or environment")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error getting environment variable '{var_name}': {e}")
        return None

def save_env_var(var_name: str, var_value: str, profile_name: Optional[str] = None) -> bool:
    """
    Save environment variable to a specific profile in env_vars.json
    
    Args:
        var_name: Name of the environment variable
        var_value: Value to save
        profile_name: Profile to save to (uses current if None)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        env_file_path = get_env_vars_file_path()
        
        # Load existing env vars
        if os.path.exists(env_file_path):
            with open(env_file_path, 'r') as f:
                env_vars = json.load(f)
        else:
            env_vars = {"current_profile": "ollama_default", "profiles": {}}
        
        # Ensure profiles section exists
        if "profiles" not in env_vars:
            env_vars["profiles"] = {}
        
        # Use provided profile or current profile
        if profile_name is None:
            profile_name = get_current_profile()
        
        # Ensure profile exists
        if profile_name not in env_vars["profiles"]:
            env_vars["profiles"][profile_name] = {}
        
        # Save the variable
        env_vars["profiles"][profile_name][var_name] = var_value
        
        # Write back to file
        with open(env_file_path, 'w') as f:
            json.dump(env_vars, f, indent=2)
        
        logger.info(f"✅ Saved {var_name}='{var_value}' to profile '{profile_name}'")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save env var '{var_name}': {e}")
        return False

def build_llm_config_from_active_profile(llm_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a strict LLM configuration from the active profile only.

    This constructs the llm_config dictionary by reading exclusively from the
    currently active profile in `env_vars.json` (no environment fallbacks).

    Provider-specific validation rules:
    - openrouter: requires LLM_API_KEY and BASE_URL
    - openai: requires LLM_API_KEY (BASE_URL optional)
    - ollama: requires OLLAMA_BASE_URL

    Args:
        llm_overrides: Optional overrides to merge into the resulting config.
                       Only keys with non-None values will override.

    Returns:
        Dict[str, Any]: The constructed llm_config.

    Raises:
        ValueError: If required provider fields are missing.
    """
    profile_name = get_current_profile()
    cfg = get_profile_config(profile_name) or {}

    provider = str(cfg.get("LLM_PROVIDER") or "").lower()
    if not provider:
        raise ValueError("LLM_PROVIDER missing in active profile")

    # Models: use profile-only fallbacks (no env)
    primary_model = (
        cfg.get("PRIMARY_MODEL")
        or cfg.get("REASONER_MODEL")
        or cfg.get("ADVISOR_MODEL")
        or cfg.get("CODER_MODEL")
    )
    reasoner_model = cfg.get("REASONER_MODEL") or primary_model
    advisor_model = cfg.get("ADVISOR_MODEL") or primary_model
    coder_model = cfg.get("CODER_MODEL") or primary_model

    # Base URLs / keys strictly from profile
    base_url = cfg.get("BASE_URL")
    api_key = cfg.get("LLM_API_KEY")
    ollama_base = cfg.get("OLLAMA_BASE_URL")
    ollama_model = cfg.get("OLLAMA_MODEL")

    # Validate provider-specific requirements
    if provider == "openrouter":
        if not api_key:
            raise ValueError("Missing OpenRouter API key: set 'LLM_API_KEY' in profile")
        if not base_url:
            raise ValueError("Missing OpenRouter BASE_URL: set 'BASE_URL' in profile")
    elif provider == "openai":
        if not api_key:
            raise ValueError("Missing OpenAI API key: set 'LLM_API_KEY' in profile")
        # base_url optional for OpenAI
    elif provider == "ollama":
        if not ollama_base:
            raise ValueError("Missing Ollama base URL: set 'OLLAMA_BASE_URL' in profile")
        if not (reasoner_model or advisor_model or coder_model or ollama_model):
            raise ValueError("No model specified for Ollama in profile")
    else:
        raise ValueError(f"Unsupported LLM provider in profile: {provider}")

    # Operational settings strictly from profile (with safe defaults)
    timeout_s = float(cfg.get("TIMEOUT_S") or 30)
    max_conc = int(cfg.get("LLM_MAX_PARALLEL_BATCHES") or cfg.get("MAX_CONCURRENCY") or 2)
    enable_langues = bool(cfg.get("ENABLE_LANGUES", True))
    enable_localisation = bool(cfg.get("ENABLE_LOCALISATION", True))
    weights = cfg.get("WEIGHTS") or {}

    llm_config: Dict[str, Any] = {
        "LLM_PROVIDER": provider,
        "BASE_URL": base_url,
        "LLM_API_KEY": api_key,
        "REASONER_MODEL": reasoner_model,
        "ADVISOR_MODEL": advisor_model,
        "CODER_MODEL": coder_model,
        "TIMEOUT_S": timeout_s,
        "MAX_CONCURRENCY": max_conc,
        "ENABLE_LANGUES": enable_langues,
        "ENABLE_LOCALISATION": enable_localisation,
        "WEIGHTS": weights,
        # Ollama-specific (may be unused depending on provider)
        "OLLAMA_BASE_URL": ollama_base,
        "OLLAMA_MODEL": ollama_model,
        # Optional OpenRouter headers if present in profile
        "OPENROUTER_REFERRER": cfg.get("OPENROUTER_REFERRER"),
        "OPENROUTER_X_TITLE": cfg.get("OPENROUTER_X_TITLE"),
    }

    # Merge safe overrides
    if isinstance(llm_overrides, dict):
        llm_config.update({k: v for k, v in llm_overrides.items() if v is not None})

    logger.info(
        f"🔧 Built llm_config from profile='{profile_name}' provider='{provider}' "
        f"reasoner='{llm_config.get('REASONER_MODEL')}' advisor='{llm_config.get('ADVISOR_MODEL')}' coder='{llm_config.get('CODER_MODEL')}'"
    )

    return llm_config

def reload_archon_graph(show_reload_success: bool = True):
    """
    Reload the archon graph configuration.

    Note: Kept as a lightweight placeholder for compatibility with callers
    like `streamlit_pages/environment.py`, which may pass the optional
    parameter `show_reload_success`.

    Args:
        show_reload_success (bool): Optional flag from UI callers; currently
            ignored but accepted to maintain backward/forward compatibility.

    Returns:
        bool: Always True to indicate the reload hook executed.
    """
    logger.info("🔄 Reloading archon graph configuration")
    return True

def set_current_profile(profile_name: str) -> bool:
    """Set the current active profile"""
    try:
        env_vars = load_env_vars()
        env_vars["current_profile"] = profile_name
        return save_env_vars(env_vars)
    except Exception as e:
        logger.error(f"❌ Failed to set current profile: {e}")
        return False

def get_all_profiles() -> list:
    """Get all available profiles"""
    try:
        env_vars = load_env_vars()
        return list(env_vars.get("profiles", {}).keys())
    except Exception as e:
        logger.error(f"❌ Failed to get all profiles: {e}")
        return []

def create_profile(profile_name: str) -> bool:
    """Create a new profile"""
    try:
        env_vars = load_env_vars()
        if "profiles" not in env_vars:
            env_vars["profiles"] = {}
        
        if profile_name not in env_vars["profiles"]:
            env_vars["profiles"][profile_name] = {}
            return save_env_vars(env_vars)
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create profile: {e}")
        return False

def delete_profile(profile_name: str) -> bool:
    """Delete a profile"""
    try:
        env_vars = load_env_vars()
        if "profiles" in env_vars and profile_name in env_vars["profiles"]:
            del env_vars["profiles"][profile_name]
            return save_env_vars(env_vars)
        return True
    except Exception as e:
        logger.error(f"❌ Failed to delete profile: {e}")
        return False

def get_profile_env_vars(profile_name: str) -> dict:
    """Get all environment variables for a specific profile"""
    try:
        env_vars = load_env_vars()
        return env_vars.get("profiles", {}).get(profile_name, {})
    except Exception as e:
        logger.error(f"❌ Failed to get profile env vars: {e}")
        return {}

def save_env_vars(env_vars: Dict[str, Any]) -> bool:
    """Save environment variables to env_vars.json"""
    env_file_path = get_env_vars_file_path()
    
    try:
        with open(env_file_path, 'w') as f:
            json.dump(env_vars, f, indent=2)
        logger.info(f"✅ Saved env_vars to {env_file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving env_vars.json: {e}")
        return False

def switch_profile(new_profile: str) -> bool:
    """Switch to a different profile"""
    try:
        env_vars = load_env_vars()
        
        if new_profile not in env_vars.get("profiles", {}):
            logger.error(f"❌ Profile '{new_profile}' not found")
            return False
        
        env_vars["current_profile"] = new_profile
        
        if save_env_vars(env_vars):
            logger.info(f"✅ Switched to profile '{new_profile}'")
            return True
        else:
            logger.error(f"❌ Failed to save profile switch to '{new_profile}'")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error switching to profile '{new_profile}': {e}")
        return False

def validate_profile_config(profile_name: str) -> bool:
    """Validate that a profile has required configuration"""
    try:
        config = get_profile_config(profile_name)
        
        # Check for required fields
        required_fields = ["LLM_PROVIDER"]
        for field in required_fields:
            if field not in config:
                logger.warning(f"⚠️ Profile '{profile_name}' missing required field: {field}")
                return False
        
        provider = config.get("LLM_PROVIDER", "").lower()
        
        # Provider-specific validation
        if provider == "ollama":
            if "OLLAMA_BASE_URL" not in config:
                logger.warning(f"⚠️ Ollama profile '{profile_name}' missing OLLAMA_BASE_URL")
                return False
        elif provider in ["openrouter", "openai"]:
            if "LLM_API_KEY" not in config:
                logger.warning(f"⚠️ {provider} profile '{profile_name}' missing LLM_API_KEY")
                return False
        
        logger.debug(f"✅ Profile '{profile_name}' validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating profile '{profile_name}': {e}")
        return False

def create_new_tab_button(label, tab_name, key=None, use_container_width=False):
    """Create a button that opens a specified tab in a new browser window"""
    # Create a unique key if none provided
    if key is None:
        key = f"new_tab_{tab_name.lower().replace(' ', '_')}"
    
    # Get the base URL
    base_url = st.query_params.get("base_url", "")
    if not base_url:
        # If base_url is not in query params, use the default localhost URL
        base_url = "http://localhost:8501"
    
    # Create the URL for the new tab
    new_tab_url = f"{base_url}/?tab={tab_name}"
    
    # Create a button that will open the URL in a new tab when clicked
    if st.button(label, key=key, use_container_width=use_container_width):
        webbrowser.open_new_tab(new_tab_url)

def get_clients() -> Tuple[Any, Optional[Any], Optional[Any]]:
    """
    Initialiser tous les clients nécessaires pour Archon:
    - Client d'embedding (OpenAI ou autre)
    - Client Supabase pour la base de données vectorielle
    - Client Neo4j pour la base de données graphe
    
    Returns:
        Tuple contenant (embedding_client, supabase_client, neo4j_client)
    """
    try:
        # Détermination du provider d'embedding
        embedding_client = None
        provider = get_env_var('EMBEDDING_PROVIDER') or 'OpenAI'
        base_url = get_env_var('EMBEDDING_BASE_URL') or 'https://api.openai.com/v1'

        # Import paresseux d'OpenAI si requis par le provider
        _need_openai = str(provider).strip().lower() in {"openai", "ollama"}
        _AsyncOpenAI = AsyncOpenAI
        if _need_openai and _AsyncOpenAI is None:
            try:
                from openai import AsyncOpenAI as _AO
                _AsyncOpenAI = _AO
            except ImportError:
                logger.debug("openai package not installed; skipping AsyncOpenAI client init")

        # For Ollama, use a dummy API key if not provided and ensure the base URL is correct
        if provider == "Ollama":
            api_key = get_env_var('EMBEDDING_API_KEY') or 'ollama'
            if _AsyncOpenAI is None:
                logger.debug("AsyncOpenAI unavailable for Ollama-compatible client; embedding_client=None")
            else:
                try:
                    embedding_client = _AsyncOpenAI(base_url=base_url, api_key=api_key)
                except Exception as e:
                    logger.error(f"Failed to initialize AsyncOpenAI: {e}")
                    embedding_client = None
        else:
            api_key = get_env_var('EMBEDDING_API_KEY') or 'no-api-key-provided'
            if _need_openai and _AsyncOpenAI is not None:
                try:
                    embedding_client = _AsyncOpenAI(base_url=base_url, api_key=api_key)
                except Exception as e:
                    logger.error(f"Failed to initialize AsyncOpenAI: {e}")
                    embedding_client = None
            else:
                # Provider not requiring OpenAI client or package not present
                logger.debug("Skipping AsyncOpenAI init: provider doesn't require it or package missing")

        # Supabase client setup
        supabase = None
        supabase_url = get_env_var("SUPABASE_URL")
        supabase_key = get_env_var("SUPABASE_SERVICE_KEY")

        # Cross-profile fallback: if missing in current profile/env, scan all profiles
        if (not supabase_url or not supabase_key):
            try:
                _env = load_env_vars()
                for _pname, _pcfg in (_env.get("profiles") or {}).items():
                    if not supabase_url:
                        supabase_url = _pcfg.get("SUPABASE_URL") or supabase_url
                    if not supabase_key:
                        supabase_key = _pcfg.get("SUPABASE_SERVICE_KEY") or supabase_key
                    if supabase_url and supabase_key:
                        logger.debug(f"Supabase config found in profile '{_pname}' via fallback scan")
                        break
            except Exception as _e:
                logger.debug(f"Supabase cross-profile scan skipped: {_e}")
        if supabase_url and supabase_key:
            try:
                if _supabase_create_client is not None:
                    # Preferred modern API
                    supabase = _supabase_create_client(supabase_url, supabase_key)
                    logger.debug("Supabase client (v2 create_client) initialized successfully")
                elif _SupabaseClient is not None:
                    # Fallback legacy constructor (if available)
                    supabase = _SupabaseClient(supabase_url, supabase_key)
                    logger.debug("Supabase client (legacy Client) initialized successfully")
                else:
                    logger.debug("Supabase package not providing create_client/Client; skipping init")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase: {e}")
        else:
            logger.debug("Supabase configuration incomplete (missing URL or SERVICE_KEY)")
        
        # Neo4j client setup (opt-in via NEO4J_ENABLED)
        neo4j_client = None
        neo4j_enabled = get_bool_env("NEO4J_ENABLED", False)
        if not neo4j_enabled:
            logger.debug("Neo4j disabled via NEO4J_ENABLED=false (default)")
        else:
            neo4j_uri = get_env_var("NEO4J_URI")
            neo4j_user = get_env_var("NEO4J_USER")
            neo4j_password = get_env_var("NEO4J_PASSWORD")

            if neo4j_uri and neo4j_user and neo4j_password and Neo4jClient is not None:
                try:
                    # Neo4jClient(uri, user, password)
                    neo4j_client = Neo4jClient(
                        neo4j_uri,
                        neo4j_user,
                        neo4j_password,
                    )
                    logger.debug(f"Neo4j client initialized successfully at {neo4j_uri}")
                except Exception as e:
                    logger.error(f"Failed to initialize Neo4j: {e}")
            else:
                logger.debug("Neo4j configuration incomplete or Neo4jClient not available")

        return embedding_client, supabase, neo4j_client
    
    except Exception as e:
        logger.error(f"Error initializing clients: {e}")
        return None, None, None

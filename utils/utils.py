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
import streamlit as st

# Imports nécessaires pour la fonction get_clients
try:
    from openai import AsyncOpenAI
    from supabase import Client
    from archon.utils.neo4j_client import Neo4jClient
except ImportError as e:
    logging.warning(f"Import error for client libraries: {e}")

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

def get_env_vars_file_path() -> str:
    """Get the path to the env_vars.json file"""
    # Try multiple possible locations
    possible_paths = [
        "/app/workbench/env_vars.json",
        "/app/src/archon/workbench/env_vars.json",
        "src/archon/workbench/env_vars.json",
        "workbench/env_vars.json"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Default path if none found
    return "/app/src/archon/workbench/env_vars.json"

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
        # LLM client setup
        embedding_client = None
        provider = get_env_var('EMBEDDING_PROVIDER') or 'OpenAI'
        base_url = get_env_var('EMBEDDING_BASE_URL') or 'https://api.openai.com/v1'
        
        # For Ollama, use a dummy API key if not provided and ensure the base URL is correct
        if provider == "Ollama":
            api_key = get_env_var('EMBEDDING_API_KEY') or 'ollama'
            embedding_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        else:
            api_key = get_env_var('EMBEDDING_API_KEY') or 'no-api-key-provided'
            embedding_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        # Supabase client setup
        supabase = None
        supabase_url = get_env_var("SUPABASE_URL")
        supabase_key = get_env_var("SUPABASE_SERVICE_KEY")
        if supabase_url and supabase_key:
            try:
                supabase = Client(supabase_url, supabase_key)
            except Exception as e:
                logger.error(f"Failed to initialize Supabase: {e}")
        
        # Neo4j client setup
        neo4j_client = None
        neo4j_uri = get_env_var("NEO4J_URI")
        neo4j_user = get_env_var("NEO4J_USER")
        neo4j_password = get_env_var("NEO4J_PASSWORD")
        neo4j_database = get_env_var("NEO4J_DATABASE") or "neo4j"
        
        if neo4j_uri and neo4j_user and neo4j_password:
            try:
                neo4j_client = Neo4jClient(
                    uri=neo4j_uri,
                    username=neo4j_user,
                    password=neo4j_password,
                    database=neo4j_database
                )
                logger.debug(f"Neo4j client initialized successfully at {neo4j_uri}")
            except Exception as e:
                logger.error(f"Failed to initialize Neo4j: {e}")

        return embedding_client, supabase, neo4j_client
    
    except Exception as e:
        logger.error(f"Error initializing clients: {e}")
        return None, None, None

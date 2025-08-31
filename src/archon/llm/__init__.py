"""
Unified LLM Provider for Archon
Supports multiple LLM providers with dynamic profile switching

This module provides a clean interface to interact with various LLM providers
using a profile-based configuration system.
"""
from typing import Optional
from .llm_providers import (
    LLMProvider,
    LLMConfig,
    ProfileConfig,
    create_provider,
    load_config,
    load_profile,
    get_config_path
)

# Export public API
__all__ = [
    'LLMProvider',
    'LLMConfig',
    'ProfileConfig',
    'llm_provider',
    'get_llm_provider',
    'load_config',
    'load_profile',
    'get_config_path'
]

# Global provider instance
llm_provider: Optional[LLMProvider] = None

def get_llm_provider() -> LLMProvider:
    """
    Get or create the global LLM provider instance.
    
    Returns:
        LLMProvider: The singleton instance of the LLM provider
    """
    global llm_provider
    if llm_provider is None:
        llm_provider = create_provider()
    return llm_provider

# Initialize the global provider instance on import (clean)
try:
    llm_provider = create_provider()
    if llm_provider and llm_provider.config:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Initialized LLM Provider with {llm_provider.config.provider}")
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to initialize LLM provider: {e}")
    llm_provider = None

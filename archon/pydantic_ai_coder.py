"""Proxy module for pydantic_ai_coder ensuring test monkeypatches propagate.

This module re-exports selected symbols from
`archon.archon.agents.pydantic_ai_coder` but also synchronizes the `Agent`
symbol so tests that patch `coder.Agent` (this module) influence the inner
factory implementation.
"""

from archon.archon.agents import pydantic_ai_coder as _inner

# Re-export commonly used callables and constants
dynamic_coder_prompt = _inner.dynamic_coder_prompt
retrieve_relevant_documentation = _inner.retrieve_relevant_documentation
list_documentation_pages = _inner.list_documentation_pages
get_page_content = _inner.get_page_content

# Default Agent binding points to the inner module's Agent; unit tests may patch this
Agent = _inner.Agent

# Expose env helpers so unit tests can monkeypatch them on this proxy
get_bool_env = _inner.get_bool_env
validate_rag_env = _inner.validate_rag_env


def create_pydantic_ai_coder(custom_model=None):
    """Create the coder agent while honoring a potentially patched Agent.

    If `Agent` has been monkeypatched on this proxy module (e.g., by unit tests),
    copy it into the inner implementation before delegating, so the factory uses
    the patched class instead of the real pydantic_ai.Agent.
    """
    # Propagate patched symbols into inner module if applicable
    try:
        _inner.Agent = globals().get("Agent", _inner.Agent)
        _inner.get_bool_env = globals().get("get_bool_env", _inner.get_bool_env)
        _inner.validate_rag_env = globals().get("validate_rag_env", _inner.validate_rag_env)
    except Exception:
        # Fail closed: leave the inner Agent as-is
        pass
    return _inner.create_pydantic_ai_coder(custom_model=custom_model)

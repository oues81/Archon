"""Compatibility shim for tests expecting archon.config.config.

Exposes load_config and related helpers by forwarding to the canonical
implementation used by the LLM provider profiles.
"""
from __future__ import annotations

from typing import Any, Dict

# Forward to the implementation used by archon.llm
from k.core.llm.llm_providers import load_config as load_config  # re-export
from k.core.llm.llm_providers import get_config_path as get_config_path  # re-export

__all__ = ["load_config", "get_config_path"]

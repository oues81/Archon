"""Lightweight Ollama provider classes used in tests.

These classes provide a minimal surface so imports like
`from archon.archon.models.ollama_model import OllamaModel, OllamaClient`
work, and basic attributes are available.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class OllamaClient:
    """Simple holder for Ollama connection parameters."""
    base_url: Optional[str] = None
    timeout: int = 30


class OllamaModel:
    """Minimal model wrapper compatible with tests expectations."""
    def __init__(self, model_name: str, base_url: Optional[str] = None) -> None:
        self.model_name = model_name
        self.client = OllamaClient(base_url=base_url)


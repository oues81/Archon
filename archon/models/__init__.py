"""
Package pour les modèles de données d'Archon.
Contient les définitions des modèles utilisés dans l'application.
"""

# Les classes sont définies dans archon/archon/ollama_model.py (un niveau au-dessus)
from ..providers.ollama_model import OllamaModel, OllamaClient

# Compatibilité d'import: permettre `from archon.archon.models import ollama_model`
# et `from archon.archon.models.ollama_model import OllamaModel`
import sys as _sys
from types import ModuleType as _ModuleType
from ..providers import ollama_model as _ollama_module
_sys.modules[__name__ + ".ollama_model"] = _ollama_module

__all__ = ['OllamaModel', 'OllamaClient']

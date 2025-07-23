"""
Package pour les modèles de données d'Archon.
Contient les définitions des modèles utilisés dans l'application.
"""

from .ollama_model import OllamaModel, OllamaClient

__all__ = ['OllamaModel', 'OllamaClient']

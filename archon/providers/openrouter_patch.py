#!/usr/bin/env python3
"""
Module de patch pour OpenRouter.
Ce module applique des correctifs à la bibliothèque OpenAI pour garantir
une authentification correcte avec OpenRouter.
"""

import os
import json
import logging
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional

# Configuration du logger
logger = logging.getLogger(__name__)

def is_module_available(module_name: str) -> bool:
    """Vérifie si un module Python est disponible."""
    return importlib.util.find_spec(module_name) is not None

def apply_openrouter_patch() -> None:
    """
    Applique un patch à la bibliothèque OpenAI pour ajouter les en-têtes
    requis par OpenRouter à chaque requête.
    """
    logger.info("Application du patch OpenRouter pour la bibliothèque OpenAI...")
    
    if not is_module_available("openai"):
        logger.warning("La bibliothèque OpenAI n'est pas disponible, patch ignoré.")
        return
    
    try:
        # Chargement du module OpenAI
        import openai
        from openai._client import OpenAI, AsyncOpenAI
        from openai._base_client import BaseClient
        
        # Chemin vers le fichier des en-têtes OpenRouter
        headers_file = Path(__file__).parent / "openrouter_headers.json"
        
        # Vérifier si le fichier existe
        if headers_file.exists():
            try:
                # Charger les en-têtes depuis le fichier
                with open(headers_file, "r", encoding="utf-8") as f:
                    headers = json.load(f)
                logger.info(f"En-têtes chargés depuis {headers_file}: {list(headers.keys())}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des en-têtes: {e}")
                headers = {}
        else:
            logger.warning(f"Fichier d'en-têtes introuvable: {headers_file}")
            # En-têtes par défaut
            api_key = os.environ.get("OPENAI_API_KEY", "")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Archon"
            }
        
        # Sauvegarder la méthode d'initialisation originale
        original_init = BaseClient.__init__
        
        # Définir une nouvelle méthode d'initialisation avec les en-têtes requis
        def patched_init(self, *args, **kwargs):
            # Appeler l'initialisation originale
            original_init(self, *args, **kwargs)
            
            # Ajouter les en-têtes requis par OpenRouter
            for header, value in headers.items():
                if header not in self.default_headers:
                    self.default_headers[header] = value
            
            logger.info(f"Client OpenAI patché avec les en-têtes: {list(self.default_headers.keys())}")
        
        # Appliquer le patch
        BaseClient.__init__ = patched_init
        
        logger.info("Patch OpenRouter appliqué avec succès à la bibliothèque OpenAI")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'application du patch OpenRouter: {e}")
        import traceback
        logger.error(traceback.format_exc())

# Appliquer le patch au chargement du module
apply_openrouter_patch()

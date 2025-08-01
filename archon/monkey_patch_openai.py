#!/usr/bin/env python3
"""
Script pour appliquer un monkey patch direct à la bibliothèque OpenAI.
Ce script modifie directement les méthodes de requête de la bibliothèque OpenAI
pour garantir que les en-têtes requis par OpenRouter sont inclus dans chaque requête.
"""

import logging
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Configuration du logger
logger = logging.getLogger(__name__)

def apply_monkey_patch():
    """
    Applique un monkey patch à la bibliothèque OpenAI pour forcer l'inclusion
    des en-têtes HTTP requis par OpenRouter dans toutes les requêtes.
    """
    logger.info("Application du monkey patch à la bibliothèque OpenAI...")
    
    try:
        # Importer la bibliothèque OpenAI
        import openai
        from openai._base_client import BaseClient
        
        # Chemin du fichier env_vars.json
        env_vars_file = Path(__file__).parent.parent / "workbench" / "env_vars.json"
        api_key = None
        
        # Récupérer la clé API depuis env_vars.json
        if env_vars_file.exists():
            try:
                with open(env_vars_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    current_profile = data.get("current_profile", "default")
                    profiles = data.get("profiles", {})
                    if current_profile in profiles:
                        api_key = profiles[current_profile].get("OPENROUTER_API_KEY")
                        logger.info(f"Clé API récupérée depuis le profil: {current_profile}")
            except Exception as e:
                logger.error(f"Erreur lors de la lecture de env_vars.json: {e}")
        
        # Utiliser la clé d'environnement si non trouvée dans le fichier
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            logger.info("Utilisation de la clé API depuis les variables d'environnement")
        
        if not api_key:
            logger.error("Aucune clé API OpenRouter trouvée!")
            return
            
        # Masquer la clé pour les logs
        masked_key = api_key[:6] + "*****" + api_key[-4:] if len(api_key) > 10 else "***"
        logger.info(f"Utilisation de la clé API: {masked_key}")
        
        # Configurer les en-têtes OpenRouter requis
        openrouter_headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Archon"
        }
        
        # Sauvegarder la méthode originale
        original_request = BaseClient.request
        
        # Définir la nouvelle méthode patched
        async def patched_request(self, *args, **kwargs):
            """Version patchée de la méthode request qui ajoute les en-têtes OpenRouter"""
            # Vérifier si la requête va vers OpenRouter
            if "openrouter.ai" in self.base_url:
                # S'assurer que les en-têtes par défaut sont configurés
                if "headers" not in kwargs:
                    kwargs["headers"] = {}
                
                # Ajouter les en-têtes requis par OpenRouter
                for key, value in openrouter_headers.items():
                    if key not in kwargs["headers"]:
                        kwargs["headers"][key] = value
                
                logger.info(f"En-têtes ajoutés à la requête OpenRouter: {list(kwargs['headers'].keys())}")
            
            # Appeler la méthode originale
            return await original_request(self, *args, **kwargs)
        
        # Appliquer le patch
        BaseClient.request = patched_request
        
        # Configurer également les variables d'environnement pour les autres clients
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        
        logger.info("✅ Monkey patch appliqué avec succès à la bibliothèque OpenAI")
        return True
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'application du monkey patch: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

# Appliquer le patch immédiatement lors de l'importation
patch_result = apply_monkey_patch()

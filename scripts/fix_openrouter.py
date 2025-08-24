#!/usr/bin/env python3
"""
Script pour corriger l'authentification OpenRouter
Ce script ajoute des patches aux bibliothèques pour s'assurer que les en-têtes HTTP requis
sont correctement inclus dans les requêtes à OpenRouter.
"""

import os
import logging
import json
from pathlib import Path

# Configurer le logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chemin vers le fichier de configuration
WORKBENCH_DIR = Path(__file__).parent / "workbench"
ENV_VARS_FILE = WORKBENCH_DIR / "env_vars.json"

def load_profile_config():
    """Charge la configuration du profil actif"""
    try:
        if not ENV_VARS_FILE.exists():
            logger.warning(f"Fichier de configuration introuvable: {ENV_VARS_FILE}")
            return {}
            
        with open(ENV_VARS_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        current_profile = config.get('current_profile', 'default')
        profiles = config.get('profiles', {})
        
        if current_profile not in profiles:
            logger.warning(f"Le profil actif '{current_profile}' n'existe pas dans la configuration.")
            return {}
            
        return profiles[current_profile]
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration du profil: {e}")
        return {}

def get_env_from_profile(var_name, default=None):
    """Récupère une variable d'environnement depuis le profil actif"""
    profile_config = load_profile_config()
    if var_name in profile_config and profile_config[var_name] not in (None, ''):
        return str(profile_config[var_name])
    return os.getenv(var_name, default)

def patch_openai_client():
    """
    Patch la bibliothèque OpenAI pour ajouter les en-têtes requis par OpenRouter.
    """
    try:
        # Importer la bibliothèque OpenAI
        import openai
        from openai._client import OpenAI, AsyncOpenAI
        from openai._base_client import BaseClient

        # Récupérer la clé API depuis le profil
        api_key = get_env_from_profile("OPENROUTER_API_KEY")
        
        if not api_key:
            logger.error("Aucune clé API OpenRouter trouvée")
            return False

        # Masquer partiellement la clé pour la journalisation
        masked_key = api_key[:6] + "*****" + api_key[-4:] if len(api_key) > 10 else "***"
        logger.info(f"Patching OpenAI client avec la clé API: {masked_key}")
        
        # Configurer les variables d'environnement requises
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

        # Sauvegarder la méthode originale
        original_init = BaseClient.__init__

        # Définir une nouvelle méthode d'initialisation avec les en-têtes requis
        def patched_init(self, *args, **kwargs):
            # Appeler l'initialisation originale
            original_init(self, *args, **kwargs)
            
            # Ajouter les en-têtes requis par OpenRouter
            self.default_headers["HTTP-Referer"] = "http://localhost"
            self.default_headers["X-Title"] = "Archon"
            
            logger.info(f"En-têtes HTTP ajoutés au client OpenAI: {list(self.default_headers.keys())}")

        # Appliquer le patch
        BaseClient.__init__ = patched_init
        
        logger.info("Client OpenAI patché avec succès pour inclure les en-têtes OpenRouter")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du patching du client OpenAI: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Fonction principale"""
    logger.info("Démarrage du script de correction pour OpenRouter...")
    
    # Appliquer le patch
    success = patch_openai_client()
    
    if success:
        logger.info("Correction appliquée avec succès.")
        # Vérifier la configuration actuelle
        profile = load_profile_config()
        logger.info(f"Profil actif: {profile.get('LLM_PROVIDER', 'Non défini')}")
        logger.info(f"Modèle principal: {profile.get('PRIMARY_MODEL', 'Non défini')}")
        logger.info(f"Modèle reasoner: {profile.get('REASONER_MODEL', 'Non défini')}")
    else:
        logger.error("Échec de l'application de la correction.")

if __name__ == "__main__":
    main()

"""
Script pour mettre à jour les variables d'environnement à partir du profil actif.
Ce script peut être appelé pour mettre à jour dynamiquement les variables d'environnement
sans avoir besoin de redémarrer le service.
"""
import os
import sys
import json
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemin vers le répertoire de travail
WORKBENCH_DIR = Path(__file__).parent.parent / "workbench"
ENV_VARS_FILE = WORKBENCH_DIR / "env_vars.json"

def load_current_profile():
    """Charge le profil actif depuis le fichier de configuration"""
    try:
        if not ENV_VARS_FILE.exists():
            logger.error(f"Fichier de configuration introuvable: {ENV_VARS_FILE}")
            return None
            
        with open(ENV_VARS_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        current_profile = config.get('current_profile', 'default')
        profiles = config.get('profiles', {})
        
        if current_profile not in profiles:
            logger.warning(f"Le profil actif '{current_profile}' n'existe pas dans la configuration.")
            return None
            
        return {
            'name': current_profile,
            'config': profiles[current_profile]
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du profil: {e}")
        return None

def update_environment_from_profile():
    """Met à jour les variables d'environnement à partir du profil actif"""
    profile = load_current_profile()
    if not profile:
        logger.error("Impossible de charger le profil actif")
        return False
        
    logger.info(f"Mise à jour des variables d'environnement depuis le profil: {profile['name']}")
    
    try:
        # Mettre à jour les variables d'environnement
        for key, value in profile['config'].items():
            # Ne pas écraser les variables système importantes
            if key.startswith('_') or key == 'name' or key == 'description':
                continue
                
            if value is not None and value != '':
                os.environ[key] = str(value)
                logger.debug(f"Défini {key} = {value[:20]}..." if len(str(value)) > 20 else f"Défini {key} = {value}")
        
        logger.info("Variables d'environnement mises à jour avec succès")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour des variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    logger.info("=== Mise à jour des variables d'environnement depuis le profil actif ===")
    
    # Afficher les variables d'environnement actuelles
    current_vars = {
        'LLM_PROVIDER': os.getenv('LLM_PROVIDER'),
        'LLM_MODEL': os.getenv('LLM_MODEL'),
        'REASONER_MODEL': os.getenv('REASONER_MODEL'),
        'EMBEDDING_PROVIDER': os.getenv('EMBEDDING_PROVIDER'),
        'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL'),
        'OLLAMA_API_BASE': os.getenv('OLLAMA_API_BASE'),
        'OPENAI_API_KEY': '***' if os.getenv('OPENAI_API_KEY') else None,
        'OPENROUTER_API_KEY': '***' if os.getenv('OPENROUTER_API_KEY') else None
    }
    logger.info("Variables d'environnement actuelles:")
    for k, v in current_vars.items():
        logger.info(f"  {k} = {v}")
    
    # Mettre à jour les variables d'environnement
    success = update_environment_from_profile()
    
    # Afficher les variables mises à jour
    if success:
        updated_vars = {
            'LLM_PROVIDER': os.getenv('LLM_PROVIDER'),
            'LLM_MODEL': os.getenv('LLM_MODEL'),
            'REASONER_MODEL': os.getenv('REASONER_MODEL'),
            'EMBEDDING_PROVIDER': os.getenv('EMBEDDING_PROVIDER'),
            'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL'),
            'OLLAMA_API_BASE': os.getenv('OLLAMA_API_BASE'),
            'OPENAI_API_KEY': '***' if os.getenv('OPENAI_API_KEY') else None,
            'OPENROUTER_API_KEY': '***' if os.getenv('OPENROUTER_API_KEY') else None
        }
        
        logger.info("\nVariables d'environnement mises à jour:")
        for k, v in updated_vars.items():
            logger.info(f"  {k} = {v}")
        
        logger.info("\nPour appliquer ces changements, vous devez recharger le module LLMProvider.")
    else:
        logger.error("Échec de la mise à jour des variables d'environnement")

"""
Test d'intégration simplifié pour OpenRouter.
Ce script teste directement la connexion à l'API OpenRouter sans dépendre de l'agent conseiller.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_openrouter_simple.log')
    ]
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement depuis le fichier .env à la racine du projet
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

# Afficher les variables d'environnement chargées pour le débogage
logger.info(f"Chemin du fichier .env: {env_path}")
logger.info(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
logger.info(f"LLM_MODEL: {os.getenv('LLM_MODEL')}")

# Ajouter le répertoire parent au path pour permettre les imports
sys.path.append(str(Path(__file__).parent.parent))

from archon.archon.llm_provider import LLMProvider, LLMConfig

async def test_openrouter_connection():
    """Teste la connexion à l'API OpenRouter avec une requête simple."""
    logger.info("=== Début du test de connexion OpenRouter ===")
    
    # Vérifier que la clé API est configurée
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("ERREUR: La variable d'environnement OPENROUTER_API_KEY n'est pas définie.")
        logger.error("Veuillez créer un fichier .env à la racine du projet avec votre clé API OpenRouter.")
        return False
    
    # Configuration pour OpenRouter
    config = LLMConfig(
        provider="openrouter",
        model=os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0.7,
        max_tokens=1000
    )
    
    # Créer le fournisseur LLM
    llm_provider = LLMProvider(config)
    
    # Message de test
    messages = [
        {"role": "system", "content": "Tu es un assistant utile qui répond brièvement en français."},
        {"role": "user", "content": "Dis-moi bonjour en français et présente-toi en une phrase."}
    ]
    
    try:
        logger.info("Envoi d'une requête de test à OpenRouter...")
        logger.info(f"Modèle: {config.model}")
        
        # Envoyer la requête
        response = await llm_provider.generate(messages)
        
        # Afficher la réponse
        logger.info("\n=== Réponse reçue ===")
        logger.info(response)
        logger.info("\n✅ Test réussi! La connexion à OpenRouter fonctionne correctement.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'appel à l'API OpenRouter: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    asyncio.run(test_openrouter_connection())

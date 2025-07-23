"""
Test de l'agent conseiller avec OpenRouter.
"""
import os
import sys
import asyncio
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_advisor_agent.log')
    ]
)
logger = logging.getLogger(__name__)

# Ajout du répertoire parent au path pour permettre les imports
sys.path.append(str(Path(__file__).parent.parent))

# Charger les variables d'environnement
from dotenv import load_dotenv
load_dotenv()

async def test_advisor_agent():
    """Teste l'agent conseiller avec OpenRouter."""
    from archon.advisor_agent import advisor_agent
    
    # Message de test
    message = "Peux-tu me dire quelle est la capitale de la France ?"
    
    logger.info(f"Envoi du message à l'agent: {message}")
    
    try:
        # Appeler l'agent conseiller
        response = await advisor_agent.arun(message)
        
        # Afficher la réponse
        logger.info("Réponse de l'agent conseiller:")
        logger.info(response)
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors de l'appel à l'agent conseiller: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    logger.info("Démarrage du test de l'agent conseiller avec OpenRouter")
    
    # Vérifier que la clé API est configurée
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("La variable d'environnement OPENROUTER_API_KEY n'est pas définie.")
        logger.error("Veuvez créer un fichier .env à la racine du projet avec votre clé API OpenRouter.")
        sys.exit(1)
    
    # Exécuter le test
    try:
        response = asyncio.run(test_advisor_agent())
        logger.info("Test terminé avec succès!")
    except Exception as e:
        logger.error(f"Le test a échoué: {str(e)}")
        sys.exit(1)

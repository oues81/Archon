"""
Script de test pour vérifier la connexion à OpenRouter.
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
        logging.FileHandler('test_openrouter_connection.log')
    ]
)
logger = logging.getLogger(__name__)

# Ajout du répertoire parent au path pour permettre les imports
sys.path.append(str(Path(__file__).parent.parent))

# Charger les variables d'environnement
from dotenv import load_dotenv
import os

# Charger à partir du fichier .env à la racine du projet
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(env_path)

# Afficher les variables d'environnement chargées pour le débogage
logger.info(f"OPENROUTER_API_KEY: {'***' if os.getenv('OPENROUTER_API_KEY') else 'Non définie'}")
logger.info(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
logger.info(f"LLM_MODEL: {os.getenv('LLM_MODEL')}")

# Importations locales
try:
    # Essayer d'abord les imports relatifs
    try:
        from ..llm.factory import ModelFactory
        from ..config.model_config import ModelConfig
    except ImportError:
        # Fallback aux imports absolus
        from archon.archon.llm.factory import ModelFactory
        from archon.archon.config.model_config import ModelConfig
        
except ImportError as e:
    logger.error(f"Erreur d'importation: {e}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1)

async def test_openrouter_connection():
    """Teste la connexion à OpenRouter avec la configuration actuelle."""
    try:
        # Configuration du modèle
        model_config = ModelConfig.from_env('openrouter')
        logger.info(f"Configuration du modèle chargée: {model_config}")
        
        # Création du modèle via la fabrique
        model = ModelFactory.create_model(model_config)
        logger.info("Modèle OpenRouter créé avec succès")
        
        # Test de requête simple
        from pydantic_ai.messages import ModelRequest, UserPromptPart
        from pydantic_ai.settings import ModelSettings
        from pydantic_ai.models import ModelRequestParameters
        
        # Créer un message utilisateur avec ModelRequest et UserPromptPart
        user_message = ModelRequest(parts=[UserPromptPart(content="Bonjour, peux-tu me dire quelle est la capitale de la France ?")])
        
        # Créer les paramètres de la requête
        model_settings = ModelSettings()
        request_parameters = ModelRequestParameters(
            function_tools=[],
            allow_text_result=True,
            result_tools=[]
        )
        
        logger.info("Envoi d'une requête de test à OpenRouter...")
        # Utiliser la méthode request pour envoyer la requête
        response, usage = await model.request(
            messages=[user_message],
            model_settings=model_settings,
            model_request_parameters=request_parameters
        )
        
        logger.info("Réponse reçue d'OpenRouter:")
        logger.info(response)
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du test de connexion à OpenRouter: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Démarrage du test de connexion à OpenRouter")
    
    # Vérifier que la clé API est définie
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("La variable d'environnement OPENROUTER_API_KEY n'est pas définie.")
        logger.error("Veuvez créer un fichier .env à la racine du projet avec votre clé API OpenRouter.")
        sys.exit(1)
    
    # Exécuter le test
    success = asyncio.run(test_openrouter_connection())
    
    if success:
        logger.info("Test de connexion à OpenRouter réussi !")
        sys.exit(0)
    else:
        logger.error("Échec du test de connexion à OpenRouter")
        sys.exit(1)

"""
Test d'intégration pour vérifier le bon fonctionnement avec OpenRouter.
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
        logging.FileHandler('test_openrouter_integration.log')
    ]
)
logger = logging.getLogger(__name__)

# Ajout du répertoire parent au path pour permettre les imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from archon.archon.advisor_agent import advisor_agent
except ImportError:
    from archon.advisor_agent import advisor_agent

from archon.config.model_config import ModelConfig
from archon.llm.factory import ModelFactory

async def test_openrouter_integration():
    """Teste l'intégration avec OpenRouter."""
    print("=== Test d'intégration OpenRouter ===")
    
    # Vérifier que la clé API est configurée
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERREUR: La variable d'environnement OPENROUTER_API_KEY n'est pas définie.")
        print("Veuillez créer un fichier .env à la racine du projet avec votre clé API OpenRouter.")
        print("Vous pouvez copier le fichier .env.example et le renommer en .env")
        return
    
    # Créer une configuration pour OpenRouter
    config = ModelConfig(
        provider="openrouter",
        model_name=os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free"),
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        headers={
            "HTTP-Referer": "https://github.com/yourusername/archon",
            "X-Title": "Archon AI Agent Test"
        }
    )
    
    # Créer le modèle
    model = ModelFactory.create_model(config)
    
    # Tester une requête simple
    print("\nEnvoi d'une requête de test à OpenRouter...")
    try:
        response = await model.run([{"role": "user", "content": "Dis-moi bonjour en français."}])
        print("\nRéponse du modèle:")
        print(response)
        print("\n✅ Test réussi! L'agent fonctionne correctement avec OpenRouter.")
    except Exception as e:
        print(f"\n❌ Erreur lors de l'appel à l'API OpenRouter: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Charger les variables d'environnement
    from dotenv import load_dotenv
    load_dotenv()
    
    # Exécuter le test
    asyncio.run(test_openrouter_integration())

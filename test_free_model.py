#!/usr/bin/env python3
"""
Test direct pour vérifier l'utilisation du modèle GRATUIT OpenRouter
"""
import sys
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le path pour importer les modules
sys.path.append('/app')
sys.path.append('/app/archon')

def test_openrouter_free_model():
    """Test direct du modèle OpenRouter gratuit"""
    try:
        # Import du fournisseur LLM
        from archon.llm_provider import llm_provider
        
        print("="*60)
        print("🧪 TEST MODÈLE OPENROUTER GRATUIT")
        print("="*60)
        
        # Afficher la configuration actuelle
        print(f"🔍 Fournisseur: {llm_provider.config.provider}")
        print(f"🔍 Modèle primaire: {llm_provider.config.primary_model}")
        print(f"🔍 Modèle reasoner: {llm_provider.config.reasoner_model}")
        print(f"🔍 Base URL: {llm_provider.config.base_url}")
        
        # Vérifier si c'est bien un modèle gratuit
        if ":free" in llm_provider.config.primary_model:
            print("✅ MODÈLE GRATUIT DÉTECTÉ (:free dans le nom)")
        else:
            print("❌ ATTENTION: MODÈLE PAYANT DÉTECTÉ (pas de :free)")
        
        # Test de PydanticAgent simple
        print("\n🧪 Test d'agent PydanticAI...")
        
        from pydantic_ai import Agent as PydanticAgent
        
        # Créer un agent simple
        test_agent = PydanticAgent(
            llm_provider.config.primary_model,
            system_prompt="You are a helpful assistant. Always respond briefly."
        )
        
        print(f"🔍 Agent créé avec modèle: {llm_provider.config.primary_model}")
        
        # Test de requête simple
        print("🔍 Envoi de requête de test...")
        result = test_agent.run_sync("Say 'Hello, this is a test with a free model' and nothing else.")
        
        print("="*60)
        print("📤 RÉPONSE REÇUE:")
        print(f"📝 {result.data}")
        print("="*60)
        
        if "test" in str(result.data).lower():
            print("✅ TEST RÉUSSI - Agent fonctionne")
        else:
            print("❌ TEST ÉCHOUÉ - Réponse inattendue")
            
        return True
        
    except Exception as e:
        print(f"❌ ERREUR DANS LE TEST: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_openrouter_free_model()

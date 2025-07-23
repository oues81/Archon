"""
Test Agent - Version simplifiée pour des tests de bout en bout
"""
import os
import json
import httpx
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestAgentConfig:
    """Configuration pour l'agent de test"""
    model: str = "google/gemma-7b-it"
    temperature: float = 0.7
    max_tokens: int = 1000
    use_openrouter: bool = True

class TestAgent:
    """Agent de test simplifié pour les tests de bout en bout"""
    
    def __init__(self, config: Optional[TestAgentConfig] = None):
        self.config = config or TestAgentConfig()
        self.base_url = (
            "https://openrouter.ai/api/v1/chat/completions" 
            if self.config.use_openrouter 
            else "http://localhost:11434/api/generate"
        )
        self.api_key = os.getenv("OPENROUTER_API_KEY") if self.config.use_openrouter else None
        
    def _prepare_payload(self, prompt: str) -> Dict[str, Any]:
        """Prépare la charge utile pour l'API"""
        if self.config.use_openrouter:
            return {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
        else:
            return {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }
    
    def _prepare_headers(self) -> Dict[str, str]:
        """Prépare les en-têtes pour la requête HTTP"""
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/archon",  # Remplacez par votre URL
            "X-Title": "Archon AI Agent"  # Nom de votre application
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def generate(self, prompt: str) -> Dict[str, Any]:
        """Génère une réponse à partir d'un prompt"""
        headers = self._prepare_headers()
        payload = self._prepare_payload(prompt)
        
        # Log de la requête
        print("\n=== Requête envoyée ===")
        print(f"URL: {self.base_url}")
        print(f"Headers: {json.dumps(headers, indent=2, default=str)}")
        print(f"Payload: {json.dumps(payload, indent=2, default=str)}")
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Envoyer la requête
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                
                # Log de la réponse
                print("\n=== Réponse reçue ===")
                print(f"Status: {response.status_code}")
                print(f"Headers: {dict(response.headers)}")
                print(f"Contenu: {response.text[:500]}")  # Affiche les 500 premiers caractères
                
                # Vérifier le code de statut HTTP
                response.raise_for_status()
                
                # Essayer de parser la réponse JSON
                try:
                    result = response.json()
                except json.JSONDecodeError as e:
                    error_msg = f"Erreur de décodage JSON: {str(e)}"
                    print(f"❌ {error_msg}")
                    return {
                        "error": error_msg,
                        "content": f"Erreur de décodage de la réponse du serveur: {str(e)}",
                        "raw_response": response.text
                    }
                
                # Vérifier la structure de la réponse OpenRouter
                if self.config.use_openrouter:
                    if "choices" not in result or not result["choices"]:
                        error_msg = "Format de réponse inattendu de l'API OpenRouter"
                        print(f"❌ {error_msg}")
                        return {
                            "error": error_msg,
                            "content": "Erreur: Format de réponse inattendu du serveur",
                            "raw_response": result
                        }
                    
                    return {
                        "content": result["choices"][0]["message"]["content"],
                        "model": result.get("model", "inconnu"),
                        "usage": result.get("usage", {})
                    }
                else:
                    # Format pour Ollama
                    if "response" not in result:
                        error_msg = "Format de réponse inattendu de l'API Ollama"
                        print(f"❌ {error_msg}")
                        return {
                            "error": error_msg,
                            "content": "Erreur: Format de réponse inattendu du serveur Ollama",
                            "raw_response": result
                        }
                    
                    return {
                        "content": result["response"],
                        "model": result.get("model", "inconnu"),
                        "usage": {"total_tokens": len(result["response"].split())}
                    }
        
        except httpx.HTTPStatusError as e:
            error_msg = f"Erreur HTTP {e.response.status_code}"
            print(f"❌ {error_msg}")
            
            # Essayer d'extraire plus d'informations sur l'erreur
            try:
                error_data = e.response.json()
                if 'error' in error_data and 'message' in error_data['error']:
                    error_msg = f"{error_msg}: {error_data['error']['message']}"
            except:
                pass
                
            return {
                "error": error_msg,
                "status_code": e.response.status_code,
                "content": f"Erreur lors de la communication avec l'API: {error_msg}",
                "response_text": e.response.text
            }
            
        except Exception as e:
            error_msg = f"Erreur inattendue: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "error": str(e),
                "content": f"Erreur inattendue: {str(e)}",
                "type": type(e).__name__
            }

# Exemple d'utilisation
async def test_example():
    # Configuration pour OpenRouter
    config = TestAgentConfig(
        model="google/gemma-7b-it",
        temperature=0.7,
        use_openrouter=True
    )
    
    agent = TestAgent(config)
    
    # Test de génération
    response = await agent.generate("Explique-moi le théorème de Pythagore en une phrase")
    print("Réponse:", response["content"])
    print("Modèle utilisé:", response["model"])
    print("Usage:", response.get("usage", "N/A"))

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_example())

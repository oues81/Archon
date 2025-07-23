"""
Client pour l'API OpenRouter
"""
import os
import json
import httpx
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

@dataclass
class OpenRouterConfig:
    """Configuration pour le client OpenRouter"""
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = os.getenv("OPENROUTER_DEFAULT_MODEL", "deepseek/deepseek-chat-v3-0324:free")
    temperature: float = 0.7
    max_tokens: int = 4000

class OpenRouterClient:
    """Client pour interagir avec l'API OpenRouter"""
    
    def __init__(self, config: Optional[OpenRouterConfig] = None):
        self.config = config or OpenRouterConfig()
        self.headers = self._prepare_headers()
    
    def _prepare_headers(self) -> Dict[str, str]:
        """Prépare les en-têtes pour les requêtes HTTP"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": "https://github.com/yourusername/archon",
            "X-Title": "Archon AI Agent"
        }
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Effectue une requête de chat completion à l'API OpenRouter
        
        Args:
            messages: Liste des messages du chat
            model: Modèle à utiliser (par défaut: celui de la configuration)
            temperature: Température pour la génération
            max_tokens: Nombre maximum de tokens à générer
            **kwargs: Arguments supplémentaires pour l'API
            
        Returns:
            Réponse de l'API OpenRouter
        """
        url = f"{self.config.base_url}/chat/completions"
        
        payload = {
            "model": model or self.config.default_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            **kwargs
        }
        
        # Log de la requête
        print("\n=== Requête OpenRouter ===")
        print(f"URL: {url}")
        print(f"Headers: {json.dumps(self.headers, indent=2, default=str)}")
        print(f"Payload: {json.dumps(payload, indent=2, default=str)}")
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    url,
                    headers=self.headers,
                    json=payload
                )
                
                # Log de la réponse
                print("\n=== Réponse OpenRouter ===")
                print(f"Status: {response.status_code}")
                print(f"Headers: {dict(response.headers)}")
                print(f"Contenu: {response.text[:500]}")
                
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            error_msg = f"Erreur HTTP {e.response.status_code}"
            try:
                error_data = e.response.json()
                if 'error' in error_data and 'message' in error_data['error']:
                    error_msg = f"{error_msg}: {error_data['error']['message']}"
            except:
                pass
            raise Exception(error_msg) from e
            
        except Exception as e:
            raise Exception(f"Erreur lors de l'appel à l'API OpenRouter: {str(e)}") from e

# Singleton pour une utilisation facile dans tout le projet
openrouter_client = OpenRouterClient()

# Exemple d'utilisation
async def example_usage():
    """Exemple d'utilisation du client OpenRouter"""
    client = OpenRouterClient()
    response = await client.chat_completion(
        messages=[
            {"role": "user", "content": "Bonjour, comment ça va ?"}
        ]
    )
    return response

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())

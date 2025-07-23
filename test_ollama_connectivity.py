import os
import requests
import sys

def test_ollama_connection():
    """Teste la connexion à Ollama et liste les modèles disponibles."""
    base_url = os.getenv('BASE_URL', 'http://host.docker.internal:11434')
    health_url = f"{base_url}/api/tags"
    
    print("=== Test de connectivité à Ollama ===\n")
    print(f"Vérification de la connexion à {base_url}")
    
    try:
        # Vérification de la santé de l'API
        response = requests.get(health_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Connexion à Ollama réussie !")
            
            # Afficher la liste des modèles disponibles
            models = data.get('models', [])
            if models:
                print("\nModèles disponibles :")
                for model in models:
                    size_mb = int(model.get('size', 0)) / (1024 * 1024)
                    print(f"- {model.get('name', 'N/A')} ({size_mb:.1f} MB)")
            else:
                print("\nAucun modèle trouvé.")
            
            return True
            
        else:
            print(f"❌ Erreur: Code de statut {response.status_code}")
            print(response.text)
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Échec de la connexion à Ollama: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_ollama_connection()
    sys.exit(0 if success else 1)

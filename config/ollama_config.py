"""
Configuration pour Ollama avec détection automatique de l'environnement.
"""
import os
import socket
from typing import Optional

def detect_ollama_url() -> str:
    """
    Détecte automatiquement l'URL d'Ollama en fonction de l'environnement.
    
    Returns:
        str: URL de base pour l'API Ollama
    """
    # 1. Vérifier si une URL est explicitement définie
    if "OLLAMA_BASE_URL" in os.environ:
        return os.environ["OLLAMA_BASE_URL"].rstrip('/')
    
    # 2. Détecter si on est dans un conteneur
    is_in_container = os.path.exists('/.dockerenv')
    
    # 3. Essayer différentes URLs en fonction de l'environnement
    test_urls = []
    
    if is_in_container:
        # Dans un conteneur, essayer d'abord le service Docker
        test_urls.append("http://ollama:11434")
    else:
        # En local, essayer d'abord localhost
        test_urls.append("http://localhost:11434")
    
    # Ajouter les alternatives
    test_urls.extend([
        "http://host.docker.internal:11434",
        "http://127.0.0.1:11434"
    ])
    
    # Tester chaque URL
    for url in test_urls:
        try:
            # Essayer de se connecter à l'API d'Ollama
            import requests
            response = requests.get(f"{url}/api/tags", timeout=2)
            if response.status_code == 200 and 'models' in response.json():
                return url.rstrip('/')
        except (requests.RequestException, ValueError):
            continue
    
    # Si aucune URL ne fonctionne, retourner une valeur par défaut
    return "http://localhost:11434"

# Configuration par défaut
OLLAMA_CONFIG = {
    "base_url": detect_ollama_url(),
    "timeout": 30.0,
    "max_retries": 3,
    "verify_ssl": False
}

def get_ollama_config() -> dict:
    """Retourne la configuration Ollama."""
    return OLLAMA_CONFIG

def update_ollama_config(**kwargs):
    """Met à jour la configuration Ollama."""
    OLLAMA_CONFIG.update(kwargs)
    
    # Si l'URL est mise à jour, s'assurer qu'elle ne se termine pas par un /
    if 'base_url' in kwargs:
        OLLAMA_CONFIG['base_url'] = OLLAMA_CONFIG['base_url'].rstrip('/')

#!/usr/bin/env python3
"""
Script pour tester la disponibilité du service Archon
"""
import requests
import time
import sys
import os

def check_service_health(url, max_retries=10, retry_delay=2):
    """
    Vérifie la disponibilité d'un service en testant son endpoint /health
    
    Args:
        url: URL de base du service
        max_retries: Nombre maximum de tentatives
        retry_delay: Délai entre les tentatives en secondes
    
    Returns:
        bool: True si le service est disponible, False sinon
    """
    print(f"Vérification de la disponibilité du service à {url}/health")
    
    for i in range(1, max_retries + 1):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Service disponible après {i} tentative(s)")
                return True
        except requests.exceptions.RequestException as e:
            print(f"Tentative {i}/{max_retries}: {str(e)}")
        
        print(f"En attente... ({i}/{max_retries})")
        time.sleep(retry_delay)
    
    print(f"❌ Service non disponible après {max_retries} tentatives")
    return False

if __name__ == "__main__":
    # URL du service à tester (utilise l'argument en ligne de commande ou la valeur par défaut)
    service_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8110"
    
    # Nombre de tentatives et délai entre les tentatives
    max_tries = int(os.getenv("MAX_HEALTH_RETRIES", "10"))
    delay = int(os.getenv("HEALTH_RETRY_DELAY", "3"))
    
    # Vérifier la disponibilité du service
    if check_service_health(service_url, max_tries, delay):
        sys.exit(0)  # Succès
    else:
        sys.exit(1)  # Échec
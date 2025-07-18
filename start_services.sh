#!/bin/bash

# Ce script démarre les services nécessaires pour Archon
# Note: La configuration du réseau Docker doit être gérée au niveau de l'hôte

# Définir le répertoire de base
BASE_DIR="/app"
WORKSPACE_DIR="/app/workbench"
LOG_DIR="/app/logs"

# Créer le répertoire de logs s'il n'existe pas
mkdir -p "$LOG_DIR"

# Afficher les variables d'environnement pour le débogage
echo "=== Variables d'environnement ==="
echo "PYTHONPATH: $PYTHONPATH"
echo "LLM_PROVIDER: $LLM_PROVIDER"
echo "BASE_URL: $BASE_URL"
echo "EMBEDDING_BASE_URL: $EMBEDDING_BASE_URL"
echo "OLLAMA_HOST: $OLLAMA_HOST"
echo "DEFAULT_WORKSPACE_PATH: $DEFAULT_WORKSPACE_PATH"
echo "==============================="

# Test de la connectivité à Ollama
echo "=== Test de la connectivité à Ollama ==="
python /app/test_ollama_connectivity.py
echo "======================================"

# Vérifier et créer le répertoire de travail si nécessaire
if [ ! -d "$WORKSPACE_DIR" ]; then
  echo "Création du répertoire de travail: $WORKSPACE_DIR"
  mkdir -p "$WORKSPACE_DIR"
  chmod 777 "$WORKSPACE_DIR"
fi

# Copier les fichiers de configuration si nécessaire
if [ ! -f "$WORKSPACE_DIR/env_vars.json" ] && [ -f "$WORKSPACE_DIR/env_vars.json.example" ]; then
    echo "Création du fichier de configuration par défaut..."
    cp "$WORKSPACE_DIR/env_vars.json.example" "$WORKSPACE_DIR/env_vars.json"
fi

# Démarrer le service principal en arrière-plan
echo "Démarrage du service principal Archon..."
cd /app

# S'assurer que les répertoires nécessaires sont dans le PYTHONPATH
export PYTHONPATH="/app:/app/archon:$PYTHONPATH"

# Installer uvicorn s'il n'est pas déjà installé
echo "Vérification de l'installation d'Uvicorn..."
if ! command -v uvicorn &> /dev/null; then
    echo "Uvicorn n'est pas installé. Installation en cours..."
    pip install uvicorn
    echo "Uvicorn installé avec succès."
fi

# Démarrer uvicorn directement sans patches
echo "Démarrage d'Uvicorn..."
uvicorn graph_service:app --host 0.0.0.0 --port 8110 --reload --log-level debug > "$LOG_DIR/uvicorn.log" 2>&1 &
UVICORN_PID=$!

echo "UVICORN_PID: $UVICORN_PID"

# Attendre que le service principal soit prêt
echo "Attente du démarrage du service principal..."
MAX_RETRIES=30
RETRY_DELAY=2

for ((i=1; i<=MAX_RETRIES; i++)); do
  if curl -s http://localhost:8110/health >/dev/null 2>&1; then
    echo "Service principal démarré avec succès après $i tentatives"
    break
  fi
  
  # Vérifier si uvicorn est toujours en cours d'exécution
  if ! ps -p $UVICORN_PID > /dev/null; then
    echo "Erreur: Le processus uvicorn s'est arrêté de manière inattendue"
    echo "=== Dernières lignes des logs uvicorn ==="
    tail -n 50 "$LOG_DIR/uvicorn.log"
    echo "========================================"
    exit 1
  fi
  
  # Afficher les logs si disponible
  if [ -f "$LOG_DIR/uvicorn.log" ]; then
    echo "=== Dernières lignes des logs uvicorn (tentative $i/$MAX_RETRIES) ==="
    tail -n 5 "$LOG_DIR/uvicorn.log"
    echo "========================================"
  fi
  
  echo "En attente du service principal... (tentative $i/$MAX_RETRIES)"
  sleep $RETRY_DELAY
done

# Vérifier si le service principal est en cours d'exécution
if ! curl -s http://localhost:8110/health >/dev/null 2>&1; then
  echo "Erreur: Impossible de démarrer le service principal après $MAX_RETRIES tentatives."
  echo "=== Dernières lignes des logs uvicorn ==="
  tail -n 50 "$LOG_DIR/uvicorn.log"
  echo "========================================"
  exit 1
fi

# Démarrer l'interface utilisateur Streamlit
echo "Démarrage de l'interface utilisateur Streamlit..."
cd /app
streamlit run streamlit_ui.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=true --server.enableXsrfProtection=false --server.fileWatcherType none > "$LOG_DIR/streamlit.log" 2>&1 &

# Afficher les logs en continu
tail -f "$LOG_DIR/uvicorn.log" "$LOG_DIR/streamlit.log"

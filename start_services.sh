#!/bin/bash

# Ce script démarre les services nécessaires pour Archon
# Note: La configuration du réseau Docker doit être gérée au niveau de l'hôte

# Définir le répertoire de base
BASE_DIR="/app"
WORKSPACE_DIR="/app/workbench"
LOG_DIR="/app/logs"

# Créer les répertoires nécessaires
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$LOG_DIR"

# Configuration des variables d'environnement pour la journalisation
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export UVICORN_LOG_LEVEL="${UVICORN_LOG_LEVEL:-info}"

# Afficher les variables d'environnement pour le débogage
echo "=== Variables d'environnement ==="
echo "PYTHONPATH: ${PYTHONPATH:-Non défini}"
echo "LLM_PROVIDER: ${LLM_PROVIDER:-Non défini}"
echo "BASE_URL: ${BASE_URL:-Non défini}"
echo "EMBEDDING_BASE_URL: ${EMBEDDING_BASE_URL:-Non défini}"
echo "OLLAMA_HOST: ${OLLAMA_HOST:-Non défini}"
echo "DEFAULT_WORKSPACE_PATH: ${DEFAULT_WORKSPACE_PATH:-Non défini}"
echo "LOG_LEVEL: $LOG_LEVEL"
echo "UVICORN_LOG_LEVEL: $UVICORN_LOG_LEVEL"
echo "================================"

# Tester la connectivité à Ollama si nécessaire
if [ "$LLM_PROVIDER" = "Ollama" ] || [ -z "$LLM_PROVIDER" ]; then
    echo "=== Test de la connectivité à Ollama ==="
    if [ -f "/app/test_ollama_connectivity.py" ]; then
        python /app/test_ollama_connectivity.py
    else
        echo "Fichier test_ollama_connectivity.py non trouvé, test ignoré"
    fi
    echo "======================================"
fi

# Vérifier et créer le répertoire de travail si nécessaire
if [ ! -d "$WORKSPACE_DIR" ]; then
    echo "Création du répertoire de travail: $WORKSPACE_DIR"
    mkdir -p "$WORKSPACE_DIR"
    chmod 777 "$WORKSPACE_DIR"
fi

# Vérifier la connectivité au serveur MCP
check_mcp_server_connectivity() {
    echo "=== Vérification de la connectivité au serveur MCP ==="
    
    if [ -z "$MCP_SERVER_URL" ]; then
        echo "[AVERTISSEMENT] La variable d'environnement MCP_SERVER_URL n'est pas définie. Le test de connectivité est ignoré."
        echo "=================================================="
        return
    fi

    MAX_RETRIES=15
    RETRY_COUNT=0
    
    echo "En attente du serveur MCP à l'adresse : $MCP_SERVER_URL"

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        # Utiliser wget pour vérifier la connectivité. Le serveur MCP est dans un autre conteneur.
        if curl --silent --fail --connect-timeout 5 "$MCP_SERVER_URL/health" > /dev/null; then
            echo "✅ Connexion au serveur MCP réussie !"
            echo "=================================================="
            return
        fi
        
        RETRY_COUNT=$((RETRY_COUNT+1))
        echo "Tentative de connexion au serveur MCP... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 3
    done
    
    echo "[ERREUR] Impossible de se connecter au serveur MCP après $MAX_RETRIES tentatives."
    echo "Veuillez vous assurer que le service 'mcp-server' est démarré et accessible."
    echo "======================================================================"
    # Ne pas quitter, car le service principal peut fonctionner sans MCP.
}

# Vérifier la connectivité au serveur MCP
check_mcp_server_connectivity

# Copier les fichiers de configuration si nécessaire
if [ ! -f "$WORKSPACE_DIR/env_vars.json" ] && [ -f "$WORKSPACE_DIR/env_vars.json.example" ]; then
    echo "Création du fichier de configuration par défaut..."
    cp "$WORKSPACE_DIR/env_vars.json.example" "$WORKSPACE_DIR/env_vars.json"
fi

# Configurer le PYTHONPATH
export PYTHONPATH="/app:/app/archon:${PYTHONPATH:-}"

# Installer uvicorn s'il n'est pas déjà installé
echo "Vérification des dépendances..."
if ! command -v uvicorn &> /dev/null; then
    echo "Uvicorn n'est pas installé. Installation en cours..."
    pip install uvicorn
fi

# Démarrer le service principal avec la configuration de journalisation
echo "Démarrage du service backend Archon..."
uvicorn archon.graph_service:app \
    --host 0.0.0.0 \
    --port 8110 \
    --log-config /app/uvicorn_logging.ini \
    --log-level "$UVICORN_LOG_LEVEL" \
    > "$LOG_DIR/uvicorn_stdout.log" 2> "$LOG_DIR/uvicorn_stderr.log" &
UVICORN_PID=$!

echo "Service backend démarré avec le PID: $UVICORN_PID"

# Attendre que le service principal soit prêt
echo "Attente du démarrage du service principal..."
MAX_RETRIES=30
RETRY_DELAY=2
SERVICE_STARTED=0

for ((i=1; i<=MAX_RETRIES; i++)); do
    if python -c "import requests; requests.get('http://localhost:8110/health')" >/dev/null 2>&1; then
        echo "Service principal démarré avec succès après $i secondes"
        SERVICE_STARTED=1
        break
    fi
    
    # Vérifier si uvicorn est toujours en cours d'exécution
    if ! ps -p $UVICORN_PID > /dev/null; then
        echo "Erreur: Le processus uvicorn s'est arrêté de manière inattendue"
        echo "=== Dernières lignes des logs d'erreur ==="
        tail -n 20 "$LOG_DIR/uvicorn_stderr.log"
        echo "========================================="
        exit 1
    fi
    
    echo "En attente du service principal... ($i/$MAX_RETRIES)"
    sleep $RETRY_DELAY
done

# Vérifier si le service principal est en cours d'exécution
if [ $SERVICE_STARTED -eq 0 ]; then
    echo "Erreur: Impossible de démarrer le service principal après $MAX_RETRIES secondes."
    echo "=== Dernières lignes des logs d'erreur ==="
    tail -n 50 "$LOG_DIR/uvicorn_stderr.log"
    echo "========================================="
    exit 1
fi

# Démarrer l'interface utilisateur Streamlit avec journalisation configurée
echo "Démarrage de l'interface utilisateur Streamlit..."
cd /app
python -m streamlit run streamlit_ui.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=true \
    --server.enableXsrfProtection=false \
    --server.fileWatcherType none \
    --logger.level=info \
    > "$LOG_DIR/streamlit_stdout.log" 2> "$LOG_DIR/streamlit_stderr.log" &
STREAMLIT_PID=$!

echo "Interface Streamlit démarrée avec le PID: $STREAMLIT_PID"

# Configurer un gestionnaire de signaux pour arrêter correctement les processus
cleanup() {
    echo "Arrêt des services..."
    kill -TERM $UVICORN_PID $STREAMLIT_PID 2>/dev/null
    wait $UVICORN_PID $STREAMLIT_PID 2>/dev/null
    echo "Services arrêtés."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Afficher les logs en continu
echo "=== Démarrage de la surveillance des logs ==="
echo "Appuyez sur Ctrl+C pour arrêter les services"

tail -f "$LOG_DIR/uvicorn_stdout.log" "$LOG_DIR/uvicorn_stderr.log" "$LOG_DIR/streamlit_stdout.log" "$LOG_DIR/streamlit_stderr.log"

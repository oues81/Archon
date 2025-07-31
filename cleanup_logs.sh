#!/bin/bash

# Script de nettoyage des anciens fichiers de logs
# Conserve uniquement les 10 derniers fichiers de chaque type

LOG_DIR="/app/logs"

# Fonction pour nettoyer les anciens fichiers de logs
cleanup_old_logs() {
    local pattern=$1
    local keep=${2:-10}  # Par défaut, conserve les 10 fichiers les plus récents
    
    echo "Nettoyage des anciens fichiers correspondant à: $pattern"
    
    # Vérifier si des fichiers correspondent au motif
    if compgen -G "$LOG_DIR/$pattern" > /dev/null; then
        # Trier par date de modification (du plus récent au plus ancien) et supprimer les plus anciens
        find "$LOG_DIR" -name "$pattern" -type f -printf '%T@ %p\n' | \
            sort -nr | \
            tail -n +$((keep + 1)) | \
            cut -d' ' -f2- | \
            xargs -r rm -f
    else
        echo "Aucun fichier trouvé correspondant à: $pattern"
    fi
}

# Nettoyer les différents types de logs
cleanup_old_logs "uvicorn_*.log*" 10
cleanup_old_logs "streamlit_*.log*" 10
cleanup_old_logs "access.log*" 5
cleanup_old_logs "archon.log*" 10

echo "Nettoyage des logs terminé"

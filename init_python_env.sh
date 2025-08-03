#!/bin/bash

# Script d'initialisation pour configurer correctement l'environnement Python
# Ce script est exécuté au démarrage du conteneur

echo "=== Configuration de l'environnement Python ==="

# Ajouter les chemins au PYTHONPATH
export PYTHONPATH="/app:/app/src:${PYTHONPATH}"

# Créer un fichier .pth dans le site-packages pour assurer que les chemins sont toujours disponibles
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Création du fichier archon.pth dans ${SITE_PACKAGES}"
echo "/app" > "${SITE_PACKAGES}/archon.pth"
echo "/app/src" >> "${SITE_PACKAGES}/archon.pth"

# Vérification des chemins
echo "=== Vérification de l'environnement Python ==="
echo "PYTHONPATH: ${PYTHONPATH}"
python -c "import sys; print('Python path:', sys.path)"
python -c "import sys; print('Python executable:', sys.executable)"

echo "=== Configuration terminée ==="

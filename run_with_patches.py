import sys
import os

# Ajouter le répertoire des patches au début du PYTHONPATH
sys.path.insert(0, os.path.abspath('/app/archon'))

# Importer et appliquer les patches avant tout autre import
try:
    from archon.patches import apply_all_patches
    if not apply_all_patches():
        print("Échec de l'application des patches", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"Erreur lors de l'application des patches: {e}", file=sys.stderr)
    sys.exit(1)

# Importer et exécuter l'application principale
from graph_service import app as application

if __name__ == "__main__":
    import uvicorn
    import logging
    import asyncio
    import os
    import sys

    # Désactiver complètement les logs de watchfiles.main
    logging.getLogger('watchfiles.main').setLevel(logging.ERROR)
    logging.getLogger('watchfiles').setLevel(logging.ERROR)

    # Configuration complète de Uvicorn pour éviter la pollution des logs
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["watchfiles"] = {"level": "ERROR", "handlers": ["default"], "propagate": False}
    log_config["loggers"]["watchfiles.main"] = {"level": "ERROR", "handlers": ["default"], "propagate": False}

    # Configuration Uvicorn avec paramètres optimisés pour réduire les messages watchfiles
    uvicorn.run(
        "graph_service:app", 
        host="0.0.0.0", 
        port=8110, 
        reload=True,
        reload_delay=5.0,  
        reload_excludes=["*.log", "__pycache__/*", "*.pyc", ".git/*", "*.md"],
        log_level="warning",  
        log_config=log_config,
    )

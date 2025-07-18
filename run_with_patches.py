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
    uvicorn.run("graph_service:app", host="0.0.0.0", port=8110, reload=True, log_level="debug")

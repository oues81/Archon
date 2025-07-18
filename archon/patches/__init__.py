"""
Package de patches pour l'application Archon.

Ce module est conservé pour la rétrocompatibilité mais ne contient plus de patches
maintenant que les dépendances ont été mises à jour.
"""
import logging

# Configurer le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_all_patches():
    """
    Cette fonction est conservée pour la rétrocompatibilité mais ne fait plus rien
    car les dépendances ont été mises à jour pour résoudre les problèmes.
    
    Returns:
        bool: Toujours True car il n'y a plus de patches à appliquer
    """
    logger.info("Aucun patch à appliquer - les dépendances ont été mises à jour")
    return True

# Ne pas appliquer automatiquement les patches
try:
    if __name__ == "__main__":
        apply_all_patches()
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation des patches: {e}")

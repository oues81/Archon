#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'exécution directe du crawler MCP.
Permet de tester et d'exécuter le crawler sans dépendre d'autres composants.
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_crawler_runner")

# Charger les variables d'environnement
load_dotenv()

# Ajouter le chemin vers le module archon
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importer les modules nécessaires depuis archon
from archon.crawl_mcp_docs import main, CrawlProgressTracker

def progress_callback(data):
    """Affiche la progression du crawler."""
    status = data.get("status", "")
    message = data.get("message", "")
    urls_processed = data.get("urls_processed", 0)
    urls_found = data.get("urls_found", 0)
    chunks_stored = data.get("chunks_stored", 0)
    
    if status == "completed":
        logger.info(f"Crawling terminé: {urls_processed}/{urls_found} URLs traitées, {chunks_stored} chunks stockés")
    else:
        logger.info(f"Progression: {urls_processed}/{urls_found} URLs, {chunks_stored} chunks - {message}")

def run_crawler():
    """Exécute le crawler MCP."""
    logger.info("Démarrage du crawler MCP...")
    
    # Créer un tracker avec callback pour suivre la progression
    tracker = CrawlProgressTracker(progress_callback)
    
    # Exécuter le crawler avec la version synchrone
    from archon.crawl_mcp_docs import main_with_requests
    main_with_requests(tracker)
    success = True
    
    if success:
        logger.info("Crawling MCP terminé avec succès")
    else:
        logger.error("Erreur lors du crawling MCP")
    
    return success

if __name__ == "__main__":
    run_crawler()

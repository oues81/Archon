#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'exécution directe du crawler Windsurf Workflows & Context Engineering.
Permet de tester et d'exécuter le crawler sans dépendre d'autres composants.
"""

import os
import sys
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("windsurf_crawler_runner")

# Charger les variables d'environnement
load_dotenv()

# Ajouter le chemin vers le module archon
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from archon.crawl_windsurf_workflows_docs import main_with_requests, CrawlProgressTracker

def progress_callback(data):
    status = data.get("event") or data.get("status") or ""
    message = data.get("message", "")
    stats = data.get("stats") or {}
    urls_processed = stats.get("urls_processed", 0)
    urls_found = stats.get("urls_found", 0)
    chunks_stored = stats.get("chunks_stored", 0)
    logger.info(f"[{status}] {message} | {urls_processed}/{urls_found} URLs, {chunks_stored} chunks")


def run_crawler():
    logger.info("Démarrage du crawler Windsurf Workflows & Context Engineering...")
    tracker = CrawlProgressTracker(progress_callback)
    main_with_requests(tracker)
    logger.info("Crawling Windsurf terminé")
    return True

if __name__ == "__main__":
    run_crawler()

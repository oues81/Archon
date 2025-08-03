#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour vérifier le fonctionnement du crawler MCP.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from archon.crawl_mcp_docs import (
    get_mcp_urls,
    ProcessedChunk,
    CrawlProgressTracker,
    fetch_url_content,
    chunk_text,
    get_title_and_summary
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_mcp_crawler")

# Charger les variables d'environnement depuis .env
load_dotenv()

async def test_get_urls():
    """Tester la récupération des URLs MCP."""
    urls = get_mcp_urls()
    logger.info(f"URLs récupérées: {len(urls)}")
    for url in urls:
        logger.info(f"URL: {url}")
    
    return urls

async def test_fetch_content(urls):
    """Tester la récupération du contenu depuis les URLs."""
    for url in urls[:2]:  # Tester seulement les 2 premières URLs
        logger.info(f"Récupération du contenu depuis {url}...")
        content = await fetch_url_content(url)
        if content:
            logger.info(f"Contenu récupéré: {len(content)} caractères")
            logger.info(f"Début du contenu: {content[:200]}...")
            
            # Tester le chunking
            chunks = chunk_text(content)
            logger.info(f"Document divisé en {len(chunks)} chunks")
            
            # Tester l'extraction des titres et résumés
            for i, chunk in enumerate(chunks[:2]):  # Tester seulement les 2 premiers chunks
                title, summary = get_title_and_summary(chunk, url, i)
                logger.info(f"Chunk {i} - Titre: {title}")
                logger.info(f"Chunk {i} - Résumé: {summary}")
        else:
            logger.error(f"Échec de récupération du contenu depuis {url}")

async def test_tracker():
    """Tester le tracker de progression."""
    def progress_callback(data):
        logger.info(f"Progression: {data}")
    
    tracker = CrawlProgressTracker(progress_callback)
    tracker.start()
    tracker.urls_found = 10
    tracker.urls_processed = 5
    tracker.urls_succeeded = 4
    tracker.urls_failed = 1
    tracker.chunks_stored = 20
    tracker.log("Test du tracker")
    tracker.complete()
    
    logger.info(f"État final du tracker: {tracker.get_status()}")

async def main():
    """Fonction principale de test."""
    logger.info("Démarrage des tests du crawler MCP")
    
    # Tester la récupération des URLs
    urls = await test_get_urls()
    
    # Tester la récupération et le traitement du contenu
    await test_fetch_content(urls)
    
    # Tester le tracker de progression
    await test_tracker()
    
    logger.info("Tests terminés")

if __name__ == "__main__":
    asyncio.run(main())

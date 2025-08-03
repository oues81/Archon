#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test simplifié du crawler MCP sans dépendances externes.
Ce script teste uniquement les fonctions de récupération et de traitement
sans nécessiter OpenAI ou Supabase.
"""

import os
import asyncio
import logging
from archon.crawl_mcp_docs import (
    get_mcp_urls,
    convert_github_url_to_raw,
    fetch_url_content,
    chunk_text,
    get_title_and_summary,
    ProcessedChunk
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_test")

async def test_url_conversion():
    """Teste la conversion des URLs GitHub en URLs raw."""
    github_urls = [
        "https://github.com/modelcontextprotocol/specification/blob/main/SPECIFICATION.md",
        "https://github.com/modelcontextprotocol/docs/blob/main/src/index.md"
    ]
    
    for url in github_urls:
        raw_url = convert_github_url_to_raw(url)
        logger.info(f"URL originale: {url}")
        logger.info(f"URL raw: {raw_url}")
        
        # Vérifier que l'URL a bien été convertie
        assert "raw.githubusercontent.com" in raw_url, f"Conversion incorrecte pour {url}"

async def test_fetch_content():
    """Teste la récupération du contenu depuis les URLs convertis."""
    urls = get_mcp_urls()[:2]  # Prendre seulement 2 URLs pour le test
    
    for url in urls:
        logger.info(f"Récupération du contenu depuis {url}...")
        content = await fetch_url_content(url)
        
        if content:
            logger.info(f"✅ Contenu récupéré: {len(content)} caractères")
            logger.info(f"Début: {content[:100]}...")
        else:
            logger.error(f"❌ Échec de récupération pour {url}")

async def test_chunking():
    """Teste le découpage du contenu en chunks."""
    # Récupérer le contenu d'une URL
    url = get_mcp_urls()[0]
    content = await fetch_url_content(url)
    
    if not content:
        logger.error(f"❌ Impossible de récupérer le contenu pour tester le chunking")
        return
    
    # Découper en chunks
    chunks = chunk_text(content)
    logger.info(f"✅ Document divisé en {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks[:2]):  # Afficher seulement les 2 premiers chunks
        logger.info(f"Chunk {i+1}/{len(chunks)}: {len(chunk)} caractères")
        
        # Extraire le titre et le résumé
        title, summary = get_title_and_summary(chunk, url, i)
        logger.info(f"  Titre: {title}")
        logger.info(f"  Résumé: {summary[:100]}...")
        
        # Créer un objet ProcessedChunk
        processed = ProcessedChunk(
            url=url,
            chunk_number=i,
            title=title,
            summary=summary,
            content=chunk,
            category="MCP Documentation",
            embedding=None  # Pas d'embedding pour ce test
        )
        
        logger.info(f"  Objet ProcessedChunk créé avec succès")

async def main():
    """Fonction principale de test."""
    logger.info("=== DÉBUT DES TESTS DU CRAWLER MCP ===")
    
    logger.info("--- Test 1: Conversion d'URLs GitHub ---")
    await test_url_conversion()
    
    logger.info("\n--- Test 2: Récupération du contenu ---")
    await test_fetch_content()
    
    logger.info("\n--- Test 3: Chunking et traitement ---")
    await test_chunking()
    
    logger.info("\n=== TESTS TERMINÉS ===")

if __name__ == "__main__":
    asyncio.run(main())

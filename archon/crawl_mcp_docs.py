#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Module pour crawler et indexer la documentation MCP."""

import asyncio
import os
import json
import logging
import re
import html2text
import threading
import requests
from datetime import datetime
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple
from archon.utils.utils import get_env_var
import hashlib
from openai import AsyncOpenAI
from supabase import create_client

def extract_title_from_chunk(text: str) -> Optional[str]:
    """Extraire un titre pertinent à partir du texte d'un chunk.
    
    Cette fonction tente d'identifier un titre logique dans le contenu en cherchant
    des en-têtes markdown ou la première ligne significative.
    """
    if not text or not isinstance(text, str):
        return None
        
    # Rechercher les en-têtes markdown (# Titre, ## Titre, etc.)
    header_pattern = re.compile(r'^(#{1,6})\s+(.+?)\s*$', re.MULTILINE)
    headers = header_pattern.findall(text)
    
    if headers:
        # Prendre l'en-tête de plus haut niveau (moins de # au début)
        headers.sort(key=lambda x: len(x[0]))
        return headers[0][1].strip()
    
    # Si pas d'en-tête, prendre la première ligne non vide
    lines = text.split('\n')
    for line in lines:
        clean_line = line.strip()
        if clean_line and len(clean_line) > 3:  # Éviter les lignes trop courtes
            # Limiter la longueur du titre
            return clean_line[:100] + ('...' if len(clean_line) > 100 else '')
    
    return None

def extract_summary_from_chunk(text: str) -> Optional[str]:
    """Extraire un résumé pertinent à partir du texte d'un chunk.
    
    Cette fonction extrait les premières phrases significatives du texte comme résumé.
    """
    if not text or not isinstance(text, str):
        return None
        
    # Supprimer les en-têtes markdown pour ne garder que le contenu
    text_no_headers = re.sub(r'^#{1,6}\s+.+?\s*$', '', text, flags=re.MULTILINE)
    
    # Nettoyer le texte et obtenir les premiers caractères
    clean_text = ' '.join(text_no_headers.split())
    if not clean_text:
        return None
        
    # Limiter à environ 150-200 caractères pour le résumé
    summary_length = min(200, len(clean_text))
    summary = clean_text[:summary_length]
    
    # S'assurer de terminer sur une phrase complète si possible
    if summary_length < len(clean_text):
        # Chercher le dernier point dans notre extrait
        last_period = summary.rfind('.')
        if last_period > 0 and last_period + 1 < summary_length:
            summary = summary[:last_period + 1]
        else:
            summary += '...'
    
    return summary

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_crawler")

# Initialisation du convertisseur HTML vers Markdown
html_converter = html2text.HTML2Text()
html_converter.ignore_links = False
html_converter.ignore_images = False
html_converter.ignore_tables = False
html_converter.body_width = 0  # Pas de césure

# Initialisation des variables globales
# Définir la valeur par défaut du modèle d'embedding (Ollama utilise nomic-embed-text)
# Read embedding model from profiles/env, default to 1536-dim OpenAI model
embedding_model = get_env_var('EMBEDDING_MODEL') or 'text-embedding-3-small'

# Initialisation des clients
embedding_client = None
supabase = None
llm_client = None
# État d'initialisation (thread-safe)
_clients_init_lock = threading.Lock()
_clients_initialized = False

# Allowlist for authoritative domains and org paths (expanded)
ALLOWED_DOMAINS = {
    # Official site & spec
    "modelcontextprotocol.io",
    "spec.modelcontextprotocol.io",
    # Official GitHub org
    "github.com",
    # IDE clients / docs
    "code.visualstudio.com",
    "docs.windsurf.com",
    # Ecosystem vendors
    "supabase.com",
    "openai.github.io",
    "platform.openai.com",
    "anthropic.com",
    "docs.anthropic.com",
    # Libraries
    "docs.llamaindex.ai",
}

def is_allowed_url(url: str) -> bool:
    """Return True if the URL is within allowed domains and expected org paths.
    - Allows official sites and GitHub org 'modelcontextprotocol' READMEs
    - Excludes obvious binary assets
    """
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path = parsed.path.lower()

        if host not in ALLOWED_DOMAINS:
            return False

        # Exclude common binary/static assets
        if any(path.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".svg", ".gif", ".pdf", ".zip")):
            return False

        # If GitHub, limit to the official org and text pages (README.md, .md paths)
        if host == "github.com":
            # Expect /modelcontextprotocol/<repo>/blob/main/README.md or similar .md files
            parts = [p for p in path.split("/") if p]
            if len(parts) < 2 or parts[0] != "modelcontextprotocol":
                return False
            if not path.endswith(".md"):
                return False
        return True
    except Exception:
        return False

def init_clients():
    """Initialise les clients d'API nécessaires."""
    global embedding_client, supabase, llm_client, _clients_initialized
    # Empêcher les initialisations concurrentes
    if _clients_initialized:
        return True
    with _clients_init_lock:
        if _clients_initialized:
            return True
    
    try:
        # Afficher toutes les variables d'environnement disponibles pour le débogage
        logger.info(f"EMBEDDING_PROVIDER: {os.environ.get('EMBEDDING_PROVIDER')}")
        logger.info(f"EMBEDDING_BASE_URL: {os.environ.get('EMBEDDING_BASE_URL')}")
        logger.info(f"SUPABASE_URL: {os.environ.get('SUPABASE_URL')}")
        logger.info(f"SUPABASE_SERVICE_KEY: {'présente' if os.environ.get('SUPABASE_SERVICE_KEY') else 'absente'}")
        
        # Configuration pour Ollama (en utilisant la variable depuis le .env ou la valeur pour Docker)
        embedding_base_url = os.environ.get('EMBEDDING_BASE_URL')
        
        # Normaliser l'URL pour Docker et s'assurer du suffixe /api
        if (not embedding_base_url 
            or 'host.docker.internal' in embedding_base_url 
            or 'localhost' in embedding_base_url 
            or 'ollama:11434' in embedding_base_url):
            embedding_base_url = "http://ollama:11434/api"
            logger.info(f"Normalisation de l'URL Ollama pour Docker: {embedding_base_url}")
        
        # S'assurer que le suffixe /api est présent
        if not embedding_base_url.rstrip('/').endswith('/api'):
            embedding_base_url = embedding_base_url.rstrip('/') + '/api'
            logger.info(f"Ajout du suffixe /api à l'URL Ollama: {embedding_base_url}")
        
        logger.info(f"Initialisation du client Ollama avec l'URL: {embedding_base_url}")
        embedding_client = AsyncOpenAI(base_url=embedding_base_url, api_key="ollama")
        # Utiliser le même endpoint OpenAI-compatible pour le LLM (chat completions)
        llm_client = AsyncOpenAI(base_url=embedding_base_url, api_key="ollama")
        logger.info("Clients OpenAI-compatibles initialisés avec succès (embeddings et LLM)")
        
        # Initialisation du client Supabase avec les vraies clés du fichier .env
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
        
        # Vérifier que les URLs ne sont pas des placeholders
        if supabase_url and 'your-project-id' in supabase_url:
            logger.warning(f"URL Supabase incorrecte (placeholder détecté): {supabase_url}")
            supabase_url = "https://kyqznnxtdjteefvdkwto.supabase.co"
            logger.info(f"Utilisation de l'URL Supabase de production: {supabase_url}")
        
        if supabase_url and supabase_key:
            try:
                logger.info(f"Tentative de connexion à Supabase: {supabase_url}")
                supabase = create_client(supabase_url, supabase_key)
                
                # Tester que la connexion fonctionne
                test_response = supabase.table("site_pages").select("id").limit(1).execute()
                logger.info(f"Client Supabase initialisé avec succès et testé: {test_response}")
                _clients_initialized = True
                return True
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation du client Supabase: {e}")
                # Ne pas tomber en mode test, lever une exception
                raise e
        else:
            # Essayer via profils/env_vars.json
            supabase_url = get_env_var("SUPABASE_URL")
            supabase_key = get_env_var("SUPABASE_SERVICE_KEY")
            if supabase_url and supabase_key:
                try:
                    logger.info(f"Tentative de connexion à Supabase (profil): {supabase_url}")
                    supabase = create_client(supabase_url, supabase_key)
                    # Test rapide
                    supabase.table("site_pages").select("id").limit(1).execute()
                    logger.info("Client Supabase initialisé via profil")
                    _clients_initialized = True
                    return True
                except Exception as e2:
                    logger.error(f"Échec initialisation Supabase via profil: {e2}")
            logger.error("Variables Supabase manquantes (SUPABASE_URL, SUPABASE_SERVICE_KEY)")
            raise Exception("Variables Supabase obligatoires manquantes")
            
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des clients: {e}")
        raise e

# Ne pas initialiser immédiatement; laisser main()/UI faire l'init lorsque l'env est prêt

# Taille maximale des chunks
MAX_CHUNK_SIZE = 5000

@dataclass
class ProcessedChunk:
    """Dataclass représentant un chunk traité de documentation."""
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    category: str
    embedding: Optional[List[float]] = None

class CrawlProgressTracker:
    """Classe pour suivre la progression du processus de crawling."""
    
    def __init__(self, 
                 progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """Initialiser le tracker de progression.
        
        Args:
            progress_callback: Fonction à appeler avec les mises à jour de progression
        """
        self.progress_callback = progress_callback
        self.urls_found = 0
        self.urls_processed = 0
        self.urls_succeeded = 0
        self.urls_failed = 0
        self.chunks_stored = 0
        self.logs = []
        self.is_running = False
        self.start_time = None
        self.end_time = None
    
    def log(self, message: str):
        """Ajouter un message de log et mettre à jour la progression."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(message)  # Également imprimer dans la console
        
        # Appeler le callback de progression si fourni
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def start(self):
        """Marquer le processus de crawling comme démarré."""
        self.is_running = True
        self.start_time = datetime.now()
        self.log("Processus de crawling démarré")
        
        # Appeler le callback de progression si fourni
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def complete(self):
        """Marquer le processus de crawling comme terminé."""
        self.is_running = False
        self.end_time = datetime.now()
        duration_seconds = (self.end_time - self.start_time).total_seconds()
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        self.log(f"Processus de crawling terminé en {duration_str}")
        self.log(f"URLs traitées: {self.urls_processed}/{self.urls_found}")
        self.log(f"Réussite: {self.urls_succeeded}, Échec: {self.urls_failed}")
        self.log(f"Chunks stockés: {self.chunks_stored}")
        
        # Appeler le callback de progression si fourni
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def get_status(self) -> Dict[str, Any]:
        """Obtenir le statut actuel du processus de crawling."""
        duration = None
        if self.start_time:
            end_time = self.end_time if self.end_time else datetime.now()
            duration = (end_time - self.start_time).total_seconds()
        
        success_rate = 0
        if self.urls_processed > 0:
            success_rate = (self.urls_succeeded / self.urls_processed) * 100
        
        progress = 0
        if self.urls_found > 0:
            progress = (self.urls_processed / self.urls_found) * 100
        
        return {
            "is_running": self.is_running,
            "urls_found": self.urls_found,
            "urls_processed": self.urls_processed,
            "urls_succeeded": self.urls_succeeded,
            "urls_failed": self.urls_failed,
            "chunks_stored": self.chunks_stored,
            "progress": progress,
            "success_rate": success_rate,
            "duration_seconds": duration,
            "logs": self.logs.copy()  # Copy to avoid modification issues
        }
    
    def is_completed(self) -> bool:
        """Retourner True si le processus de crawling est terminé."""
        return not self.is_running and self.start_time is not None
    
    def is_successful(self) -> bool:
        """Retourner True si le processus de crawling s'est terminé avec succès."""
        return self.is_completed() and self.urls_succeeded > 0

def chunk_text(text: str, chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
    """Diviser le texte en chunks, en respectant les blocs de code et les paragraphes.
    
    Args:
        text: Texte à diviser en chunks
        chunk_size: Taille approximative de chaque chunk en caractères
    
    Returns:
        List[str]: Liste des chunks de texte
    """
    # Si le texte est suffisamment court, le renvoyer tel quel
    if len(text) <= chunk_size:
        return [text]
    
    # Diviser le texte en paragraphes (séparés par des lignes vides)
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        # Estimer la longueur en ajoutant la longueur du paragraphe + 2 pour les '\n\n'
        para_length = len(para) + 2
        
        # Si ajouter ce paragraphe dépasserait la limite et qu'on a déjà du contenu
        if current_length + para_length > chunk_size and current_length > 0:
            # Finaliser le chunk courant
            chunks.append('\n\n'.join(current_chunk))
            # Commencer un nouveau chunk avec ce paragraphe
            current_chunk = [para]
            current_length = para_length
        else:
            # Sinon, ajouter ce paragraphe au chunk courant
            current_chunk.append(para)
            current_length += para_length
    
    # Ajouter le dernier chunk s'il n'est pas vide
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def extract_title_from_markdown(content: str, url: str) -> str:
    """Extraire le titre à partir du contenu markdown ou de l'URL.
    
    Args:
        content: Contenu markdown
        url: URL source
        
    Returns:
        str: Titre extrait
    """
    # Chercher un titre H1 (# Titre) dans le contenu
    h1_match = re.search(r'^\s*#\s+(.+?)(\n|$)', content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()
    
    # Chercher un titre H2 (## Titre) si H1 n'est pas trouvé
    h2_match = re.search(r'^\s*##\s+(.+?)(\n|$)', content, re.MULTILINE)
    if h2_match:
        return h2_match.group(1).strip()
    
    # Si aucun titre n'est trouvé, extraire le nom du fichier de l'URL
    parts = url.split('/')
    if len(parts) > 0:
        filename = parts[-1]
        # Enlever l'extension et remplacer les tirets par des espaces
        if '.' in filename:
            basename = filename.split('.')[0]
            return basename.replace('-', ' ').replace('_', ' ').title()
    
    # Si tout échoue, utiliser le domaine comme titre
    domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    if domain_match:
        return f"Documentation {domain_match.group(1).split('.')[0].title()}"
    
    # Fallback: retourner un titre générique
    return "Documentation MCP"

def get_title_and_summary(chunk: str, url: str, chunk_number: int = 0) -> Tuple[str, str]:
    """Extraire le titre et le résumé à partir du chunk."""
    try:
        # Pour le premier chunk (0), extraire le titre du contenu markdown ou de l'URL
        if chunk_number == 0:
            title = extract_title_from_markdown(chunk, url)
        else:
            # Pour les chunks suivants, utiliser le même titre avec indication du numéro
            base_title = extract_title_from_markdown(chunk, url)
            title = f"{base_title} (partie {chunk_number+1})"
        
        # Générer un résumé court du document
        # Utiliser les 200 premiers caractères du contenu comme résumé
        clean_text = re.sub(r'\s+', ' ', chunk.strip())
        summary = clean_text[:200] + "..." if len(clean_text) > 200 else clean_text
        
        return title, summary
    except Exception as e:
        print(f"Error extracting title and summary: {e}")
        # Fallbacks en cas d'erreur
        if chunk_number == 0:
            return "Documentation MCP", "Document MCP"
        else:
            return f"Documentation MCP (partie {chunk_number+1})", "Document MCP"

def _sanitize_zerowidth(text: str) -> str:
    """Supprime les caractères de largeur nulle (ZWSP, ZWNJ, ZWJ, BOM) d'une chaîne.

    Args:
        text: Chaîne potentiellement contaminée par des caractères invisibles
    Returns:
        Chaîne nettoyée
    """
    try:
        return re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    except Exception:
        return text

async def get_title_and_summary_llm(chunk: str, url: str) -> Optional[Tuple[str, str]]:
    """Utilise un LLM (OpenAI-compatible) pour extraire un titre et un résumé concis.

    Préfère un format JSON structuré {"title": str, "summary": str}. En cas d'erreur,
    retourne None pour déclencher le fallback heuristique.
    """
    try:
        if not llm_client:
            logger.debug("LLM non initialisé – tentative d'initialisation paresseuse")
            try:
                init_clients()
            except Exception:
                return None
            if not llm_client:
                return None
        system_prompt = (
            "You extract a short, informative title and a precise, helpful summary for a documentation chunk. "
            "Return JSON with keys 'title' and 'summary'. The summary should capture the main points of the chunk, "
            "be specific (bullet-like sentences if useful), and stay under ~3 sentences."
        )
        # Model selection: allow override via PRIMARY_MODEL, else a lightweight default
        model_name = os.environ.get("PRIMARY_MODEL") or "gpt-4o-mini"
        content_snippet = chunk[:1200]
        resp = await llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{content_snippet}"},
            ],
            response_format={"type": "json_object"},
        )
        payload = resp.choices[0].message.content
        data = json.loads(payload)
        title = _sanitize_zerowidth(str(data.get("title", "")).strip()) or None
        summary = str(data.get("summary", "")).strip() or None
        if not title or not summary:
            return None
        return title, summary
    except Exception as e:
        logger.debug(f"LLM title/summary extraction failed, fallback to heuristic: {e}")
        return None

async def get_embedding(text: str) -> List[float]:
    """Génère un vecteur d'embedding pour le texte donné.
    
    Args:
        text: Texte pour lequel générer un embedding
        
    Returns:
        List[float]: Vecteur d'embedding ou vecteur nul en cas d'erreur
    """
    # Récupérer la dimension des embeddings depuis les variables d'environnement
    embedding_dim = int(os.environ.get('EMBEDDING_DIMENSIONS', 768))
    
    # Vérifier que le client est initialisé (lazy init thread-safe)
    if not embedding_client:
        logger.debug("Client d'embedding non initialisé – tentative d'initialisation paresseuse")
        try:
            init_clients()
        except Exception:
            logger.error("Échec d'initialisation du client d'embedding")
            return [0.0] * embedding_dim
        if not embedding_client:
            logger.error("Client d'embedding indisponible après initialisation")
            return [0.0] * embedding_dim
    
    # Limiter la taille du texte pour éviter les erreurs de token limit
    max_text_length = 8000
    if len(text) > max_text_length:
        logger.warning(f"Texte tronqué de {len(text)} à {max_text_length} caractères")
        text = text[:max_text_length]
    
    # Utiliser la variable globale embedding_model déjà définie
    global embedding_model
    
    try:
        # Utiliser le client configuré (OpenAI ou Ollama) pour générer l'embedding
        logger.info(f"Génération d'embedding avec le modèle {embedding_model} (longueur texte: {len(text)})")
        
        # Ajouter un retry pour gérer les problèmes de connexion intermittents
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = await embedding_client.embeddings.create(
                    model=embedding_model,
                    input=text
                )
                
                # Récupérer le vecteur d'embedding
                embedding_vector = response.data[0].embedding
                logger.info(f"Embedding généré avec succès, dimension: {len(embedding_vector)}")
                return embedding_vector
                
            except Exception as retry_error:
                if attempt < max_retries - 1:
                    logger.warning(f"Tentative {attempt+1}/{max_retries} échouée: {retry_error}. Réessai dans {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Backoff exponentiel
                else:
                    raise retry_error
        
    except Exception as e:
        logger.error(f"Erreur de génération d'embedding après {max_retries} tentatives: {e}")
        # Retourner un vecteur nul compatible avec la dimension attendue
        return [0.0] * embedding_dim

async def process_chunk(chunk: str, chunk_number: int, url: str) -> Optional[ProcessedChunk]:
    """Traiter un seul chunk de texte de façon asynchrone."""
    try:
        # Extraire le titre et le résumé: d'abord via LLM, sinon heuristique
        extracted = await get_title_and_summary_llm(chunk, url)
        if extracted:
            title, summary = extracted
        else:
            title, summary = get_title_and_summary(chunk, url, chunk_number)
        title = _sanitize_zerowidth(title)
        
        # Extraire la catégorie de l'URL
        parts = url.split('/')
        category = parts[-1] if parts else "unknown"
        if '.' in category:
            category = category.split('.')[0]
        
        # Obtenir l'embedding
        embedding = await get_embedding(chunk)
        
        # Créer et retourner l'objet ProcessedChunk
        return ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title=title,
            summary=summary,
            content=chunk,
            category=category,
            embedding=embedding
        )
    except Exception as e:
        print(f"Erreur lors du traitement du chunk: {e}")
        return None

async def insert_chunk(chunk: ProcessedChunk) -> bool:
    """Insérer un chunk traité dans Supabase."""
    try:
        # Créer un identifiant unique pour le chunk basé sur l'URL et le numéro de chunk
        # Préparer les données pour l'insertion (pas d'ID string si la colonne est bigint)
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": {
                "source": "mcp_docs",
                "category": chunk.category,
                "crawled_date": datetime.now().isoformat(),
            }
        }
        
        if chunk.embedding:
            data["embedding"] = chunk.embedding
        
        # Utiliser upsert idempotent sur la contrainte (url, chunk_number)
        supabase.table("site_pages").upsert(data, on_conflict="url,chunk_number").execute()
        
        print(f"Chunk {chunk.chunk_number} inséré pour {chunk.url}")
        return True
    except Exception as e:
        print(f"Erreur lors de l'insertion du chunk: {e}")
        return False

async def process_and_store_document(url: str, markdown: str, tracker: Optional[CrawlProgressTracker] = None) -> int:
    """Traiter un document et stocker ses chunks."""
    try:
        # Diviser le contenu en chunks
        chunks = chunk_text(markdown)
        
        if tracker:
            tracker.log(f"Document divisé en {len(chunks)} chunks: {url}")
        else:
            print(f"Document divisé en {len(chunks)} chunks: {url}")
        
        chunks_stored = 0
        
        # Traiter et stocker chaque chunk
        for i, chunk_content in enumerate(chunks):
            processed_chunk = await process_chunk(chunk_content, i, url)
            
            if processed_chunk and processed_chunk.embedding:
                success = await insert_chunk(processed_chunk)
                
                if success:
                    chunks_stored += 1
                    
                    if tracker:
                        tracker.chunks_stored += 1
        
        return chunks_stored
    except Exception as e:
        if tracker:
            tracker.log(f"Erreur lors du traitement du document {url}: {e}")
        else:
            print(f"Erreur lors du traitement du document {url}: {e}")
        return 0

def convert_github_url_to_raw(url: str) -> str:
    """Convertit une URL GitHub en URL raw pour récupérer le contenu brut.
    
    Args:
        url: URL GitHub au format https://github.com/ORG/REPO/blob/BRANCH/PATH
        
    Returns:
        str: URL convertie en format raw.githubusercontent.com
    """
    raw_url = url
    
    if 'github.com' in url:
        try:
            if '/blob/' in url:
                # Format: https://github.com/ORG/REPO/blob/BRANCH/PATH
                parts = url.split('/')
                if len(parts) >= 7:
                    org = parts[3]          # organisation (ex: modelcontextprotocol)
                    repo = parts[4]          # nom du dépôt (ex: docs, specification)
                    branch = parts[6]        # branche (généralement 'main')
                    path = '/'.join(parts[7:])  # chemin du fichier
                    raw_url = f"https://raw.githubusercontent.com/{org}/{repo}/{branch}/{path}"
                    logger.info(f"URL GitHub transformée: {raw_url}")
            elif '/tree/' in url:
                # Pour les URLs de répertoire, essayer de récupérer le README.md
                parts = url.split('/')
                if len(parts) >= 7:
                    org = parts[3]
                    repo = parts[4]
                    branch = parts[6]
                    path = '/'.join(parts[7:])
                    raw_url = f"https://raw.githubusercontent.com/{org}/{repo}/{branch}/{path}/README.md"
                    logger.info(f"Tentative de récupération du README.md: {raw_url}")
        except Exception as e:
            logger.error(f"Erreur de conversion d'URL GitHub: {e}")
            # En cas d'erreur, retourner l'URL d'origine
    
    return raw_url

def fetch_url_content_sync(url: str) -> Optional[str]:
    """Récupère le contenu d'une URL (HTML/Markdown) et le convertit en Markdown (synchrone)."""
    try:
        raw_url = convert_github_url_to_raw(url)
        response = requests.get(raw_url, timeout=30)
        if response.status_code != 200:
            return None
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            return html_converter.handle(response.text)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la récupération de l'URL {url}: {e}")
        return None
    except Exception as e:
        print(f"Erreur inattendue lors de la récupération de {url}: {e}")
        return None

async def crawl_url(url: str, tracker: Optional[CrawlProgressTracker] = None) -> int:
    """Crawler une seule URL et traiter son contenu.
    
    Args:
        url: L'URL à crawler
        tracker: Tracker de progression optionnel
        
    Returns:
        int: Le nombre de chunks stockés
    """
    try:
        if tracker:
            tracker.log(f"Traitement de {url}")
        
        # Récupérer le contenu
        content = await fetch_url_content(url)
        
        if not content:
            if tracker:
                tracker.urls_processed += 1
                tracker.urls_failed += 1
                tracker.log(f"Échec de récupération du contenu pour {url}")
            return 0
        
        # Traiter et stocker le document
        chunks_stored = await process_and_store_document(url, content, tracker)
        
        if tracker:
            tracker.urls_processed += 1
            if chunks_stored > 0:
                tracker.urls_succeeded += 1
                tracker.log(f"Traitement réussi pour {url}, {chunks_stored} chunks stockés")
            else:
                tracker.urls_failed += 1
                tracker.log(f"Échec du traitement pour {url}, aucun chunk stocké")
        
        return chunks_stored
    except Exception as e:
        if tracker:
            tracker.urls_processed += 1
            tracker.urls_failed += 1
            tracker.log(f"Erreur lors du traitement de {url}: {e}")
        print(f"Erreur lors du traitement de {url}: {e}")
        return 0

async def crawl_urls(urls: List[str], tracker: Optional[CrawlProgressTracker] = None, batch_size: int = 5) -> int:
    """Crawler plusieurs URLs en parallèle par lots.
    
    Args:
        urls: Liste d'URLs à crawler
        tracker: Tracker de progression optionnel
        batch_size: Taille des lots pour le traitement parallèle
        
    Returns:
        int: Le nombre total de chunks stockés
    """
    total_chunks_stored = 0
    
    # Traiter les URLs par lots pour éviter de surcharger
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i+batch_size]
        
        if tracker:
            tracker.log(f"Traitement du lot {i//batch_size + 1}/{(len(urls)+batch_size-1)//batch_size} ({len(batch)} URLs)")
        
        # Exécuter les tâches en parallèle
        tasks = [crawl_url(url, tracker) for url in batch]
        results = await asyncio.gather(*tasks)
        
        # Additionner le nombre de chunks stockés
        total_chunks_stored += sum(results)
        
        # Attendre un peu entre les lots pour éviter de surcharger les API
        if i + batch_size < len(urls):
            await asyncio.sleep(1)
    
    return total_chunks_stored

def process_chunk_sync(chunk_text: str, chunk_number: int, url: str) -> Optional[ProcessedChunk]:
    """Version synchrone de process_chunk pour le traitement en thread."""
    try:
        # Extraire le titre et le résumé du chunk
        title = extract_title_from_chunk(chunk_text) or f"Chunk {chunk_number} from {url}"
        summary = extract_summary_from_chunk(chunk_text) or "Pas de résumé disponible"
        
        # Déterminer la catégorie du contenu
        if 'specification' in url.lower() or 'spec' in url.lower():
            category = 'specification'
        elif 'tutorial' in url.lower() or 'guide' in url.lower():
            category = 'tutorial'
        elif 'reference' in url.lower() or 'api' in url.lower():
            category = 'reference'
        else:
            category = 'documentation'
        
        # Créer l'objet ProcessedChunk
        processed_chunk = ProcessedChunk(
            url=url,
            chunk_number=chunk_number,
            title=title,
            summary=summary,
            content=chunk_text,
            category=category
        )
        
        # Générer l'embedding de manière synchrone
        try:
            # Créer un nouvel événement loop pour exécuter la fonction asynchrone dans un contexte synchrone
            loop = asyncio.new_event_loop()
            processed_chunk.embedding = loop.run_until_complete(get_embedding(chunk_text))
            loop.close()
            logger.info(f"Embedding généré avec succès pour {title} (dimensions: {len(processed_chunk.embedding)})")
        except Exception as e:
            # Récupérer la dimension des embeddings depuis l'environnement ou utiliser une valeur par défaut
            embedding_dim = int(os.environ.get('EMBEDDING_DIMENSIONS', 768))  # Ollama nomic-embed-text utilise 768
            logger.error(f"Erreur lors de la génération de l'embedding: {e}")
            logger.warning(f"Utilisation d'un embedding nul de dimension {embedding_dim}")
            processed_chunk.embedding = [0.0] * embedding_dim
        
        return processed_chunk
    except Exception as e:
        logger.error(f"Erreur lors du traitement du chunk {chunk_number} pour {url}: {e}")
        return None

def insert_chunk_sync(chunk: ProcessedChunk) -> bool:
    """Version synchrone de insert_chunk pour le traitement en thread.
    
    Args:
        chunk (ProcessedChunk): Chunk traité avec embedding
        
    Returns:
        bool: True si l'insertion est réussie, False sinon.
    """
    global supabase
    
    # Supabase doit être configuré
    if supabase is None:
        logger.error("ERREUR CRITIQUE: Client Supabase non initialisé. Vérifiez les variables d'environnement.")
        logger.error("SUPABASE_URL: " + str(os.environ.get("SUPABASE_URL")))
        # Ne pas afficher la clé complète pour des raisons de sécurité
        if os.environ.get("SUPABASE_SERVICE_KEY"):
            key_prefix = os.environ.get("SUPABASE_SERVICE_KEY")[:10] + "..." 
            logger.error(f"SUPABASE_SERVICE_KEY: {key_prefix} (présente mais connexion échouée)")
        else:
            logger.error("SUPABASE_SERVICE_KEY: non définie")
            
        # Réinitialiser le client Supabase pour un nouvel essai
        try:
            supabase_url = os.environ.get("SUPABASE_URL") or get_env_var("SUPABASE_URL")
            supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") or get_env_var("SUPABASE_SERVICE_KEY")
            if supabase_url and supabase_key:
                logger.info(f"Tentative de réinitialisation du client Supabase avec: {supabase_url}")
                supabase = create_client(supabase_url, supabase_key)
                logger.info("Client Supabase réinitialisé avec succès")
        except Exception as e:
            logger.error("Impossible d'initialiser le client Supabase: aucune configuration valide trouvée")
            raise RuntimeError("Supabase non configuré")
    # S'assurer que l'embedding est une liste de floats (pour pgvector/float8[])
    embedding_list: Optional[List[float]] = None
    if isinstance(chunk.embedding, list):
        try:
            embedding_list = [float(x) for x in chunk.embedding]
        except Exception:
            embedding_list = None
    
    # Préparer les données à insérer (metadata en JSON objet, pas string)
    data = {
        "url": chunk.url,
        "title": chunk.title,
        "content": chunk.content,
        "summary": chunk.summary,
        "chunk_number": chunk.chunk_number,
        "embedding": embedding_list,
        "metadata": {
            "source": "mcp_docs",
            "category": chunk.category,
            "indexed_at": datetime.now().isoformat(),
        },
    }
    
    # Insérer les données dans Supabase (synchrone)
    try:
        # Vérifier une dernière fois que le client est initialisé
        if supabase is None:
            logger.error("ERREUR CRITIQUE: Client Supabase toujours non initialisé après tentative de réinitialisation")
            return False
        
        logger.info(f"Insertion/Upsert du chunk '{chunk.title}' dans Supabase...")
        try:
            # Upsert idempotent basé sur la contrainte unique (url, chunk_number)
            response = supabase.table("site_pages").upsert(data, on_conflict="url,chunk_number").execute()
        except Exception as up_e:
            # Si la contrainte unique n'existe pas, fallback en insert avec avertissement
            msg = str(up_e)
            if "ON CONFLICT" in msg or "unique" in msg.lower():
                logger.warning("Aucune contrainte unique (url, chunk_number) détectée: fallback en insert."
                               " Recommander d'ajouter un index unique sur (url, chunk_number) pour idempotence.")
                response = supabase.table("site_pages").insert(data).execute()
            else:
                raise
        
        if hasattr(response, 'data') and response.data:
            logger.info(f"Chunk '{chunk.title}' inséré avec succès dans Supabase")
            return True
        else:
            logger.warning(f"Inséré dans Supabase, mais sans confirmation: {response}")
            return True  # Optimiste, supposer que ça a fonctionné
    except Exception as e:
        logger.error(f"Erreur lors de l'insertion dans Supabase: {e}")
        # Vérifier si l'erreur est liée à une connexion ou à un problème d'authentification
        if "auth" in str(e).lower() or "cred" in str(e).lower() or "key" in str(e).lower():
            logger.critical("Problème d'authentification Supabase - vérifiez les clés API")
        elif "connect" in str(e).lower():
            logger.critical("Problème de connexion Supabase - vérifiez la connectivité réseau")
        return False
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des données pour l'insertion: {e}")
        return False

def process_and_store_document_sync(url: str, markdown: str, tracker: Optional[CrawlProgressTracker] = None) -> int:
    """Version synchrone de process_and_store_document pour le traitement en thread."""
    try:
        # Diviser le contenu en chunks
        chunks = chunk_text(markdown)
        
        if tracker:
            tracker.log(f"Document divisé en {len(chunks)} chunks: {url}")
        else:
            print(f"Document divisé en {len(chunks)} chunks: {url}")
        
        chunks_stored = 0

        # Traitement normal: générer, insérer, compter (pas de mode test implicite)
        for i, chunk_content in enumerate(chunks):
            processed_chunk = process_chunk_sync(chunk_content, i, url)
            
            if processed_chunk and processed_chunk.embedding:
                success = insert_chunk_sync(processed_chunk)
                
                if success:
                    chunks_stored += 1
                    
                    if tracker:
                        tracker.chunks_stored += 1
        
        return chunks_stored
    except Exception as e:
        if tracker:
            tracker.log(f"Erreur lors du traitement du document {url}: {e}")
        else:
            print(f"Erreur lors du traitement du document {url}: {e}")
        return 0

def crawl_parallel_with_requests(urls: List[str], tracker: Optional[CrawlProgressTracker] = None, max_concurrent: int = 5):
    """Crawler plusieurs URLs en parallèle avec une limite de concurrence en utilisant des requêtes HTTP directes."""
    
    # Sémaphore pour limiter la concurrence
    semaphore = threading.Semaphore(max_concurrent)
    
    def process_url(url: str):
        """Traiter une seule URL."""
        try:
            # Acquérir le sémaphore (limiter la concurrence)
            semaphore.acquire()
            
            if tracker:
                tracker.log(f"Traitement de {url}")
            else:
                print(f"Traitement de {url}")
            
            try:
                # Récupérer le contenu
                content = fetch_url_content_sync(url)
                
                if content:
                    # Traiter et stocker le document avec la version synchrone
                    chunks_stored = process_and_store_document_sync(url, content, tracker)
                    
                    if tracker:
                        tracker.urls_processed += 1
                        tracker.urls_succeeded += 1
                        tracker.log(f"Traitement réussi pour {url}, {chunks_stored} chunks stockés")
                    else:
                        print(f"Traitement réussi pour {url}, {chunks_stored} chunks stockés")
                else:
                    if tracker:
                        tracker.urls_processed += 1
                        tracker.urls_failed += 1
                        tracker.log(f"Échec de récupération du contenu pour {url}")
                    else:
                        print(f"Échec de récupération du contenu pour {url}")
            except Exception as e:
                if tracker:
                    tracker.urls_processed += 1
                    tracker.urls_failed += 1
                    tracker.log(f"Erreur lors du traitement de {url}: {e}")
                else:
                    print(f"Erreur lors du traitement de {url}: {e}")
            finally:
                # Libérer le sémaphore
                semaphore.release()
                
        except Exception as e:
            if tracker:
                tracker.log(f"Erreur inattendue lors du traitement de {url}: {e}")
            else:
                print(f"Erreur inattendue lors du traitement de {url}: {e}")
    
    # Créer et démarrer un thread pour chaque URL
    threads = []
    for url in urls:
        thread = threading.Thread(target=process_url, args=(url,))
        thread.start()
        threads.append(thread)
    
    # Attendre que tous les threads se terminent
    for thread in threads:
        thread.join()

def main_with_requests(tracker: Optional[CrawlProgressTracker] = None):
    """Fonction principale utilisant des requêtes HTTP directes au lieu de l'automatisation du navigateur."""
    try:
        # Démarrer le suivi si le tracker est fourni
        if tracker:
            tracker.start()
        else:
            print("Démarrage du processus de crawling...")
        
        # Effacer d'abord les enregistrements existants
        if tracker:
            tracker.log("Effacement des enregistrements de docs MCP existants...")
        else:
            print("Effacement des enregistrements de docs MCP existants...")
        clear_existing_records_sync()
        if tracker:
            tracker.log("Enregistrements existants effacés")
        else:
            print("Enregistrements existants effacés")
        
        # Obtenir les URLs de la documentation MCP
        if tracker:
            tracker.log("Récupération des URLs de la documentation MCP...")
        else:
            print("Récupération des URLs de la documentation MCP...")
        urls = get_mcp_urls()  # Utilise notre fonction à jour

        # Filtrer les URLs non autorisées par la allowlist
        allowed_urls = [u for u in urls if is_allowed_url(u)]
        if tracker:
            removed = len(urls) - len(allowed_urls)
            if removed:
                tracker.log(f"Filtrage allowlist: {removed} URL(s) exclue(s)")
        urls = allowed_urls
        
        if not urls:
            if tracker:
                tracker.log("Aucune URL trouvée à crawler")
                tracker.complete()
            else:
                print("Aucune URL trouvée à crawler")
            return
        
        if tracker:
            tracker.urls_found = len(urls)
            tracker.log(f"Trouvé {len(urls)} URLs à crawler")
        else:
            print(f"Trouvé {len(urls)} URLs à crawler")
        
        # Crawler les URLs en utilisant des requêtes HTTP directes
        crawl_parallel_with_requests(urls, tracker)
        
        # Marquer comme terminé si le tracker est fourni
        if tracker:
            tracker.complete()
        else:
            print("Processus de crawling terminé")
            
    except Exception as e:
        if tracker:
            tracker.log(f"Erreur dans le processus de crawling: {str(e)}")
            tracker.complete()
        else:
            print(f"Erreur dans le processus de crawling: {str(e)}")

def start_crawl_with_requests(progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> CrawlProgressTracker:
    """Démarrer le processus de crawling en utilisant des requêtes HTTP directes dans un thread séparé et renvoyer le tracker."""
    tracker = CrawlProgressTracker(progress_callback)
    
    def run_crawl():
        try:
            main_with_requests(tracker)  # Utilise la version synchrone avec requests
        except Exception as e:
            print(f"Erreur dans le thread de crawl: {e}")
            tracker.log(f"Erreur de thread: {str(e)}")
            tracker.complete()
    
    # Démarrer le processus de crawling dans un thread séparé
    thread = threading.Thread(target=run_crawl)
    thread.daemon = True
    thread.start()
    
    return tracker

def get_mcp_urls() -> List[str]:
    """Construire une liste élargie d'URLs MCP autorisées (plusieurs centaines possibles).

    Combine des seeds canoniques + expansion automatique via sitemaps (sites officiels)
    et exploration des arbres GitHub des dépôts MCP officiels.
    """
    seeds: List[str] = [
        # Officiel
        "https://modelcontextprotocol.io/introduction",
        "https://modelcontextprotocol.io/overview",
        "https://modelcontextprotocol.io/faqs",
        "https://modelcontextprotocol.io/quickstart/client",
        "https://modelcontextprotocol.io/quickstart/server",
        "https://modelcontextprotocol.io/docs/learn/architecture",
        "https://modelcontextprotocol.io/examples",
        # Spécification
        "https://spec.modelcontextprotocol.io/specification/",
        "https://spec.modelcontextprotocol.io/specification/basic/",
        # GitHub org pages clés
        "https://github.com/modelcontextprotocol/servers",
        "https://github.com/modelcontextprotocol/servers/blob/main/README.md",
        "https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/README.md",
        "https://github.com/modelcontextprotocol/.github/blob/main/CONTRIBUTING.md",
        # IDE clients
        "https://code.visualstudio.com/mcp",
        "https://code.visualstudio.com/docs/copilot/chat/mcp-servers",
        "https://docs.windsurf.com/windsurf/mcp",
        "https://docs.windsurf.com/windsurf/cascade/mcp",
        # Vendors & libs
        "https://supabase.com/docs/guides/getting-started/mcp",
        "https://supabase.com/features/mcp-server",
        "https://openai.github.io/openai-agents-python/mcp/",
        "https://platform.openai.com/docs/mcp",
        "https://docs.anthropic.com/en/docs/claude-code/mcp",
        "https://docs.anthropic.com/en/docs/build-with-claude/mcp",
        "https://docs.llamaindex.ai/en/stable/examples/tools/mcp/",
    ]

    expanded = set(seeds)

    # Expansion via sitemaps pour les domaines officiels
    try:
        expanded.update(fetch_sitemap_urls("https://modelcontextprotocol.io/sitemap.xml"))
    except Exception as e:
        logger.warning(f"Sitemap MCP site erreur: {e}")
    try:
        expanded.update(fetch_sitemap_urls("https://spec.modelcontextprotocol.io/sitemap.xml"))
    except Exception as e:
        logger.warning(f"Sitemap MCP spec erreur: {e}")

    # Expansion GitHub: parcourir l'arborescence des dépôts importants
    gh_repos = [
        ("modelcontextprotocol", "servers"),
        ("modelcontextprotocol", "modelcontextprotocol"),
    ]
    for org, repo in gh_repos:
        try:
            expanded.update(fetch_github_repo_docs(org, repo))
        except Exception as e:
            logger.warning(f"GitHub tree erreur pour {org}/{repo}: {e}")

    # Filtrer par allowlist et normaliser
    urls: List[str] = []
    for u in expanded:
        try:
            host = urlparse(u).hostname or ""
            if any(host.endswith(d) for d in ALLOWED_DOMAINS):
                urls.append(u)
        except Exception:
            continue

    urls = sorted(set(urls))
    logger.info(f"Liste MCP construite: {len(urls)} URLs")
    return urls

def fetch_sitemap_urls(sitemap_url: str) -> List[str]:
    """Récupérer toutes les URLs d'un sitemap XML.

    Args:
        sitemap_url: URL du sitemap.xml
    Returns:
        Liste d'URLs trouvées
    """
    try:
        resp = requests.get(sitemap_url, timeout=20)
        if resp.status_code != 200:
            return []
        xml = resp.text
        # Extraire <loc>...</loc>
        locs = re.findall(r"<loc>(.*?)</loc>", xml)
        return [loc.strip() for loc in locs if loc.strip()]
    except Exception:
        return []

async def fetch_url_content(url: str) -> Optional[str]:
    """Wrapper asynchrone autour de fetch_url_content_sync."""
    return await asyncio.to_thread(fetch_url_content_sync, url)
def fetch_github_repo_docs(org: str, repo: str) -> List[str]:
    """Lister les fichiers docs/README*.md d'un dépôt GitHub (API publique non authentifiée).

    On utilise l'API git trees recursive pour récupérer les chemins, puis
    on convertit en URLs GitHub blob pour traitement.
    """
    base = f"https://api.github.com/repos/{org}/{repo}/git/trees/HEAD?recursive=1"
    try:
        r = requests.get(base, timeout=30, headers={"Accept": "application/vnd.github+json"})
        if r.status_code != 200:
            return []
        data = r.json()
        urls: List[str] = []
        for item in data.get("tree", []):
            path = item.get("path", "")
            if not isinstance(path, str):
                continue
            if path.lower().endswith(('.md', '.mdx')) and (
                'readme' in path.lower() or '/docs/' in path.lower() or path.lower().startswith('docs/')
            ):
                urls.append(f"https://github.com/{org}/{repo}/blob/main/{path}")
        return urls
    except Exception:
        return []

def clear_existing_records() -> bool:
    """Effacer les enregistrements MCP existants de la table `site_pages` (synchrone)."""
    try:
        supabase.table("site_pages").delete().eq("metadata->>source", "mcp_docs").execute()
        print("Enregistrements MCP existants effacés")
        return True
    except Exception as e:
        print(f"Erreur lors de l'effacement des enregistrements existants: {e}")
        return False
        
def clear_existing_records_sync() -> bool:
    """Version synchrone pour effacer les enregistrements MCP existants."""
    try:
        return clear_existing_records()
    except Exception as e:
        print(f"Erreur lors de l'effacement synchrone des enregistrements: {e}")
        return False

async def main(tracker: Optional[CrawlProgressTracker] = None) -> bool:
    """Fonction principale du crawler MCP."""
    try:
        # Initialiser les clients (Ollama + Supabase) quand l'environnement est chargé
        init_clients()
        # Démarrer le tracker
        if tracker:
            tracker.start()
        
        # Effacer les enregistrements existants
        if tracker:
            tracker.log("Effacement des enregistrements MCP existants...")
        clear_existing_records()
        
        # Obtenir les URLs
        if tracker:
            tracker.log("Récupération des URLs de la documentation MCP...")
        urls = get_mcp_urls()
        
        if not urls:
            if tracker:
                tracker.log("Aucune URL trouvée à crawler")
                tracker.complete()
            return False
        
        if tracker:
            tracker.urls_found = len(urls)
            tracker.log(f"Trouvé {len(urls)} URLs à crawler")
        
        # Crawler les URLs
        await crawl_urls(urls, tracker)
        
        # Marquer comme terminé
        if tracker:
            tracker.complete()
        
        return True
    except Exception as e:
        if tracker:
            tracker.log(f"Erreur dans le processus de crawling: {str(e)}")
            tracker.complete()
        print(f"Erreur dans le processus de crawling: {str(e)}")
        return False

def start_crawl_async(progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> CrawlProgressTracker:
    """Démarrer le processus de crawling dans une tâche asynchrone et renvoyer le tracker."""
    tracker = CrawlProgressTracker(progress_callback)
    
    # Créer une nouvelle boucle d'événements dans un thread séparé
    def run_async_crawl():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main(tracker))
        except Exception as e:
            print(f"Erreur dans la boucle asyncio: {e}")
            if tracker:
                tracker.log(f"Erreur fatale: {str(e)}")
                tracker.complete()
        finally:
            loop.close()
    
    # Démarrer le thread
    import threading
    thread = threading.Thread(target=run_async_crawl)
    thread.daemon = True
    thread.start()
    
    return tracker

if __name__ == "__main__":
    # Exécuter le crawler directement
    print("Démarrage du crawler MCP...")
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    print("Crawler MCP terminé.")

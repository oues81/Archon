"""
Crawler for Windsurf Cascade Workflows and Context Engineering documentation.

This module mirrors the design of the MCP docs crawler but targets Windsurf
workflows, rules, and context engineering practices. It:
- Collects a curated and expanded set of URLs (docs.windsurf.com + relevant GitHub READMEs)
- Fetches content via HTTP requests
- Splits into chunks
- Generates title and summary via an LLM (JSON output), with heuristic fallback
- Generates embeddings
- Upserts into Supabase table `site_pages` with source metadata "windsurf_workflows"

Environment variables required:
- SUPABASE_URL
- SUPABASE_SERVICE_KEY
- EMBEDDING_BASE_URL (OpenAI-compatible, likely http://host:port/api)
- PRIMARY_MODEL (optional; default: gpt-4o-mini)

Notes:
- This is a request/threads-based crawler (no browser automation).
- Idempotent upserts are performed on (url, chunk_number) if the unique index exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import re
import json
import time
import threading
import logging
import asyncio
from datetime import datetime

import requests

try:
    from supabase import create_client
except Exception:  # pragma: no cover - optional import at runtime
    create_client = None  # type: ignore

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - optional import at runtime
    AsyncOpenAI = None  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------------------- Data structures -----------------------------

@dataclass
class ProcessedChunk:
    url: str
    title: str
    content: str
    summary: str
    chunk_number: int
    embedding: Optional[List[float]]
    category: str = "workflow"

# ------------------------------- Clients (lazy) -----------------------------

_embedding_client_lock = threading.Lock()
_llm_client_lock = threading.Lock()
_embedding_client: Optional[Any] = None
_llm_client: Optional[Any] = None
_clients_initialized = False


def _sanitize_zerowidth(text: str) -> str:
    # Remove most zero-width and bidi control chars
    return re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]", "", text)


def ensure_clients() -> None:
    global _embedding_client, _llm_client, _clients_initialized
    if _clients_initialized:
        return
    with _embedding_client_lock:
        if _embedding_client is None and AsyncOpenAI is not None:
            base_url = os.environ.get("EMBEDDING_BASE_URL")
            api_key = os.environ.get("EMBEDDING_API_KEY", "sk-no-key")
            if base_url:
                try:
                    _embedding_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
                    logger.debug("Embedding client initialized")
                except Exception as e:
                    logger.warning(f"Failed to init embedding client: {e}")
        else:
            logger.debug("Embedding client already set or AsyncOpenAI unavailable")
    with _llm_client_lock:
        if _llm_client is None and AsyncOpenAI is not None:
            base_url = os.environ.get("EMBEDDING_BASE_URL")
            api_key = os.environ.get("EMBEDDING_API_KEY", "sk-no-key")
            if base_url:
                try:
                    _llm_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
                    logger.debug("LLM client initialized")
                except Exception as e:
                    logger.warning(f"Failed to init LLM client: {e}")
        else:
            logger.debug("LLM client already set or AsyncOpenAI unavailable")
    _clients_initialized = True


async def get_embedding(text: str) -> List[float]:
    ensure_clients()
    if _embedding_client is None:
        raise RuntimeError("Embedding client is not initialized; set EMBEDDING_BASE_URL")
    model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    resp = await _embedding_client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding  # type: ignore[attr-defined]


async def get_title_and_summary_llm(chunk: str, url: str) -> Optional[Tuple[str, str]]:
    ensure_clients()
    if _llm_client is None:
        return None
    try:
        system_prompt = (
            "You extract a short, informative title and a precise, helpful summary for a documentation chunk. "
            "Return JSON with keys 'title' and 'summary'. The summary should capture the main points, "
            "be specific (bullet-like sentences if useful), and stay under ~3 sentences."
        )
        model_name = os.environ.get("PRIMARY_MODEL") or "gpt-4o-mini"
        content_snippet = chunk[:1200]
        resp = await _llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{content_snippet}"},
            ],
            response_format={"type": "json_object"},
        )
        payload = resp.choices[0].message.content  # type: ignore[attr-defined]
        data = json.loads(payload)
        title = _sanitize_zerowidth(str(data.get("title", "")).strip()) or None
        summary = str(data.get("summary", "")).strip() or None
        if not title or not summary:
            return None
        return title, summary
    except Exception as e:  # pragma: no cover - network
        logger.debug(f"LLM title/summary extraction failed, fallback to heuristic: {e}")
        return None


def extract_title_from_markdown(content: str, url: str) -> str:
    # Find first markdown H1/H2 or first non-empty line
    m = re.search(r"^\s*#\s+(.+)$", content, flags=re.MULTILINE)
    if not m:
        m = re.search(r"^\s*##\s+(.+)$", content, flags=re.MULTILINE)
    if m:
        return _sanitize_zerowidth(m.group(1).strip())
    for line in content.splitlines():
        line = line.strip()
        if line:
            return _sanitize_zerowidth(line[:80])
    return f"Windsurf Docs: {url.split('/')[-1] or 'Page'}"


# ------------------------------ Supabase client -----------------------------

_supabase = None

def _get_supabase():
    global _supabase
    if _supabase is not None:
        return _supabase
    if create_client is None:
        raise RuntimeError("supabase-py not installed")
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("Supabase env vars missing: SUPABASE_URL, SUPABASE_SERVICE_KEY")
    _supabase = create_client(url, key)
    return _supabase


# --------------------------------- Helpers ---------------------------------

def chunk_text(text: str, max_tokens: int = 1400) -> List[str]:
    # Simple chunking by characters with overlap
    max_chars = max_tokens * 4
    overlap = int(max_chars * 0.1)
    chunks: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = j - overlap
    return chunks


def fetch_url_content_sync(url: str, timeout: int = 20) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "ArchonCrawler/1.0"})
        if r.status_code == 200:
            # Prefer markdown/plaintext; if HTML, strip minimal artifacts
            txt = r.text
            return txt
        logger.warning(f"HTTP {r.status_code} for {url}")
        return None
    except Exception as e:  # pragma: no cover - network
        logger.warning(f"Request failed for {url}: {e}")
        return None


def is_allowed_url(url: str) -> bool:
    allow = [
        "https://docs.windsurf.com/windsurf/",
        "https://docs.windsurf.com/windsurf/cascade",
        "https://docs.windsurf.com/windsurf/context",
        "https://docs.windsurf.com/windsurf/workflows",
        "https://docs.windsurf.com/windsurf/context-engineering",
        "https://docs.windsurf.com/windsurf/cascade/methods",
        # Selected GitHub content for context engineering intro/examples
        "https://github.com/coleam00/context-engineering-intro/",
        "https://raw.githubusercontent.com/coleam00/context-engineering-intro/",
    ]
    return any(url.startswith(p) for p in allow)


def get_windsurf_urls() -> List[str]:
    seeds: List[str] = [
        # Core docs (Cascade + workflows + context engineering)
        "https://docs.windsurf.com/windsurf/cascade/",
        "https://docs.windsurf.com/windsurf/cascade/workflows",
        "https://docs.windsurf.com/windsurf/context-engineering",
        "https://docs.windsurf.com/windsurf/context-engineering/principles",
        "https://docs.windsurf.com/windsurf/workflows",
        # How-tos and MCP pages that overlap with workflows methodology
        "https://docs.windsurf.com/windsurf/cascade/mcp",
        # Example public repo with context engineering examples
        "https://github.com/coleam00/context-engineering-intro",
        "https://github.com/coleam00/context-engineering-intro/blob/main/README.md",
    ]

    # Deduplicate, enforce allowlist
    seeds = [u for u in dict.fromkeys(seeds)]
    urls = [u for u in seeds if is_allowed_url(u)]
    return urls


# ---------------------------- Processing pipeline ---------------------------

def process_chunk_sync(chunk_text_val: str, chunk_number: int, url: str) -> Optional[ProcessedChunk]:
    try:
        title: Optional[str] = None
        summary: Optional[str] = None

        # Try LLM first
        try:
            loop = asyncio.new_event_loop()
            coro = get_title_and_summary_llm(chunk_text_val, url)
            llm_res = loop.run_until_complete(coro)
            loop.close()
            if llm_res:
                title, summary = llm_res
        except Exception as e:
            logger.debug(f"LLM extraction error, will fallback: {e}")

        # Heuristic fallback
        if not title:
            base_title = extract_title_from_markdown(chunk_text_val, url)
            title = base_title if chunk_number == 0 else f"{base_title} (part {chunk_number+1})"
        if not summary:
            clean_text = re.sub(r"\s+", " ", chunk_text_val.strip())
            summary = clean_text[:200] + ("..." if len(clean_text) > 200 else "")

        processed = ProcessedChunk(
            url=url,
            title=title,
            content=chunk_text_val,
            summary=summary,
            chunk_number=chunk_number,
            embedding=None,
            category="windsurf_workflows",
        )

        # Embedding generation
        try:
            loop = asyncio.new_event_loop()
            processed.embedding = loop.run_until_complete(get_embedding(chunk_text_val))
            loop.close()
            logger.info(
                f"Embedding generated for '{title}' (dim={len(processed.embedding) if processed.embedding else 0})"
            )
        except Exception as e:
            dim = int(os.environ.get("EMBEDDING_DIMENSIONS", 768))
            logger.error(f"Embedding generation failed: {e}; using zero vector of dim {dim}")
            processed.embedding = [0.0] * dim

        return processed
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_number} for {url}: {e}")
        return None


def insert_chunk_sync(chunk: ProcessedChunk) -> bool:
    supabase = _get_supabase()

    # Ensure embedding list of floats
    embedding_list: Optional[List[float]] = None
    if isinstance(chunk.embedding, list):
        try:
            embedding_list = [float(x) for x in chunk.embedding]
        except Exception:
            embedding_list = None

    data = {
        "url": chunk.url,
        "title": chunk.title,
        "content": chunk.content,
        "summary": chunk.summary,
        "chunk_number": chunk.chunk_number,
        "embedding": embedding_list,
        "metadata": {
            "source": "windsurf_workflows",
            "category": chunk.category,
            "indexed_at": datetime.now().isoformat(),
        },
    }

    try:
        try:
            response = supabase.table("site_pages").upsert(data, on_conflict="url,chunk_number").execute()
        except Exception as up_e:
            msg = str(up_e)
            if "ON CONFLICT" in msg or "unique" in msg.lower():
                logger.warning(
                    "Unique constraint (url,chunk_number) not present: falling back to insert. "
                    "Recommend adding unique index for idempotence."
                )
                response = supabase.table("site_pages").insert(data).execute()
            else:
                raise

        if hasattr(response, "data") and response.data:
            logger.info(f"Inserted '{chunk.title}' for {chunk.url} (chunk {chunk.chunk_number})")
            return True
        logger.warning(f"Insert/upsert returned without data confirmation: {response}")
        return True
    except Exception as e:
        logger.error(f"Supabase insert error: {e}")
        return False


# ----------------------------- Crawl orchestration --------------------------

class CrawlProgressTracker:
    """Minimal tracker compatible with existing UI expectations."""
    def __init__(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.callback = callback
        self.urls_found = 0
        self.urls_processed = 0
        self.urls_succeeded = 0
        self.urls_failed = 0
        self.chunks_stored = 0
        self._active = False

    def _emit(self, event: str, message: str):
        if self.callback:
            try:
                self.callback({
                    "event": event,
                    "message": message,
                    "stats": self.as_dict(),
                    "ts": time.time(),
                })
            except Exception:
                pass

    def as_dict(self) -> Dict[str, Any]:
        return {
            "urls_found": self.urls_found,
            "urls_processed": self.urls_processed,
            "urls_succeeded": self.urls_succeeded,
            "urls_failed": self.urls_failed,
            "chunks_stored": self.chunks_stored,
            "active": self._active,
        }

    def start(self):
        self._active = True
        self._emit("start", "Windsurf workflows crawl started")

    def log(self, message: str):
        logger.info(message)
        self._emit("log", message)

    def complete(self):
        self._active = False
        self._emit("complete", "Crawl complete")


def process_and_store_document_sync(url: str, content: str, tracker: Optional[CrawlProgressTracker] = None) -> int:
    try:
        chunks = chunk_text(content)
        if tracker:
            tracker.log(f"Document split into {len(chunks)} chunks: {url}")
        count = 0
        for i, chunk in enumerate(chunks):
            processed = process_chunk_sync(chunk, i, url)
            if processed and processed.embedding:
                if insert_chunk_sync(processed):
                    count += 1
                    if tracker:
                        tracker.chunks_stored += 1
        return count
    except Exception as e:
        if tracker:
            tracker.log(f"Error processing {url}: {e}")
        else:
            logger.error(f"Error processing {url}: {e}")
        return 0


def crawl_parallel_with_requests(urls: List[str], tracker: Optional[CrawlProgressTracker] = None, max_concurrent: int = 5):
    semaphore = threading.Semaphore(max_concurrent)

    def worker(u: str):
        try:
            semaphore.acquire()
            if tracker:
                tracker.log(f"Fetching {u}")
            content = fetch_url_content_sync(u)
            if content:
                stored = process_and_store_document_sync(u, content, tracker)
                if tracker:
                    tracker.urls_processed += 1
                    tracker.urls_succeeded += 1
                    tracker.log(f"OK {u}: {stored} chunks stored")
            else:
                if tracker:
                    tracker.urls_processed += 1
                    tracker.urls_failed += 1
                    tracker.log(f"Failed to fetch {u}")
        except Exception as e:
            if tracker:
                tracker.urls_processed += 1
                tracker.urls_failed += 1
                tracker.log(f"Error {u}: {e}")
        finally:
            semaphore.release()

    threads: List[threading.Thread] = []
    for u in urls:
        t = threading.Thread(target=worker, args=(u,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def main_with_requests(tracker: Optional[CrawlProgressTracker] = None):
    try:
        if tracker:
            tracker.start()
            tracker.log("Collecting Windsurf workflows URLs...")
        urls = get_windsurf_urls()
        urls = [u for u in urls if is_allowed_url(u)]
        if not urls:
            if tracker:
                tracker.complete()
            return
        if tracker:
            tracker.urls_found = len(urls)
            tracker.log(f"Found {len(urls)} URLs to crawl")
        crawl_parallel_with_requests(urls, tracker)
        if tracker:
            tracker.complete()
    except Exception as e:
        if tracker:
            tracker.log(f"Fatal error in crawl: {e}")
            tracker.complete()
        else:
            logger.error(f"Fatal error in crawl: {e}")


def start_crawl_with_requests(progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> CrawlProgressTracker:
    tracker = CrawlProgressTracker(progress_callback)
    def runner():
        try:
            main_with_requests(tracker)
        except Exception as e:
            logger.error(f"Crawl thread error: {e}")
            tracker.log(f"Thread error: {e}")
            tracker.complete()
    th = threading.Thread(target=runner, daemon=True)
    th.start()
    return tracker

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
from typing import List, Dict, Any, Optional, Callable, Tuple
import os
import re
import json
import time
import threading
import logging
import html2text
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

# ----------------------- HTML→Markdown converter (global) -------------------
html_converter = html2text.HTML2Text()
html_converter.ignore_links = True
html_converter.ignore_images = True
html_converter.ignore_tables = False
html_converter.body_width = 0


def _sanitize_zerowidth(text: str) -> str:
    # Remove most zero-width and bidi control chars
    return re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]", "", text)

# ------------------------------ HTML helpers -------------------------------
def _looks_like_html(text: str) -> bool:
    try:
        t = text.lstrip().lower()
        return t.startswith("<!doctype html") or t.startswith("<html") or ("<head" in t and "<body" in t)
    except Exception:
        return False

def _strip_html(text: str) -> str:
    try:
        # Remove script/style blocks
        no_scripts = re.sub(r"<script[\s\S]*?>[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
        no_styles = re.sub(r"<style[\s\S]*?>[\s\S]*?</style>", " ", no_scripts, flags=re.IGNORECASE)
        # Remove tags
        no_tags = re.sub(r"<[^>]+>", " ", no_styles)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", no_tags).strip()
        return cleaned
    except Exception:
        return text

def _extract_title_from_html(html: str) -> Optional[str]:
    try:
        # Try H1 first (often the content title on docs)
        m = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return _sanitize_zerowidth(_strip_html(m.group(1)))[:100]
        # Try Open Graph / Twitter meta titles common on Mintlify (attribute order independent)
        meta_tags = re.findall(r"<meta[^>]*>", html, flags=re.IGNORECASE)
        for tag in meta_tags:
            low = tag.lower()
            if ("property=\"og:title\"" in low) or ("name=\"og:title\"" in low) or ("name=\"twitter:title\"" in low) or ("name=\"title\"" in low):
                m2 = re.search(r"content=\"(.*?)\"", tag, flags=re.IGNORECASE)
                if m2:
                    return _sanitize_zerowidth(_strip_html(m2.group(1)))[:100]
        # Fallback to <title>
        m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return _sanitize_zerowidth(_strip_html(m.group(1)))[:100]
        return None
    except Exception:
        return None

def _extract_meta_description(html: str) -> Optional[str]:
    try:
        m = re.search(r"<meta[^>]+property=[\"']og:description[\"'][^>]+content=[\"'](.*?)[\"'][^>]*>", html, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"<meta[^>]+name=[\"']description[\"'][^>]+content=[\"'](.*?)[\"'][^>]*>", html, flags=re.IGNORECASE)
        if m:
            return _sanitize_zerowidth(_strip_html(m.group(1)))
        return None
    except Exception:
        return None

def _extract_main_html(html: str) -> Optional[str]:
    try:
        # Prefer explicit main/article containers
        m = re.search(r"<(main|article)[^>]*>([\s\S]*?)</(main|article)>", html, flags=re.IGNORECASE)
        if m:
            return m.group(0)
        # Try role=main container
        m = re.search(r"<(div|section)[^>]*role=\"main\"[^>]*>([\s\S]*?)</(div|section)>", html, flags=re.IGNORECASE)
        if m:
            return m.group(0)
        # Fallback: extract body and strip wrappers/scripts/styles
        m = re.search(r"<body[^>]*>([\s\S]*?)</body>", html, flags=re.IGNORECASE)
        body = m.group(1) if m else html
        body = re.sub(r"<head[\s\S]*?</head>", " ", body, flags=re.IGNORECASE)
        body = re.sub(r"<header[\s\S]*?</header>", " ", body, flags=re.IGNORECASE)
        body = re.sub(r"<nav[\s\S]*?</nav>", " ", body, flags=re.IGNORECASE)
        body = re.sub(r"<footer[\s\S]*?</footer>", " ", body, flags=re.IGNORECASE)
        body = re.sub(r"<script[\s\S]*?</script>", " ", body, flags=re.IGNORECASE)
        body = re.sub(r"<style[\s\S]*?</style>", " ", body, flags=re.IGNORECASE)
        return body
    except Exception:
        return None

def ensure_clients() -> None:
    """Lazily initialize embedding and LLM clients.

    Embeddings use EMBEDDING_BASE_URL/EMBEDDING_API_KEY.
    LLM uses LLM_BASE_URL/LLM_API_KEY or OPENAI_API_BASE/OPENAI_API_KEY.
    """
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
            llm_base = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE")
            llm_key = (
                os.environ.get("LLM_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("EMBEDDING_API_KEY", "sk-no-key")
            )
            if llm_base:
                try:
                    _llm_client = AsyncOpenAI(base_url=llm_base, api_key=llm_key)
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
    # Support multiple provider schemas: OpenAI-compatible (data[0].embedding) and Ollama ({"embedding": [...]})
    try:
        if hasattr(resp, "data") and resp.data:
            item = resp.data[0]
            emb = getattr(item, "embedding", None)
            if emb:
                return list(emb)
        # Try dict-like access
        if isinstance(resp, dict) and "embedding" in resp:
            return list(resp["embedding"])  # type: ignore[index]
        # Some clients expose .model_dump() or .to_dict()
        if hasattr(resp, "model_dump"):
            dump = resp.model_dump()  # type: ignore[attr-defined]
            if isinstance(dump, dict):
                if "data" in dump and dump["data"]:
                    emb = dump["data"][0].get("embedding")
                    if emb:
                        return list(emb)
                if "embedding" in dump:
                    return list(dump["embedding"])  # type: ignore[index]
    except Exception:
        pass
    raise RuntimeError("No embedding data received")


async def get_title_and_summary_llm(chunk: str, url: str) -> Optional[Tuple[str, str]]:
    ensure_clients()
    # Skip LLM if pointing to Ollama's non-chat endpoint to avoid 404 spam
    if _llm_client is None or "ollama" in (
        (os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_API_BASE") or "").lower()
    ):
        return None
    try:
        system_prompt = (
            "You extract a short, informative title and a precise, helpful summary for a documentation chunk. "
            "Return JSON with keys 'title' and 'summary'. The summary should capture the main points, "
            "be specific (bullet-like sentences if useful), and stay under ~3 sentences."
        )
        model_name = os.environ.get("PRIMARY_MODEL") or "gpt-4o-mini"
        # Preprocess snippet to remove boilerplate/logo lines before sending to the LLM
        content_snippet = preprocess_markdown(chunk)[:1600]
        resp = await _llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{content_snippet}"},
            ],
            response_format={"type": "json_object"},
        )
        payload = getattr(resp.choices[0].message, "content", None)  # type: ignore[attr-defined]
        if not payload:
            return None
        # Some providers may wrap/augment JSON; try to extract the first JSON object
        text = str(payload)
        m = re.search(r"\{[\s\S]*\}", text)
        json_str = m.group(0) if m else text
        data = json.loads(json_str)
        title_raw = str(data.get("title", "")).strip()
        summary_raw = str(data.get("summary", "")).strip()
        # Sanitize results
        def clean(s: str) -> str:
            s = _strip_html(s)
            s = re.sub(r"[`*_#>]+", " ", s)
            return re.sub(r"\s+", " ", s).strip()
        title = _sanitize_zerowidth(clean(title_raw)) or None
        summary = clean(summary_raw) or None
        if not title or not summary:
            return None
        return title, summary
    except Exception as e:  # pragma: no cover - network
        logger.debug(f"LLM title/summary extraction failed, fallback to heuristic: {e}")
        return None


def extract_title_from_markdown(content: str, url: str) -> str:
    # If HTML, try to extract <h1>/<title> first
    if _looks_like_html(content):
        html_title = _extract_title_from_html(content)
        if html_title:
            return html_title
        # As a fallback, strip tags and proceed
        content = _strip_html(content)

    # Find first markdown H1/H2 or first non-empty meaningful line
    m = re.search(r"^\s*#\s+(.+)$", content, flags=re.MULTILINE)
    if not m:
        m = re.search(r"^\s*##\s+(.+)$", content, flags=re.MULTILINE)
    if m:
        return _sanitize_zerowidth(m.group(1).strip())
    def _strip_md(s: str) -> str:
        # Remove images and links and leftover markdown
        s = re.sub(r"!\[[^\]]*\]\([^\)]*\)", " ", s)  # images
        s = re.sub(r"\[[^\]]+\]\([^\)]*\)", lambda m: m.group(0).split(']')[0][1:], s)  # links -> text
        s = re.sub(r"[`*_#>]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        # Skip nav/logo/boilerplate
        if low.startswith("[windsurf docs home page") or "logo.svg" in low or low.startswith("!"):
            continue
        cleaned = _strip_md(line)
        if not cleaned or cleaned.startswith("<!doctype") or cleaned.startswith("<html"):
            continue
        return _sanitize_zerowidth(cleaned[:100])
    # URL-based fallback
    try:
        from urllib.parse import urlparse
        path = urlparse(url).path.rstrip('/').split('/')
        parts = [p for p in path if p]
        last = parts[-1] if parts else 'Page'
        last = last.replace('-', ' ').replace('_', ' ').strip()
        title = last.title() if last else 'Page'
        return f"{title}"
    except Exception:
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


def fetch_url_content_sync(url: str, timeout: int = 30) -> Optional[str]:
    try:
        raw_url = convert_github_url_to_raw(url)
        r = requests.get(raw_url, timeout=timeout, headers={"User-Agent": "ArchonCrawler/1.0"})
        if r.status_code != 200:
            logger.warning(f"HTTP {r.status_code} for {url}")
            return None
        content_type = r.headers.get("content-type", "")
        text = r.text
        # Force HTML→Markdown for Windsurf docs even if headers are odd
        if "docs.windsurf.com" in url or "text/html" in content_type or _looks_like_html(text):
            try:
                # Prefer main/article content when present to avoid nav/scripts (e.g., Mintlify)
                main_html = _extract_main_html(text)
                html_to_convert = main_html if main_html else text
                return html_converter.handle(html_to_convert)
            except Exception:
                # Fallback to raw text if conversion fails
                return text
        return text
    except Exception as e:  # pragma: no cover - network
        logger.warning(f"Request failed for {url}: {e}")
        return None


def is_allowed_url(url: str) -> bool:
    allow = [
        # Windsurf docs sections
        "https://docs.windsurf.com/windsurf/cascade/",
        "https://docs.windsurf.com/windsurf/cascade",
        "https://docs.windsurf.com/windsurf/mcp",
        "https://docs.windsurf.com/windsurf/memories",
        "https://docs.windsurf.com/windsurf/models",
        "https://docs.windsurf.com/windsurf/terminal",
        "https://docs.windsurf.com/windsurf/getting-started",
        "https://docs.windsurf.com/context-awareness/",
        "https://docs.windsurf.com/best-practices/",
        "https://docs.windsurf.com/llms.txt",
        # Context engineering community repo
        "https://github.com/coleam00/context-engineering-intro",
        "https://github.com/coleam00/context-engineering-intro/",
        "https://raw.githubusercontent.com/coleam00/context-engineering-intro/",
        # High-signal prompt/context engineering guides
        "https://platform.openai.com/docs/guides/prompt-engineering",
        "https://docs.anthropic.com/",
        # Vector DB & AI pipelines (for indexing context)
        "https://supabase.com/docs/guides/ai",
        # LlamaIndex conceptual docs
        "https://docs.llamaindex.ai/",
    ]
    return any(url.startswith(p) for p in allow)


def get_windsurf_urls() -> List[str]:
    seeds: List[str] = [
        # Cascade — Core/Workflows/Planning/Web search/Memories
        "https://docs.windsurf.com/windsurf/cascade/cascade",
        "https://docs.windsurf.com/windsurf/cascade/workflows",
        "https://docs.windsurf.com/windsurf/cascade/planning-mode",
        "https://docs.windsurf.com/windsurf/cascade/web-search",
        "https://docs.windsurf.com/windsurf/cascade/memories",
        # Windsurf — Memories (global)
        "https://docs.windsurf.com/windsurf/memories",
        # MCP
        "https://docs.windsurf.com/windsurf/cascade/mcp",
        "https://docs.windsurf.com/windsurf/mcp",
        # Context awareness & indexing
        "https://docs.windsurf.com/context-awareness/overview",
        "https://docs.windsurf.com/context-awareness/local-indexing",
        # Prompting & best practices
        "https://docs.windsurf.com/best-practices/prompt-engineering",
        # Models/Terminal/Getting started
        "https://docs.windsurf.com/windsurf/models",
        "https://docs.windsurf.com/windsurf/terminal",
        "https://docs.windsurf.com/windsurf/getting-started",
        # Index list (useful for discovery)
        "https://docs.windsurf.com/llms.txt",
        # Context Engineering resources (community/open)
        "https://github.com/coleam00/context-engineering-intro",
        "https://github.com/coleam00/context-engineering-intro/blob/main/README.md",
        "https://github.com/coleam00/context-engineering-intro/blob/main/INITIAL_EXAMPLE.md",
        # Broader prompt engineering & context resources
        "https://platform.openai.com/docs/guides/prompt-engineering",
        "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering",
        # Vector DB usage for AI assistants
        "https://supabase.com/docs/guides/ai",
        # LlamaIndex concepts for context mgmt
        "https://docs.llamaindex.ai/en/stable/getting_started/concepts.html",
    ]

    # Deduplicate, enforce allowlist
    seeds = [u for u in dict.fromkeys(seeds)]
    urls = [u for u in seeds if is_allowed_url(u)]
    return urls


# ---------------------------- Processing pipeline ---------------------------

def process_chunk_sync(chunk_text_val: str, chunk_number: int, url: str, base_title: Optional[str] = None) -> Optional[ProcessedChunk]:
    try:
        title: Optional[str] = None
        summary: Optional[str] = None

        # LLM-first extraction (safe event loop in worker thread)
        try:
            ensure_clients()
            if _llm_client is not None:
                loop = asyncio.new_event_loop()
                try:
                    llm_res = loop.run_until_complete(get_title_and_summary_llm(chunk_text_val, url))
                finally:
                    loop.close()
                if llm_res:
                    title, summary = llm_res
        except Exception as e:
            logger.debug(f"LLM extraction error, will fallback: {e}")

        # Heuristic fallback (MCP-like)
        if not title or not summary:
            t, s = get_title_and_summary_heuristic(chunk_text_val, url, chunk_number)
            # If we have a base_title for this document, prefer it for non-first chunks
            if base_title and chunk_number > 0:
                t = f"{base_title} (partie {chunk_number+1})"
            title = _sanitize_zerowidth(t)
            summary = s

        # Final sanitize to avoid accidental quotes/doctype leftovers
        safe_title = _sanitize_zerowidth(_strip_html(str(title))).strip().strip("\"'")
        safe_summary = _strip_html(str(summary)).strip()
        tl = safe_title.lower()
        if (not safe_title) or tl.startswith('<!doctype') or tl.startswith('<html'):
            # URL-based fallback title if sanitation shows HTML/doctype or empty
            safe_title = extract_title_from_markdown(chunk_text_val, url)
        # Truncate overly long titles
        if len(safe_title) > 120:
            safe_title = safe_title[:120]
        processed = ProcessedChunk(
            url=url,
            title=safe_title,
            content=chunk_text_val,
            summary=safe_summary,
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

# ----------------------- Title/Summary heuristic (MCP-like) -----------------
def extract_summary_from_chunk(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    # If HTML leaked through, strip tags and scripts/styles quickly
    if _looks_like_html(text):
        text = _strip_html(text)
    # Drop obvious nav/logo/image/link noise lines
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        low = line.strip().lower()
        if not low:
            continue
        if low.startswith('[windsurf docs home page'):
            continue
        if low.startswith('!['):
            continue
        if 'logo.svg' in low:
            continue
        # Remove common Mintlify/Windsurf nav keywords even if concatenated
        if re.search(r"ask\s*ai", low) or re.search(r"feature\s*request", low) or re.search(r"download", low) or re.search(r"search", low):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text_no_headers = re.sub(r'^#{1,6}\s+.+?\s*$', '', text, flags=re.MULTILINE)
    clean = ' '.join(text_no_headers.split())
    if not clean:
        return None
    summary_length = min(200, len(clean))
    summary = clean[:summary_length]
    if summary_length < len(clean):
        last_period = summary.rfind('.')
        if last_period > 0 and last_period + 1 < summary_length:
            summary = summary[:last_period + 1]
        else:
            summary += '...'
    return summary

def get_title_and_summary_heuristic(chunk: str, url: str, chunk_number: int = 0) -> Tuple[str, str]:
    try:
        if chunk_number == 0:
            title = extract_title_from_markdown(chunk, url)
        else:
            base = extract_title_from_markdown(chunk, url)
            title = f"{base} (partie {chunk_number+1})"
        # If the chunk still contains HTML head fragments, try meta description first
        if _looks_like_html(chunk):
            meta_desc = _extract_meta_description(chunk)
        else:
            meta_desc = None
        summary = (meta_desc or extract_summary_from_chunk(chunk) or "")
        return title, summary
    except Exception:
        if chunk_number == 0:
            return "Windsurf Documentation", ""
        return f"Windsurf Documentation (partie {chunk_number+1})", ""

# --------------------------- URL helpers (GitHub) ---------------------------
def convert_github_url_to_raw(url: str) -> str:
    try:
        if "github.com" in url and "/blob/" in url:
            return url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
        return url
    except Exception:
        return url


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
        # Guarantee overwrite semantics regardless of DB unique constraints
        try:
            supabase.table("site_pages").delete().eq("url", chunk.url).eq("chunk_number", chunk.chunk_number).execute()
        except Exception as del_e:
            logger.debug(f"Pre-delete failed (may be fine): {del_e}")
        response = supabase.table("site_pages").insert(data).execute()

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
    """Tracker de progression aligné avec l'interface utilisée dans l'UI Streamlit.

    Fournit get_status(), logs, is_running, durées, etc., similaire au tracker MCP.
    """
    def __init__(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.callback = callback
        self.urls_found = 0
        self.urls_processed = 0
        self.urls_succeeded = 0
        self.urls_failed = 0
        self.chunks_stored = 0
        self.logs: List[str] = []
        self.is_running: bool = False
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def _emit(self, event: str, message: str):
        if self.callback:
            try:
                self.callback(self.get_status())
            except Exception:
                pass

    def as_dict(self) -> Dict[str, Any]:
        return {
            "urls_found": self.urls_found,
            "urls_processed": self.urls_processed,
            "urls_succeeded": self.urls_succeeded,
            "urls_failed": self.urls_failed,
            "chunks_stored": self.chunks_stored,
            "active": self.is_running,
        }

    def get_status(self) -> Dict[str, Any]:
        duration = None
        if self.start_time is not None:
            end = self.end_time if self.end_time is not None else time.time()
            duration = end - self.start_time
        success_rate = (self.urls_succeeded / self.urls_processed * 100) if self.urls_processed else 0.0
        progress = (self.urls_processed / self.urls_found * 100) if self.urls_found else 0.0
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
            "logs": self.logs.copy(),
        }

    def start(self):
        self.is_running = True
        self.start_time = time.time()
        self.log("Windsurf workflows crawl started")

    def log(self, message: str):
        logger.info(message)
        ts = time.strftime("%H:%M:%S", time.localtime())
        self.logs.append(f"[{ts}] {message}")
        self._emit("log", message)

    def complete(self):
        self.is_running = False
        self.end_time = time.time()
        self.log("Crawl complete")


# --------------------------- Content normalization --------------------------
def preprocess_markdown(text: str) -> str:
    """Remove boilerplate/nav/logo and trim to first meaningful content block."""
    try:
        if _looks_like_html(text):
            text = html_converter.handle(_extract_main_html(text) or text)
    except Exception:
        pass
    lines: List[str] = []
    started = False
    for raw in text.splitlines():
        line = raw.strip()
        low = line.lower()
        if not line:
            if started:
                lines.append("")
            continue
        # Drop common Mintlify/Windsurf nav and boilerplate
        if low.startswith('[windsurf docs home page'):
            continue
        if low.startswith('!['):  # image lines
            continue
        if 'logo.svg' in low:
            continue
        if re.search(r"ask\s*ai", low):
            continue
        if re.search(r"feature\s*request", low):
            continue
        if re.search(r"^search", low) or '⌘k' in low or re.search(r"search\s*…?", low):
            continue
        if re.search(r"download", low):
            continue
        if 'sitemap' in low:
            continue
        if 'googletagmanager' in low or 'mintlify-assets' in low:
            continue
        if low.startswith('<!doctype') or low.startswith('<html'):
            continue
        # After first meaningful line, keep subsequent lines
        if not started:
            started = True
        lines.append(raw)
    return "\n".join(lines).strip()


def process_and_store_document_sync(url: str, content: str, tracker: Optional[CrawlProgressTracker] = None) -> int:
    try:
        # Preprocess to drop boilerplate before chunking
        normalized = preprocess_markdown(content)
        # Compute a base title once for the document for consistent chunk titles
        base_title = extract_title_from_markdown(normalized, url)
        chunks = chunk_text(normalized)
        if tracker:
            tracker.log(f"Document split into {len(chunks)} chunks: {url}")
        count = 0
        for i, chunk in enumerate(chunks):
            processed = process_chunk_sync(chunk, i, url, base_title=base_title)
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


def clear_existing_records() -> None:
    """Clear previously indexed Windsurf workflows records from Supabase."""
    try:
        sb = _get_supabase()
        sb.table("site_pages").delete().eq("metadata->>source", "windsurf_workflows").execute()
        logger.info("Cleared existing Windsurf workflows records from Supabase")
    except Exception as e:  # pragma: no cover - network/db
        logger.error(f"Failed clearing Windsurf workflows records: {e}")
"""
Generic, config‑driven crawler core for Archon.

This module unifies the pipeline:
URLs -> fetch -> normalize -> chunk -> enrich (title/summary) -> embed -> upsert (Supabase)
with progress tracking and provider‑agnostic clients.

Environment variables used (do not log their values):
- SUPABASE_URL
- SUPABASE_SERVICE_KEY
- EMBEDDING_BASE_URL (OpenAI‑compatible)
- EMBEDDING_API_KEY (optional; defaults safe)
- EMBEDDING_MODEL (default: nomic-embed-text)
- LLM_BASE_URL or OPENAI_API_BASE (OpenAI‑compatible)
- LLM_API_KEY or OPENAI_API_KEY (optional; defaults safe)
- EMBEDDING_DIMENSIONS (fallback dimension for zero vectors; default 768)

This core is intentionally minimal and self‑contained. Existing specific crawlers
(e.g., crawl_mcp_docs.py) can be refactored to call run_crawl() with a dedicated config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import os
import re
import time
import json
import logging
import threading
import asyncio
from urllib.parse import urlparse

import requests
import html2text
import httpx

try:
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore

try:
    from supabase import create_client
except Exception:  # pragma: no cover
    create_client = None  # type: ignore

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------- Configuration schema ---------------------------

class CrawlerConfig(BaseModel):
    # Metadata
    name: str = Field(..., min_length=1)
    source_tag: str = Field(..., min_length=1)
    description: Optional[str] = None

    # Sources
    seeds: List[str] = []
    sitemaps: List[str] = []
    github_repos: List[Dict[str, str]] = []  # {org, repo, branch?}
    max_pages: Optional[int] = None

    # Filters
    allow_domains: List[str] = []
    include_patterns: List[str] = []
    exclude_patterns: List[str] = []
    respect_robots_txt: bool = True

    # Fetch
    user_agent: Optional[str] = None
    timeout_seconds: int = 30
    retries: int = 2
    backoff_base_seconds: float = 1.5
    concurrency: int = 5
    mode: str = "requests"  # or "async" (not used in this minimal core)

    # Normalization
    html_to_md: bool = True
    content_selectors: Optional[List[str]] = None  # reserved for future
    remove_boilerplate_rules: Optional[List[str]] = None

    # Chunking
    chunk_size: int = 5000
    chunk_overlap: int = 0
    respect_code_blocks: bool = True

    # Enrichment
    llm_title_summary: bool = True
    require_llm_enrichment: bool = True
    llm_model: Optional[str] = None  # default from env
    llm_max_chars: int = 1200

    # Embeddings
    embedding_model: Optional[str] = None  # default from env
    embedding_provider_base_url: Optional[str] = None  # default from env

    # Storage
    table: str = "site_pages"
    upsert_on: List[str] = ["url", "chunk_number"]
    extra_metadata: Dict[str, Any] = {}


# ------------------------------- Data types --------------------------------

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]]


class CrawlProgressTracker:
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

    def _emit(self):
        if self.callback:
            try:
                self.callback(self.get_status())
            except Exception:
                pass

    def log(self, message: str):
        ts = time.strftime("%H:%M:%S", time.localtime())
        self.logs.append(f"[{ts}] {message}")
        logger.info(message)
        self._emit()

    def start(self):
        self.is_running = True
        self.start_time = time.time()
        self.log("Crawl started")

    def complete(self):
        self.is_running = False
        self.end_time = time.time()
        self.log("Crawl completed")

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


# ------------------------------ Client helpers -----------------------------

_embedding_client_lock = threading.Lock()
_llm_client_lock = threading.Lock()
_supabase_lock = threading.Lock()
_embedding_client: Optional[Any] = None
_llm_client: Optional[Any] = None
_supabase = None


def _ensure_clients():
    global _embedding_client, _llm_client, _supabase
    # Embedding client
    if _embedding_client is None and AsyncOpenAI is not None:
        with _embedding_client_lock:
            if _embedding_client is None:
                # Prefer profile-provided embedding URLs; normalize to OpenAI-compatible /v1
                raw_base = (
                    os.environ.get("EMBEDDING_BASE_URL")
                    or os.environ.get("EMBEDDING_PROVIDER_BASE_URL")
                    or os.environ.get("OLLAMA_BASE_URL")
                )
                api_key = os.environ.get("EMBEDDING_API_KEY", "sk-no-key")
                try:
                    logger.debug(
                        f"Embedding env snapshot: BASE={raw_base!r} MODEL={os.environ.get('EMBEDDING_MODEL')!r} KEY_SET={'yes' if os.environ.get('EMBEDDING_API_KEY') else 'no'}"
                    )
                except Exception:
                    pass
                def _normalize_base(u: str) -> str:
                    if not u:
                        return u
                    # Normalize common variants to OpenAI-compatible /v1 base
                    if u.endswith("/api"):
                        u = u[:-4] + "/v1"
                    # Ensure single /v1 suffix regardless of initial path (handles host:port and other paths)
                    if not u.rstrip("/").endswith("/v1"):
                        u = u + ("v1" if u.endswith("/") else "/v1")
                    return u
                base_url = _normalize_base(raw_base) if raw_base else None
                if base_url:
                    logger.info(f"Embedding client base_url: {base_url}")
                    try:
                        http_client = httpx.AsyncClient(http2=False, timeout=15.0, trust_env=False)
                        _embedding_client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
                    except Exception as e:  # pragma: no cover
                        logger.error(f"Embedding client init failed: {e}")
                        _embedding_client = None
    # LLM client
    if _llm_client is None and AsyncOpenAI is not None:
        with _llm_client_lock:
            if _llm_client is None:
                raw_base = (
                    os.environ.get("LLM_BASE_URL")
                    or os.environ.get("OPENAI_API_BASE")
                    or os.environ.get("BASE_URL")
                )
                api_key = (
                    os.environ.get("LLM_API_KEY")
                    or os.environ.get("OPENAI_API_KEY")
                    or os.environ.get("OPENROUTER_API_KEY")
                )
                try:
                    logger.debug(
                        f"LLM env snapshot: BASE={raw_base!r} MODEL={os.environ.get('PRIMARY_MODEL') or os.environ.get('OPENROUTER_MODEL')!r} KEY_SET={'yes' if api_key else 'no'}"
                    )
                except Exception:
                    pass
                def _normalize_llm_base(u: str) -> str:
                    if not u:
                        return u
                    if u.endswith("/api"):
                        u = u[:-4] + "/v1"
                    if not u.rstrip("/").endswith("/v1"):
                        u = u + ("v1" if u.endswith("/") else "/v1")
                    return u
                base = _normalize_llm_base(raw_base) if raw_base else None
                if base and api_key:
                    try:
                        # OpenRouter requires HTTP-Referer and X-Title headers to avoid 400s
                        default_headers = None
                        if "openrouter.ai" in base:
                            _ref = os.environ.get("OPENROUTER_REFERER") or os.environ.get("HTTP_REFERER") or "http://localhost"
                            _title = os.environ.get("OPENROUTER_APP_NAME") or "Archon Crawler"
                            default_headers = {"HTTP-Referer": _ref, "X-Title": _title}
                        if default_headers:
                            _llm_client = AsyncOpenAI(base_url=base, api_key=api_key, default_headers=default_headers)
                        else:
                            _llm_client = AsyncOpenAI(base_url=base, api_key=api_key)
                        logger.info(f"LLM client base_url: {base}")
                    except Exception as e:  # pragma: no cover
                        logger.warning(f"LLM client init failed: {e}")
    # Supabase client
    if _supabase is None and create_client is not None:
        with _supabase_lock:
            if _supabase is None:
                url = os.environ.get("SUPABASE_URL")
                key = os.environ.get("SUPABASE_SERVICE_KEY")
                if url and key:
                    try:
                        _supabase = create_client(url, key)
                    except Exception as e:  # pragma: no cover
                        logger.error(f"Supabase client init failed: {e}")


# ------------------------------- URL discovery -----------------------------

def _fetch_sitemap_urls(sitemap_url: str, timeout: int = 20) -> List[str]:
    try:
        r = requests.get(sitemap_url, timeout=timeout)
        if r.status_code != 200:
            return []
        xml = r.text
        locs = re.findall(r"<loc>(.*?)</loc>", xml)
        return [loc.strip() for loc in locs if loc.strip()]
    except Exception:
        return []


def _fetch_github_repo_docs(org: str, repo: str, branch: str = "HEAD") -> List[str]:
    base = f"https://api.github.com/repos/{org}/{repo}/git/trees/{branch}?recursive=1"
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
            if path.lower().endswith((".md", ".mdx")) and (
                "readme" in path.lower() or "/docs/" in path.lower() or path.lower().startswith("docs/")
            ):
                urls.append(f"https://github.com/{org}/{repo}/blob/main/{path}")
        return urls
    except Exception:
        return []


def _allowed_url(u: str, allow_domains: List[str]) -> bool:
    try:
        host = urlparse(u).hostname or ""
        return any(host.endswith(d) for d in allow_domains) if allow_domains else True
    except Exception:
        return False


def _apply_patterns(u: str, include: List[str], exclude: List[str]) -> bool:
    try:
        if include:
            if not any(re.search(p, u) for p in include):
                return False
        if exclude:
            if any(re.search(p, u) for p in exclude):
                return False
        return True
    except Exception:
        return False


# --------------------------- Fetch and normalize ----------------------------

_html = html2text.HTML2Text()
_html.ignore_links = False
_html.ignore_images = True
_html.ignore_tables = False
_html.body_width = 0


def _convert_github_raw(url: str) -> str:
    if "github.com" in url and "/blob/" in url:
        return url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return url


def fetch_url_content_sync(url: str, timeout: int = 30) -> Optional[str]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ArchonCrawler/1.0"
        }
        raw_url = _convert_github_raw(url)
        r = requests.get(raw_url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        ctype = r.headers.get("content-type", "")
        if "text/html" in ctype:
            return _html.handle(r.text)
        return r.text
    except Exception as e:
        logger.warning(f"Fetch failed {url}: {e}")
        return None


def preprocess_markdown(text: str, rules: Optional[List[str]] = None) -> str:
    # Implement minimal boilerplate trimming; rules are simple lowercase substrings to drop lines.
    if not text:
        return ""
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        low = line.lower()
        if rules and any(r in low for r in rules):
            continue
        lines.append(raw)
    return "\n".join(lines).strip()


# -------------------------------- Chunking ---------------------------------

def chunk_text(text: str, size: int = 5000, respect_code_blocks: bool = True) -> List[str]:
    chunks: List[str] = []
    if not text:
        return chunks
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        segment = text[start:end]
        if respect_code_blocks:
            fence = segment.rfind("```")
            if fence != -1 and fence > int(size * 0.3):
                end = start + fence
        elif "\n\n" in segment:
            lb = segment.rfind("\n\n")
            if lb > int(size * 0.3):
                end = start + lb
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end, start + 1)
    return chunks


# -------------------------- Enrichment and vectors --------------------------

def _sanitize_zerowidth(text: str) -> str:
    return re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]", "", text)


async def _get_title_and_summary_llm(chunk: str, url: str, model: Optional[str], max_chars: int) -> Optional[Tuple[str, str]]:
    _ensure_clients()
    if AsyncOpenAI is None:
        logger.warning("LLM client class not available; cannot enrich title/summary")
        return None
    if os.environ.get("LLM_BASE_URL") is None and os.environ.get("OPENAI_API_BASE") is None:
        logger.warning("LLM base URL not set; cannot enrich title/summary (set LLM_BASE_URL or OPENAI_API_BASE)")
        return None
    try:
        client = _llm_client
        if client is None:
            logger.warning("LLM client not initialized; cannot enrich title/summary (check API base/key)")
            return None
        # Resolve model strictly from argument or active profile
        used_model = (
            model
            or os.environ.get("PRIMARY_MODEL")
            or os.environ.get("OPENROUTER_MODEL")
        )
        if not used_model:
            logger.error("LLM enrichment disabled: no model configured (set PRIMARY_MODEL or OPENROUTER_MODEL)")
            return None
        # Normalize model id for OpenRouter (no leading 'openrouter/' accepted)
        try:
            base_url = getattr(client, "base_url", None)
            base_str = str(base_url) if base_url else ""
            if "openrouter.ai" in base_str and used_model.startswith("openrouter/"):
                used_model = used_model[len("openrouter/"):]
        except Exception:
            pass
        try:
            base_dbg = str(getattr(client, "base_url", ""))
        except Exception:
            base_dbg = ""
        logger.info(f"Enrichment: model={used_model} base={base_dbg} url={url}")
        # Build prompt
        content = chunk[:max_chars].strip()
        try:
            resp = await client.chat.completions.create(
                model=used_model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that returns strict JSON only."},
                    {"role": "user", "content": f"Summarize the following doc chunk for RAG. Return JSON with keys: title, summary. URL: {url}. Chunk: {content}"}
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            # Try to surface HTTP error details for diagnostics
            err_text = None
            try:
                resp_obj = getattr(e, "response", None)
                if resp_obj is not None:
                    err_text = getattr(resp_obj, "text", None)
                    if callable(err_text):
                        err_text = err_text()
            except Exception:
                pass
            logger.error(f"LLM enrichment request failed: {e} body={err_text}")
            return None
        msg = getattr(resp, "choices")[0].message.content  # type: ignore[index]
        data = json.loads(msg)
        title = _sanitize_zerowidth(data.get("title") or "Document")
        summary = data.get("summary") or ""
        return title[:100], summary[:400]
    except Exception as e:  # pragma: no cover
        logger.debug(f"LLM title/summary failed: {e}")
        return None


async def _batch_llm_title_summary(
    chunks: List[str],
    url: str,
    model: Optional[str],
    max_chars: int,
    batch_size: int = 20,
) -> List[Optional[Tuple[str, str]]]:
    """Enrich a list of chunks with titles/summaries using mini-batches.

    Returns a list of (title, summary) aligned to input order. Items may be None if enrichment failed.
    """
    _ensure_clients()
    if AsyncOpenAI is None or _llm_client is None:
        return [None] * len(chunks)
    # truncate chunks to respect token budget
    safe_chunks = [(c[:max_chars] if c else "") for c in chunks]
    out: List[Optional[Tuple[str, str]]] = [None] * len(chunks)

    sem = asyncio.Semaphore(max(1, int(os.environ.get("LLM_MAX_PARALLEL_BATCHES", "2"))))

    async def _run_one_batch(start: int, end: int):
        sub = safe_chunks[start:end]
        if not sub:
            return
        # Build a strict JSON instruction
        sys_msg = (
            "You generate concise JSON for documentation chunks. "
            "For each input item, output a short, informative title (<=100 chars) and a concise summary (<=400 chars). "
            'Respond ONLY with a JSON array of objects: [{"index": <int>, "title": <str>, "summary": <str>}].'
        )
        items = [{"index": i + start, "chunk": sub[i]} for i in range(len(sub))]
        user_msg = json.dumps({"url": url, "items": items})
        try:
            async with sem:
                resp = await _llm_client.chat.completions.create(
                    model=(model or os.environ.get("PRIMARY_MODEL") or os.environ.get("OPENROUTER_MODEL") or ""),
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.2,
                )
                msg = getattr(resp, "choices")[0].message.content  # type: ignore[index]
                data = json.loads(msg)
                if isinstance(data, list):
                    for obj in data:
                        try:
                            idx = int(obj.get("index"))
                            title = _sanitize_zerowidth((obj.get("title") or "Document"))[:100]
                            summary = (obj.get("summary") or "")[:400]
                            if 0 <= idx < len(out):
                                out[idx] = (title, summary)
                        except Exception:
                            continue
        except Exception as e:
            logger.debug(f"Batch LLM enrichment failed ({start}-{end}): {e}")

    # Dispatch batches with limited parallelism
    tasks: List[asyncio.Task] = []
    for s in range(0, len(safe_chunks), batch_size):
        e = min(len(safe_chunks), s + batch_size)
        tasks.append(asyncio.create_task(_run_one_batch(s, e)))
    if tasks:
        await asyncio.gather(*tasks)

    return out


async def _get_embedding(text: str, model: Optional[str]) -> List[float]:
    _ensure_clients()
    used_model = model or os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    try:
        client = _embedding_client
        if client is None:
            raise RuntimeError("Embedding client not initialized; set EMBEDDING_BASE_URL")
        try:
            base_str = str(getattr(client, "base_url", ""))
        except Exception:
            base_str = ""
        logger.info(f"Embedding request via SDK: model={used_model} base={base_str}")
        resp = await client.embeddings.create(model=used_model, input=text)
        # OpenAI‑style
        if hasattr(resp, "data") and resp.data:
            item = resp.data[0]
            emb = getattr(item, "embedding", None)
            if emb:
                return list(emb)
        # Dict‑style
        if isinstance(resp, dict) and "embedding" in resp:
            return list(resp["embedding"])  # type: ignore[index]
        if hasattr(resp, "model_dump"):
            dump = resp.model_dump()  # type: ignore[attr-defined]
            if isinstance(dump, dict):
                if dump.get("data"):
                    emb = dump["data"][0].get("embedding")
                    if emb:
                        return list(emb)
                if dump.get("embedding"):
                    return list(dump["embedding"])  # type: ignore[index]
        raise RuntimeError("No embedding returned")
    except Exception as e:
        logger.warning(f"SDK embeddings error: {repr(e)}")
        # Direct HTTP fallback to accommodate custom routers (e.g., Allama at host root)
        try:
            base = ""
            try:
                base = str(getattr(_embedding_client, "base_url", ""))
            except Exception:
                base = ""
            if not base:
                base = os.environ.get("EMBEDDING_BASE_URL", "").rstrip("/")
            if base:
                suffixes = ["/v1/embeddings", "/embeddings", "/api/embeddings"]
                payload = {"model": used_model, "input": text}
                async with httpx.AsyncClient(http2=False, timeout=15.0, trust_env=False) as hc:
                    for suf in suffixes:
                        url = base.rstrip("/") + suf
                        try:
                            logger.info(f"Embedding HTTP fallback try: {url}")
                            r = await hc.post(url, json=payload, headers={"Content-Type": "application/json"})
                            logger.info(f"Embedding HTTP fallback status: {r.status_code} for {url}")
                            if r.status_code == 200:
                                try:
                                    data = r.json()
                                except Exception:
                                    data = json.loads(r.text)
                                if isinstance(data, dict):
                                    if "data" in data and isinstance(data["data"], list) and data["data"]:
                                        item = data["data"][0]
                                        emb = item.get("embedding") if isinstance(item, dict) else None
                                        if emb:
                                            return list(emb)
                                    if "embedding" in data:
                                        return list(data["embedding"])  # type: ignore[index]
                        except Exception as ie:
                            logger.warning(f"Embedding HTTP fallback error for {url}: {repr(ie)}")
                            continue
        except Exception:
            pass
        logger.error(f"Embedding failed: {e}")
        dim = int(os.environ.get("EMBEDDING_DIMENSIONS", 768))
        logger.warning(f"Falling back to zero-vector of dimension {dim}; set EMBEDDING_BASE_URL/EMBEDDING_MODEL or adjust EMBEDDING_DIMENSIONS (e.g., 768 or 1536)")
        return [0.0] * dim


# -------------------------------- Storage ----------------------------------

def _insert_chunk_sync(table: str, chunk: ProcessedChunk) -> bool:
    _ensure_clients()
    if _supabase is None:
        raise RuntimeError("Supabase client not initialized; set SUPABASE_URL")
    data = {
        "url": chunk.url,
        "chunk_number": chunk.chunk_number,
        "title": chunk.title,
        "summary": chunk.summary,
        "content": chunk.content,
        "metadata": chunk.metadata,
    }
    if chunk.embedding is not None:
        data["embedding"] = chunk.embedding
    try:
        # Idempotent write: use upsert on (url, chunk_number)
        _supabase.table(table).upsert(data, on_conflict="url,chunk_number").execute()
        return True
    except Exception as e:
        logger.error(f"Supabase insert error: {e}")
        return False


# ------------------------------ Orchestration -------------------------------

def _gather_urls(cfg: CrawlerConfig) -> List[str]:
    urls: List[str] = []
    urls.extend(cfg.seeds or [])
    for sm in cfg.sitemaps or []:
        urls.extend(_fetch_sitemap_urls(sm))
    for rec in cfg.github_repos or []:
        org = rec.get("org")
        repo = rec.get("repo")
        branch = rec.get("branch", "HEAD")
        if org and repo:
            urls.extend(_fetch_github_repo_docs(org, repo, branch))
    # Filter + deduplicate
    seen = set()
    out: List[str] = []
    for u in urls:
        if u in seen:
            continue
        if not _allowed_url(u, cfg.allow_domains):
            continue
        if not _apply_patterns(u, cfg.include_patterns, cfg.exclude_patterns):
            continue
        seen.add(u)
        out.append(u)
        if cfg.max_pages and len(out) >= cfg.max_pages:
            break
    return out


def _fallback_title(url: str, chunk_number: int) -> str:
    base = url.split("/")[-1] or "Document"
    if "." in base:
        base = base.split(".")[0]
    return f"{base} (part {chunk_number + 1})" if chunk_number else base


async def _process_and_store(url: str, raw_text: str, cfg: CrawlerConfig, tracker: Optional[CrawlProgressTracker]) -> int:
    normalized = preprocess_markdown(raw_text, cfg.remove_boilerplate_rules)
    chunks = chunk_text(normalized, cfg.chunk_size, cfg.respect_code_blocks)
    if tracker:
        tracker.log(f"Split into {len(chunks)} chunks: {url}")
    stored = 0
    # Precompute titles/summaries in batches when enabled
    batch_results: List[Optional[Tuple[str, str]]] = []
    if cfg.llm_title_summary:
        try:
            batch_size = int(os.environ.get("LLM_ENRICH_BATCH_SIZE", "20"))
            batch_results = await _batch_llm_title_summary(chunks, url, cfg.llm_model, cfg.llm_max_chars, batch_size)
        except Exception:
            batch_results = [None] * len(chunks)
    for i, chunk in enumerate(chunks):
        # Title/Summary (mandatory if require_llm_enrichment)
        title: Optional[str] = None
        summary: str = ""
        if cfg.llm_title_summary:
            # prefer batched result, fallback to per-chunk call
            ts = batch_results[i] if i < len(batch_results) else None
            if ts is None:
                ts = await _get_title_and_summary_llm(chunk, url, cfg.llm_model, cfg.llm_max_chars)
            if ts:
                title, summary = ts
        if title is None and cfg.require_llm_enrichment:
            if tracker:
                tracker.log(f"Skip chunk (no enrichment): {url} [chunk {i}]")
            continue
        if not title:
            title, summary = _fallback_title(url, i), (chunk[:200] + "...") if len(chunk) > 200 else chunk
        # Embedding
        embedding = await _get_embedding(chunk, cfg.embedding_model)
        meta = {
            "source": cfg.source_tag,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta.update(cfg.extra_metadata or {})
        processed = ProcessedChunk(
            url=url,
            chunk_number=i,
            title=title,
            summary=summary,
            content=chunk,
            metadata=meta,
            embedding=embedding,
        )
        if _insert_chunk_sync(cfg.table, processed):
            stored += 1
            if tracker:
                tracker.chunks_stored += 1
    return stored


def run_crawl(config: CrawlerConfig, progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None) -> str:
    """Run a crawl based on config in a background thread and return a crawl_id."""
    tracker = CrawlProgressTracker(progress_cb)
    crawl_id = f"crawl_{int(time.time()*1000)}"

    def worker():
        try:
            # Propagate per-crawl embedding overrides into environment for client init
            if getattr(config, "embedding_provider_base_url", None):
                os.environ["EMBEDDING_BASE_URL"] = config.embedding_provider_base_url  # type: ignore[attr-defined]
                logger.info(f"Crawl override EMBEDDING_BASE_URL={config.embedding_provider_base_url}")
            if getattr(config, "embedding_model", None):
                os.environ["EMBEDDING_MODEL"] = config.embedding_model  # type: ignore[attr-defined]
                logger.info(f"Crawl override EMBEDDING_MODEL={config.embedding_model}")
            # Avoid accidental fallbacks overriding the chosen base
            if os.environ.get("EMBEDDING_FALLBACK_BASE_URL"):
                os.environ.pop("EMBEDDING_FALLBACK_BASE_URL", None)
            # Emit a concise config summary for diagnostics
            try:
                logger.info(
                    "Crawl config: name=%s source=%s seeds=%d sitemaps=%d max_pages=%s llm_title_summary=%s require_llm_enrichment=%s chunk_size=%d",
                    getattr(config, "name", "-"),
                    getattr(config, "source_tag", "-"),
                    len(getattr(config, "seeds", []) or []),
                    len(getattr(config, "sitemaps", []) or []),
                    str(getattr(config, "max_pages", None)),
                    str(getattr(config, "llm_title_summary", False)),
                    str(getattr(config, "require_llm_enrichment", True)),
                    int(getattr(config, "chunk_size", 5000)),
                )
                logger.debug(
                    "Env snapshot (LLM/Emb): LLM_BASE_URL=%r OPENAI_API_BASE=%r PRIMARY_MODEL=%r OPENROUTER_MODEL=%r EMBEDDING_BASE_URL=%r EMBEDDING_MODEL=%r",
                    os.environ.get("LLM_BASE_URL"),
                    os.environ.get("OPENAI_API_BASE"),
                    os.environ.get("PRIMARY_MODEL"),
                    os.environ.get("OPENROUTER_MODEL"),
                    os.environ.get("EMBEDDING_BASE_URL"),
                    os.environ.get("EMBEDDING_MODEL"),
                )
            except Exception:
                pass
            tracker.start()
            urls = _gather_urls(config)
            tracker.urls_found = len(urls)
            tracker.log(f"Collected {len(urls)} candidate URLs")
            if not urls:
                tracker.complete()
                return
            sem = threading.Semaphore(max(1, config.concurrency))

            async def process_url(u: str):
                try:
                    text = fetch_url_content_sync(u, timeout=config.timeout_seconds)
                    if not text:
                        tracker.urls_processed += 1
                        tracker.urls_failed += 1
                        tracker.log(f"No content: {u}")
                        return
                    stored = await _process_and_store(u, text, config, tracker)
                    tracker.urls_processed += 1
                    if stored > 0:
                        tracker.urls_succeeded += 1
                        tracker.log(f"OK {u}: {stored} chunks")
                    else:
                        tracker.urls_failed += 1
                        tracker.log(f"Failed to store: {u}")
                except Exception as e:
                    tracker.urls_processed += 1
                    tracker.urls_failed += 1
                    tracker.log(f"Error {u}: {e}")
                finally:
                    sem.release()

            async def runner():
                loop = asyncio.get_running_loop()
                tasks: List[asyncio.Task] = []
                for u in urls:
                    sem.acquire()
                    tasks.append(loop.create_task(process_url(u)))
                if tasks:
                    await asyncio.gather(*tasks)

            asyncio.run(runner())
            tracker.complete()
        except Exception as e:
            tracker.log(f"Fatal error: {e}")
            tracker.complete()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return crawl_id

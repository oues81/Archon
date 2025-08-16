from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field, ValidationError


DEFAULT_API_URL = os.getenv("CV_API_URL", "http://192.168.28.245:8000")


class ParseRequest(BaseModel):
    filename: str
    file_base64: str


class ParsePagesRequest(ParseRequest):
    pages: Optional[str] = None  # API accepts string or null


class ParseResponse(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[Dict[str, Any]] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)


class ParsePagesResponse(BaseModel):
    page_count: int
    results: List[ParseResponse]


class CVAnalyzerArgs(BaseModel):
    # One of the following modes is required:
    files: Optional[List[str]] = None  # Paths on disk
    text_base64: Optional[str] = None  # Already base64-encoded content
    filename: Optional[str] = None

    # Optional: restrict to pages, as string accepted by API (e.g., "1,2-3")
    pages: Optional[str] = None

    # Override API URL if needed
    api_url: Optional[str] = None

    def validate_mode(self) -> Tuple[str, Any]:
        """Validate mutually exclusive input modes and return (mode, payload)."""
        has_files = bool(self.files)
        has_text = bool(self.text_base64 and self.filename)
        if has_files == has_text:
            raise ValueError("Provide either 'files' or both 'text_base64' and 'filename'.")
        return ("files" if has_files else "inline", self.files if has_files else (self.filename, self.text_base64))


async def _post_json(client: httpx.AsyncClient, url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    resp = await client.post(url, json=data, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()

async def _get_json(client: httpx.AsyncClient, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    resp = await client.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _build_headers(correlation_id: Optional[str] = None) -> Dict[str, str]:
    """Construct default headers including correlation and API key if available.

    Looks for API key in env vars CV_API_KEY or RH_CV_API_KEY and sets both
    Authorization: Bearer <key> and X-API-Key headers for compatibility.
    """
    headers: Dict[str, str] = {}
    if correlation_id:
        headers["X-Correlation-Id"] = correlation_id
    api_key = os.getenv("CV_API_KEY") or os.getenv("RH_CV_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-API-Key"] = api_key
    return headers


def _b64_of_file(path: str) -> Tuple[str, str]:
    with open(path, "rb") as f:
        b = f.read()
    return os.path.basename(path), base64.b64encode(b).decode("utf-8")


async def analyze(
    files: Optional[List[str]] = None,
    text_base64: Optional[str] = None,
    filename: Optional[str] = None,
    pages: Optional[str] = None,
    api_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyze CVs by calling the RH CV Parser API.

    - If `files` provided, each file is processed via /parse or /parse_pages.
    - Else `text_base64` + `filename` is processed once.

    Returns a JSON dict with a stable shape:
      {
        "mode": "files"|"inline",
        "api_url": "...",
        "items": [
          {"filename": "...", "response": <raw API JSON>, "ok": true, "error": null}
        ]
      }
    """
    args = CVAnalyzerArgs(
        files=files, text_base64=text_base64, filename=filename, pages=pages, api_url=api_url
    )
    mode, payload = args.validate_mode()

    base_url = (api_url or DEFAULT_API_URL).rstrip("/")
    parse_url = f"{base_url}/parse"
    parse_pages_url = f"{base_url}/parse_pages"

    results: List[Dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        if mode == "files":
            assert isinstance(payload, list)
            for path in payload:
                try:
                    fn, b64 = _b64_of_file(path)
                    if pages:
                        req = ParsePagesRequest(filename=fn, file_base64=b64, pages=pages)
                        raw = await _post_json(client, parse_pages_url, req.model_dump(), headers=_build_headers())
                    else:
                        req = ParseRequest(filename=fn, file_base64=b64)
                        raw = await _post_json(client, parse_url, req.model_dump(), headers=_build_headers())
                    results.append({"filename": fn, "response": raw, "ok": True, "error": None})
                except Exception as e:
                    results.append({"filename": os.path.basename(path), "response": None, "ok": False, "error": str(e)})
        else:
            # inline
            assert isinstance(payload, tuple)
            fn, b64 = payload
            try:
                if pages:
                    req = ParsePagesRequest(filename=fn, file_base64=b64, pages=pages)
                    raw = await _post_json(client, parse_pages_url, req.model_dump(), headers=_build_headers())
                else:
                    req = ParseRequest(filename=fn, file_base64=b64)
                    raw = await _post_json(client, parse_url, req.model_dump(), headers=_build_headers())
                results.append({"filename": fn, "response": raw, "ok": True, "error": None})
            except Exception as e:
                results.append({"filename": fn, "response": None, "ok": False, "error": str(e)})

    return {"mode": mode, "api_url": base_url, "items": results}


async def parse_v2(
    file_base64: str,
    filename: str,
    api_url: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Proxy to RH CV Parser API /parse_v2.

    Inputs:
      - file_base64: base64 of the file content
      - filename: original filename
      - api_url: override base URL
      - correlation_id: optional X-Correlation-Id header

    Returns: raw API JSON under stable wrapper { api_url, response }
    """
    base_url = (api_url or DEFAULT_API_URL).rstrip("/")
    url = f"{base_url}/parse_v2"
    payload = {"filename": filename, "file_base64": file_base64}
    headers = _build_headers(correlation_id)
    async with httpx.AsyncClient() as client:
        raw = await _post_json(client, url, payload, headers=headers)
    return {"api_url": base_url, "response": raw}


# ------------------------
# Preflight / Health Checks
# ------------------------

async def ping(api_url: Optional[str] = None, correlation_id: Optional[str] = None) -> Dict[str, Any]:
    base_url = (api_url or DEFAULT_API_URL).rstrip("/")
    url = f"{base_url}/ping"
    headers = _build_headers(correlation_id)
    async with httpx.AsyncClient() as client:
        try:
            data = await _get_json(client, url, headers=headers)
            return {"ok": True, "api_url": base_url, "data": data}
        except Exception as e:
            return {"ok": False, "api_url": base_url, "error": str(e)}


async def get_openapi(api_url: Optional[str] = None, correlation_id: Optional[str] = None) -> Dict[str, Any]:
    base_url = (api_url or DEFAULT_API_URL).rstrip("/")
    url = f"{base_url}/openapi.json"
    headers = _build_headers(correlation_id)
    async with httpx.AsyncClient() as client:
        try:
            spec = await _get_json(client, url, headers=headers)
            return {"ok": True, "api_url": base_url, "spec": spec}
        except Exception as e:
            return {"ok": False, "api_url": base_url, "error": str(e)}


async def get_version(api_url: Optional[str] = None, correlation_id: Optional[str] = None) -> Dict[str, Any]:
    base_url = (api_url or DEFAULT_API_URL).rstrip("/")
    url = f"{base_url}/version"
    headers = {"X-Correlation-Id": correlation_id} if correlation_id else None
    async with httpx.AsyncClient() as client:
        try:
            data = await _get_json(client, url, headers=headers)
            return {"ok": True, "api_url": base_url, "data": data}
        except Exception as e:
            return {"ok": False, "api_url": base_url, "error": str(e)}


async def get_profiles_version(api_url: Optional[str] = None, correlation_id: Optional[str] = None) -> Dict[str, Any]:
    base_url = (api_url or DEFAULT_API_URL).rstrip("/")
    url = f"{base_url}/profiles/version"
    headers = {"X-Correlation-Id": correlation_id} if correlation_id else None
    async with httpx.AsyncClient() as client:
        try:
            data = await _get_json(client, url, headers=headers)
            return {"ok": True, "api_url": base_url, "data": data}
        except Exception as e:
            return {"ok": False, "api_url": base_url, "error": str(e)}


def _version_ok(current: Optional[str], minimum: Optional[str]) -> bool:
    if not minimum or not current:
        return True
    def split(v: str) -> List[int]:
        return [int(x) for x in v.strip().lstrip('v').split('.') if x.isdigit()]
    c = split(current)
    m = split(minimum)
    # compare lexicographically by parts
    for i in range(max(len(c), len(m))):
        cv = c[i] if i < len(c) else 0
        mv = m[i] if i < len(m) else 0
        if cv != mv:
            return cv >= mv
    return True


async def health_check(
    api_url: Optional[str] = None,
    correlation_id: Optional[str] = None,
    require_v2: bool = True,
    min_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Run parser preflight checks and return a structured report.

    - ping
    - openapi has /parse_v2 (if require_v2)
    - version >= min_version (if provided)
    - profiles/version fetched (best effort)
    """
    base_url = (api_url or DEFAULT_API_URL).rstrip("/")
    report: Dict[str, Any] = {"api_url": base_url}
    p = await ping(base_url, correlation_id)
    report["ping"] = p
    oa = await get_openapi(base_url, correlation_id)
    report["openapi"] = {k: oa.get(k) for k in ("ok", "error")}
    if oa.get("ok"):
        spec = oa.get("spec", {})
        paths = spec.get("paths", {}) if isinstance(spec, dict) else {}
        has_v2 = "/parse_v2" in paths
        report["openapi"]["has_parse_v2"] = has_v2
    else:
        report["openapi"]["has_parse_v2"] = False
    ver = await get_version(base_url, correlation_id)
    cur_ver = ver.get("data", {}).get("version") if ver.get("ok") else None
    report["version"] = {"ok": ver.get("ok"), "current": cur_ver, "meets_min": _version_ok(cur_ver, min_version)}
    profv = await get_profiles_version(base_url, correlation_id)
    report["profiles_version"] = {"ok": profv.get("ok"), "data": profv.get("data") if profv.get("ok") else None}
    # overall decision
    fail_reasons: List[str] = []
    if not p.get("ok"):
        fail_reasons.append("ping failed")
    if require_v2 and not report["openapi"].get("has_parse_v2"):
        fail_reasons.append("/parse_v2 missing in openapi.json")
    if min_version and not report["version"].get("meets_min"):
        fail_reasons.append("version below minimum")
    report["ok"] = len(fail_reasons) == 0
    if not report["ok"]:
        report["error"] = ", ".join(fail_reasons)
    return report

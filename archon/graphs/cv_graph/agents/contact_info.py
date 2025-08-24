# -*- coding: utf-8 -*-
"""ContactInfoAgent (skeleton): deterministic-only for email/phone/LinkedIn + file metadata.
"""
from __future__ import annotations
import re, hashlib
from pathlib import Path
from typing import Dict, Any, Optional

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
LINKEDIN_RE = re.compile(r"https?://(www\.)?linkedin\.com/in/[A-Za-z0-9\-_/]+", re.I)


def _sha256_file(p: Optional[str]) -> str:
    try:
        if not p or not Path(p).exists():
            return ""
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception:
        return ""


def run(normalized_text: str, file_path: Optional[str]) -> Dict[str, Any]:
    email = (EMAIL_RE.search(normalized_text or "") or [None])
    email = email.group(0) if hasattr(email, "group") else ""
    linkedin = (LINKEDIN_RE.search(normalized_text or "") or [None])
    linkedin = linkedin.group(0) if hasattr(linkedin, "group") else ""
    phone = ""  # parsed later by normalizers in post-processing
    out: Dict[str, Any] = {
        "NomFichier": Path(file_path).name if file_path else "",
        "UrlFichier": str(Path(file_path).resolve()) if file_path else "",
        "ResumeHash": _sha256_file(file_path),
        "Email": email.lower() if email else "",
        "Telephone": phone,
        "LinkedInUrl": linkedin,
    }
    return out

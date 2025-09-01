# -*- coding: utf-8 -*-
"""
CV Graph - Agentic flow for CV extraction, scoring, aggregation, and upsert

Pattern mirrors archon_graph.py:
- Unified provider via get_llm_instance()
- PydanticAI agents with retry + fallback
- LangGraph StateGraph orchestration with checkpoints

Nodes:
- extract_candidate_base
- score_skills | score_experience | score_education | score_langues? | score_localisation?
- aggregate_scores
- upsert_sharepoint (optional shell-out to scripts/upsert_candidates_pnp.ps1)

# Global semaphore for scorer concurrency
_scorer_sem: Optional[asyncio.Semaphore] = None

def _get_scorer_semaphore(llm_cfg: Dict[str, Any]) -> asyncio.Semaphore:
    global _scorer_sem
    if _scorer_sem is None:
        max_c = int(llm_cfg.get("MAX_CONCURRENCY", 3))
        if max_c <= 0:
            max_c = 1
        _scorer_sem = asyncio.Semaphore(max_c)
    return _scorer_sem

Environment/profile-config expectations (llm_config):
- LLM_PROVIDER: "ollama" | "openrouter" | "openai"
- EXTRACTOR_MODEL: str
- SCORER_MODEL or SCORER_MODEL_SKILLS/etc: str
- AGGREGATOR_MODEL: str (optional; simple python aggregation if absent)
- OLLAMA_BASE_URL: str for ollama
- OLLAMA_MODEL: str fallback for errors
- WEIGHTS: { skills, experience, education, langues, localisation }
- ENABLE_LANGUES: bool
- ENABLE_LOCALISATION: bool
- TIMEOUT_S: int
"""

import os
import json
import logging
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
import asyncio
from k.cv_schema_adapter import build_canonical_payload, load_candidats_schema

logger = logging.getLogger(__name__)

# Reuse helpers from the main archon graph
from k.graphs.k.app.graph import (
    get_llm_instance,
    _with_retries,  # type: ignore
    _fallback_model,  # type: ignore
)
from k.core.utils.utils import build_llm_config_from_active_profile

# Strict JSON instruction for scorers
STRICT_JSON_SCORING = (
    "Tu dois répondre UNIQUEMENT en JSON valide, sans aucun texte avant/après. "
    "Schéma strict: {\"score\": entier 0-100, \"evidence\": liste de chaînes (0-5 éléments)}. "
    "N'inclus ni commentaires ni explications en dehors du JSON."
)

# Prompts are centralized here
from k.prompts.agent_prompts import (
    cv_extract_prompt,
    cv_skills_prompt,
    cv_experience_prompt,
    cv_education_prompt,
    cv_languages_prompt,
    cv_location_prompt,
)


def _enrich_candidate_from_text(cv_text: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristically fill missing base fields from raw CV text.

    Focuses on baseline CSV completeness, independent of LLM quality.
    """
    text = cv_text or ""
    cand = dict(candidate or {})

    # Email
    if not cand.get("email"):
        m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        if m:
            cand["email"] = m.group(0)

    # Telephone (strict parsing + normalization)
    if not cand.get("telephone"):
        m = re.search(r"(?:(?:\+\d{1,3}[ \.-]?)?(?:\(?\d{1,4}\)?[ \.-]?)?\d{2,4}[ \.-]?\d{2,4}[ \.-]?\d{2,4})", text)
        if m:
            cand["telephone"] = _parse_phone_strict(m.group(0)) or ""

    # LinkedIn
    if not (cand.get("linkedin") or cand.get("linkedin_url")):
        m = re.search(r"https?://(?:www\.)?linkedin\.com/[^\s>\)]+", text, re.IGNORECASE)
        if m:
            cand["linkedin"] = m.group(0)

    # Ville / Région / Pays from common labels
    def _first_group(patterns):
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().strip(",.;:| ")
        return ""

    if not cand.get("ville"):
        cand["ville"] = _first_group([
            r"(?:ville|city|localisation|location)\s*[:\-]\s*([A-Za-zÀ-ÖØ-öø-ÿ '\-]{2,40})",
        ]) or cand.get("ville", "")

    if not cand.get("region"):
        cand["region"] = _first_group([
            r"(?:région|region|state|province)\s*[:\-]\s*([A-Za-zÀ-ÖØ-öø-ÿ '\-]{2,40})",
        ]) or cand.get("region", "")

    if not cand.get("pays"):
        country = _first_group([
            r"(?:pays|country)\s*[:\-]\s*([A-Za-zÀ-ÖØ-öø-ÿ '\-]{2,40})",
        ])
        if not country:
            # fallback: known tokens
            for tok in ["Canada", "France", "Belgique", "Suisse", "Québec"]:
                if re.search(rf"\b{re.escape(tok)}\b", text, re.IGNORECASE):
                    country = tok
                    break
        # Normalize Québec case: Québec is a region/province, not a country
        if country:
            if country.lower() == "québec":
                cand.setdefault("region", "Québec")
                cand["pays"] = "Canada"
            else:
                cand["pays"] = country

    # Infer Canada/Québec from Montréal mention if still missing
    if (not cand.get("pays") or not cand.get("region")) and re.search(r"\bMontréal\b", text, re.IGNORECASE):
        cand.setdefault("region", "Québec")
        cand.setdefault("pays", "Canada")

    # Compétences: first comma/semicolon separated list under Skills/Compétences
    tech = (cand.get("competences") or {}).get("tech") if isinstance(cand.get("competences"), dict) else None
    if not tech:
        m = re.search(r"(?i)(?:skills|compétences?|competences?)\s*[:\-]?\s*(.+)", text)
        if m:
            line = m.group(1).splitlines()[0]
            parts = re.split(r"[;,]\s*", line)
            guessed = [p.strip() for p in parts if 1 < len(p.strip()) <= 50][:20]
            if guessed:
                cand.setdefault("competences", {})
                if isinstance(cand["competences"], dict):
                    cand["competences"].setdefault("tech", guessed)

    # Normalize minimal structure
    cand.setdefault("competences", {})
    if isinstance(cand["competences"], dict):
        cand["competences"].setdefault("tech", [])
        cand["competences"].setdefault("soft", [])
        cand["competences"].setdefault("mots_cles", [])

    for k in ["nom_complet", "ville", "region", "pays", "email", "telephone"]:
        cand.setdefault(k, "")

    # ---------------- Additional deterministic enrichments for canonical fields ----------------
    # Autorisation de travail (bool-like → string yes/no or specific permit)
    if not cand.get("autorisation_travail"):
        if re.search(r"autorisé\s+à\s+travailler|permis\s+de\s+travail|eligible\s+pour\s+travailler", text, re.I):
            cand["autorisation_travail"] = "Oui"
        elif re.search(r"besoin\s+de\s+visa|sans\s+permis\s+de\s+travail|non\s+autorisé", text, re.I):
            cand["autorisation_travail"] = "Non"

    # Disponibilité (date)
    if not cand.get("disponibilite_date"):
        m = re.search(r"disponibilit[ée]\s*[:\-]?\s*(?:immédiate|immediate)", text, re.I)
        if m:
            cand["disponibilite_date"] = datetime.utcnow().strftime("%Y-%m-%d")
        else:
            m = re.search(r"(?:disponible\s*(?:dès|a\s+partir\s+du)|disponibilit[ée])\s*[:\-]?\s*(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|\w+\s+\d{4})", text, re.I)
            if m:
                cand["disponibilite_date"] = m.group(1)

    # Prêt à se relocaliser (bool)
    if not cand.get("pret_relocaliser"):
        if re.search(r"(mobilit[ée]\s+nationale|international[e]?|pr[êe]t\s+à\s+(?:d[ée]m[ée]nager|relocaliser))", text, re.I):
            cand["pret_relocaliser"] = True
        elif re.search(r"non\s+mobile|pas\s+mobile|pas\s+de\s+mobilit[ée]", text, re.I):
            cand["pret_relocaliser"] = False

    # Poste cible / Domaine
    if not cand.get("poste_cible"):
        m = re.search(r"(?:poste\s+recherch[é]?:?|objectif\s+professionnel\s*:?|cible\s*:?|souhait\s*:?)[\s\-]*([^\n\r]{3,80})", text, re.I)
        if m:
            cand["poste_cible"] = m.group(1).strip()
    if not cand.get("domaine") and cand.get("poste_cible"):
        # Rough domain extraction from target
        dom = cand["poste_cible"].lower()
        if any(tok in dom for tok in ["données", "data", "bi", "analyst"]):
            cand["domaine"] = "Données"
        elif any(tok in dom for tok in ["dev", "software", "ingénieur", "backend", "frontend"]):
            cand["domaine"] = "Développement"

    # Seniorité
    if not cand.get("seniorite"):
        if re.search(r"\b(stagiaire|intern)\b", text, re.I):
            cand["seniorite"] = "Stagiaire"
        elif re.search(r"\b(junior|débutant)\b", text, re.I):
            cand["seniorite"] = "Junior"
        elif re.search(r"\b(interm[ée]diaire|confirm[ée])\b", text, re.I):
            cand["seniorite"] = "Intermédiaire"
        elif re.search(r"\b(senior|lead|principal)\b", text, re.I):
            cand["seniorite"] = "Senior"

    # Niveau d'études
    if not cand.get("niveau_etudes"):
        if re.search(r"\b(bac\+?5|master|ing[ée]nieur|mba)\b", text, re.I):
            cand["niveau_etudes"] = "Bac+5/Master"
        elif re.search(r"\b(bac\+?3|licence|bachelor)\b", text, re.I):
            cand["niveau_etudes"] = "Bac+3/Licence"
        elif re.search(r"\b(bac\+?2|dut|bts)\b", text, re.I):
            cand["niveau_etudes"] = "Bac+2"

    # Langues
    if not cand.get("langues"):
        langs = []
        # Français / Anglais with level
        for lang, key in [(r"fran[cç]ais", "fr"), (r"anglais", "en"), (r"espagnol", "es")] :
            m = re.search(rf"\b{lang}\b\s*(?:\(([^)]+)\)|\-\s*(\w+))?", text, re.I)
            if m:
                level = (m.group(1) or m.group(2) or "").strip()
                langs.append({"lang": key, "niveau": level})
        if langs:
            cand["langues"] = langs

    # Salaires
    if cand.get("salaire_souhaite") in (None, ""):
        m = re.search(r"(pr[ée]tentions|salaire\s+souhait[é])\s*[:\-]?\s*([0-9][0-9\s\.,]{2,})(?:\s*(?:€|eur|euros|\$|cad|usd))?", text, re.I)
        if m:
            cand["salaire_souhaite"] = re.sub(r"[^0-9]", "", m.group(2))
    if cand.get("salaire_actuel") in (None, ""):
        m = re.search(r"salaire\s+actuel\s*[:\-]?\s*([0-9][0-9\s\.,]{2,})(?:\s*(?:€|eur|euros|\$|cad|usd))?", text, re.I)
        if m:
            cand["salaire_actuel"] = re.sub(r"[^0-9]", "", m.group(1))

    return cand


def _parse_phone_strict(raw: str) -> Optional[str]:
    """Normalize phone numbers and reject year-like patterns.

    Rules:
    - Strip non-digits except leading '+'
    - Require at least 10 digits (common minimum)
    - Reject patterns like '2021-2024' or ranges
    - Return E.164-like when possible (keep leading '+') else plain digits
    """
    if not raw:
        return None
    s = raw.strip()
    # Reject typical year range patterns
    if re.search(r"\b(19|20)\d{2}\s*[–\-]\s*(19|20)\d{2}\b", s):
        return None
    # Keep leading '+' if present, drop other non-digits
    lead_plus = s.startswith("+")
    digits = re.sub(r"\D", "", s)
    if len(digits) < 10:
        return None
    if lead_plus:
        return "+" + digits
    return digits

# --- Deterministic coordinates micro-agent -----------------------------------
def _coordinates_micro_agent(cv_text: str) -> Dict[str, Any]:
    """Deterministically extract coordinates from raw text.

    Returns minimal fields with conservative regex parsing and normalization.
    Produces only high-confidence values; leaves fields empty if not found.
    """
    text = cv_text or ""

    # Email (RFC-like simple)
    email = ""
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if m:
        email = m.group(0)

    # Phone (strict parse to avoid year-ranges)
    phone = ""
    pm = re.search(r"(?:(?:\+\d{1,3}[ \.-]?)?(?:\(?\d{1,4}\)?[ \.-]?)?\d{2,4}[ \.-]?\d{2,4}[ \.-]?\d{2,4})", text)
    if pm:
        norm = _parse_phone_strict(pm.group(0))
        if norm:
            phone = norm

    # LinkedIn URL
    linkedin = ""
    lm = re.search(r"https?://(?:www\.)?linkedin\.com/[^\s>\)]+", text, re.IGNORECASE)
    if lm:
        linkedin = lm.group(0)

    # Location heuristics: ville, region, pays (conservative)
    def _first_group(patterns: List[str]) -> str:
        for pat in patterns:
            mm = re.search(pat, text, re.IGNORECASE)
            if mm:
                return mm.group(1).strip().strip(",.;:| ")
        return ""

    ville = _first_group([r"(?:ville|city|localisation|location)\s*[:\-]\s*([A-Za-zÀ-ÖØ-öø-ÿ '\-]{2,40})"]) or ""
    region = _first_group([r"(?:région|region|state|province)\s*[:\-]\s*([A-Za-zÀ-ÖØ-öø-ÿ '\-]{2,40})"]) or ""
    pays = _first_group([r"(?:pays|country)\s*[:\-]\s*([A-Za-zÀ-ÖØ-öø-ÿ '\-]{2,40})"]) or ""

    if not pays:
        for tok in ["Canada", "France", "Belgique", "Suisse", "Québec"]:
            if re.search(rf"\b{re.escape(tok)}\b", text, re.IGNORECASE):
                pays = tok
                break
    if pays.lower() == "québec":
        region = region or "Québec"
        pays = "Canada"
    if (not pays or not region) and re.search(r"\bMontréal\b", text, re.IGNORECASE):
        region = region or "Québec"
        pays = pays or "Canada"

    return {
        "email": email,
        "telephone": phone,
        "linkedin": linkedin,
        "ville": ville,
        "region": region,
        "pays": pays,
    }

# Optional Pydantic AI imports with shims (as in archon_graph.py)
try:
    from pydantic_ai import Agent as PydanticAgent
except Exception:  # pragma: no cover
    class PydanticAgent:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        async def run(self, *args, **kwargs):
            return type("_Res", (), {"data": "{}"})

# LangGraph imports with fallbacks
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
except Exception as e:  # pragma: no cover
    logger.warning(f"Could not import LangGraph: {e}")
    class StateGraph:  # type: ignore
        def __init__(self, *args, **kwargs): ...
        def add_node(self, *a, **k): ...
        def add_edge(self, *a, **k): ...
        def set_entry_point(self, *a, **k): ...
        def compile(self, *a, **k):
            class _Dummy:
                async def ainvoke(self, state):
                    return state
            return _Dummy()
    END = None
    MemorySaver = object  # type: ignore


class CVState(TypedDict, total=False):
    # Inputs
    profil_poste_json: Any
    cv_path: str
    cv_text: str
    llm_overrides: Dict[str, Any]
    correlation_id: str
    artifacts_root: str

    # Intermediate
    candidate_base_json: Dict[str, Any]

    score_skills: Optional[float]
    evidence_skills: List[str]

    score_experience: Optional[float]
    evidence_experience: List[str]

    score_education: Optional[float]
    evidence_education: List[str]

    score_langues: Optional[float]
    evidence_langues: List[str]

    score_localisation: Optional[float]
    evidence_localisation: List[str]

    score_global: Optional[float]
    recommandation: str
    match_commentaire: str

    merged_for_sharepoint: Dict[str, Any]
    upsert_status: str

    messages: List[Dict[str, Any]]
    error: Optional[str]

    # Schema-related
    schema_candidats: Dict[str, Any]
    output_schema_path: str


def _get_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return (config.get("configurable", {}).get("llm_config") or {})


def _ensure_state_lists(state: CVState) -> None:
    state.setdefault("messages", [])
    state.setdefault("evidence_skills", [])
    state.setdefault("evidence_experience", [])
    state.setdefault("evidence_education", [])
    state.setdefault("evidence_langues", [])
    state.setdefault("evidence_localisation", [])


def _dump_json(state: CVState, name: str, data: Any) -> None:
    """Persist a JSON artifact for auditability under configured artifacts_root/name.json"""
    try:
        corr = state.get("correlation_id") or "cv-graph"
        out_dir = state.get("artifacts_root") or os.path.join("out", "tmp_ingest", str(corr))
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 wrote JSON artifact: {path}")
    except Exception as e:
        logger.debug(f"_dump_json failed for {name}: {e}")


def _dump_text(state: CVState, name: str, text: str) -> None:
    """Persist a text artifact for auditability under configured artifacts_root/name.txt"""
    try:
        corr = state.get("correlation_id") or "cv-graph"
        out_dir = state.get("artifacts_root") or os.path.join("out", "tmp_ingest", str(corr))
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")
        logger.info(f"💾 wrote TEXT artifact: {path}")
    except Exception as e:
        logger.debug(f"_dump_text failed for {name}: {e}")


def _extract_json_from_text(text: str) -> Optional[Any]:
    """Best-effort extraction of the first valid JSON object/array within a text blob.

    Tries direct json.loads first; if that fails, searches for JSON-like blocks and
    attempts to parse progressively larger candidates. Returns the parsed object or None.
    """
    if not text:
        return None
    # Fast path
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to locate JSON object/array in chatty outputs
    candidates: List[str] = []
    # Blocks fenced with ```json ... ```
    fence_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidates.append(fence_match.group(1))

    # First '{' ... last '}' heuristic
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace:last_brace + 1])

    # First '[' ... last ']' heuristic
    first_brack = text.find('[')
    last_brack = text.rfind(']')
    if first_brack != -1 and last_brack != -1 and last_brack > first_brack:
        candidates.append(text[first_brack:last_brack + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


def _ensure_artifacts_root(state: CVState, config: Dict[str, Any]) -> None:
    """Ensure state['artifacts_root'] is set from config.output_root + correlation_id.

    Fallback to out/tmp_ingest/<correlation_id> if not configured.
    """
    try:
        if state.get("artifacts_root"):
            return
        corr = state.get("correlation_id") or "cv-graph"
        cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
        out_root = cfg.get("output_root")
        if out_root:
            artifacts_root = os.path.join(str(out_root), str(corr))
        else:
            artifacts_root = os.path.join("out", "tmp_ingest", str(corr))
        os.makedirs(artifacts_root, exist_ok=True)
        state["artifacts_root"] = artifacts_root
    except Exception:
        pass


async def load_schema(state: CVState, config: Dict[str, Any]) -> CVState:
    """Coordinator: load authoritative candidates schema from Windows-mounted path
    and set output schema path for downstream validation.

    - Reads: /mnt/c/projects/rh_cv_vector/out/candidats_archon_schema.json
    - Sets: state['schema_candidats'] and state['output_schema_path']
    """
    try:
        schema_path = "/mnt/c/projects/rh_cv_vector/out/candidats_archon_schema.json"
        schema: Dict[str, Any] = {}
        if os.path.exists(schema_path):
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            logger.info(f"📥 Loaded authoritative schema: {schema_path}")
        else:
            logger.warning(f"Authoritative schema not found at {schema_path}; proceeding with empty schema")
        # Local output schema (validation contract produced/maintained in repo)
        out_schema_path = os.path.join("out", "candidats_cvgraph_output.schema.json")
        state.update({
            "schema_candidats": schema,
            "output_schema_path": out_schema_path,
        })
        state.setdefault("messages", []).append({"role": "node", "name": "load_schema", "status": "ok", "path": schema_path, "output_schema": out_schema_path})
        return state
    except Exception as e:
        state.setdefault("messages", []).append({"role": "node", "name": "load_schema", "status": "error", "error": str(e)})
        return state


async def extract_candidate_base(state: CVState, config: Dict[str, Any]) -> CVState:
    """Extract candidate base JSON from CV text/path using LLM prompt."""
    _ensure_state_lists(state)
    try:
        llm_cfg = _get_llm_config(config)
        # Initialize artifacts_root from config.output_root + correlation_id if available
        try:
            corr = state.get("correlation_id") or "cv-graph"
            cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
            out_root = cfg.get("output_root")
            if out_root:
                artifacts_root = os.path.join(str(out_root), str(corr))
                os.makedirs(artifacts_root, exist_ok=True)
                state["artifacts_root"] = artifacts_root
        except Exception:
            pass
        provider = llm_cfg.get("LLM_PROVIDER")
        model = llm_cfg.get("EXTRACTOR_MODEL")
        base_url = llm_cfg.get("BASE_URL")
        api_key_present = bool(llm_cfg.get("LLM_API_KEY"))

        # Prepare input text
        cv_text = state.get("cv_text")
        if not cv_text and state.get("cv_path") and os.path.exists(state["cv_path"]):
            try:
                # naive read; real implementation should parse DOCX/PDF upstream
                with open(state["cv_path"], "rb") as f:
                    content = f.read()
                cv_text = content.decode("utf-8", errors="ignore")
            except Exception:
                cv_text = ""

        if not cv_text:
            # Minimal placeholder to keep flow running
            cv_text = ""

        # Compute metadata extras early for later use
        cv_path = state.get("cv_path") or ""
        nom_fichier = os.path.basename(cv_path) if cv_path else ""
        url_fichier = str(Path(cv_path).resolve()) if cv_path else ""
        # Prefer file-bytes hash; fallback to text hash
        file_bytes = None
        if cv_path and os.path.exists(cv_path):
            try:
                with open(cv_path, "rb") as f:
                    file_bytes = f.read()
            except Exception:
                file_bytes = None
        if file_bytes:
            resume_hash = hashlib.sha256(file_bytes).hexdigest()
        else:
            resume_hash = hashlib.sha256(cv_text.encode("utf-8", errors="ignore")).hexdigest() if cv_text else ""

        # Build prompt with template inputs expected by the centralized prompt
        prompt = (
            cv_extract_prompt
            .replace("{{profil_poste_json}}", json.dumps(state.get("profil_poste_json", {}), ensure_ascii=False))
            .replace("{{cv_texte}}", cv_text[:6000])
        )
        _dump_text(state, "prompt_extract", prompt)
        logger.info(
            f"🧩 [extract] provider={provider} model={model} base_url={base_url} api_key_present={api_key_present}"
        )

        # Enforce strict JSON-only output
        agent = PydanticAgent(
            await get_llm_instance(provider, model, llm_cfg),
            system_prompt=prompt,
        )
        timeout_s = float(_get_llm_config(config).get("TIMEOUT_S", 30))
        logger.info(
            f"🧩 [extract] timeout_s={timeout_s} corr={state.get('correlation_id')}"
        )
        try:
            res = await _with_retries(lambda: asyncio.wait_for(agent.run(""), timeout=timeout_s))
        except Exception as e:
            # Optional fallback to Ollama only if explicitly allowed
            if bool(llm_cfg.get("ALLOW_OLLAMA_FALLBACK", False)):
                fb_model = _fallback_model(llm_cfg)
                fb = PydanticAgent(await get_llm_instance("ollama", fb_model, llm_cfg), system_prompt=prompt)
                try:
                    res = await _with_retries(lambda: asyncio.wait_for(fb.run(""), timeout=timeout_s))
                except Exception:
                    res = None
            else:
                res = None
            if res is None:
                # On failure/timeout, return empty structure but keep flow alive
                candidate = {
                    "nom_complet": "", "nom_fichier": "", "resume_hash": "", "correlation_id": state.get("correlation_id", ""),
                    "linkedin_url": "", "ville": "", "region_text": "", "pays": "", "pret_relocaliser": False,
                    "disponibilite_date": None, "annees_exp_totale": None, "annees_exp_poste": None,
                    "competences": {"tech": [], "soft": [], "mots_cles": []},
                    "diplomes": [], "certifications": [],
                    "langues": [], "notes": "", "url_fichier": ""
                }
                _dump_json(state, "extract", candidate)
                msgs = state.get("messages", []) + [{"role": "node", "name": "extract", "status": "timeout"}]
                return {"candidate_base_json": candidate, "messages": msgs, "artifacts_root": state.get("artifacts_root")}

        out_txt = res.data if hasattr(res, "data") else str(res)
        _dump_text(state, "raw_extract", out_txt)
        parsed = _extract_json_from_text(out_txt)
        if isinstance(parsed, dict):
            candidate = parsed
        else:
            # Fallback empty structure using detailed schema keys
            candidate = {
                "nom_complet": "",
                "ville": "",
                "pays": "",
                "region_text": "",
                "annees_exp_totale": None,
                "annees_exp_poste": None,
                "competences": {"tech": [], "soft": [], "mots_cles": []},
                "langues": [],
                "email": "",
                "telephone": "",
                "linkedin": "",
            }
        # Deterministic coordinates micro-agent (artifact + merge if missing)
        coords = _coordinates_micro_agent(cv_text)
        _dump_json(state, "coordinates", coords)
        for k in ["email", "telephone", "linkedin", "ville", "region", "pays"]:
            if not candidate.get(k):
                candidate[k] = coords.get(k, candidate.get(k, ""))
        # Heuristic enrichment from raw text to ensure baseline CSV completeness
        candidate = _enrich_candidate_from_text(cv_text, candidate)
        # Mirror region -> region_text if extractor didn't provide region_text
        if not candidate.get("region_text") and candidate.get("region"):
            candidate["region_text"] = candidate.get("region", "")

        # Inject computed metadata if missing
        candidate.setdefault("resume_hash", resume_hash)
        candidate.setdefault("nom_fichier", nom_fichier)
        candidate.setdefault("url_fichier", url_fichier)
        _dump_json(state, "extract", candidate)
        msgs = state.get("messages", []) + [{"role": "node", "name": "extract", "status": "ok"}]
        return {"candidate_base_json": candidate, "messages": msgs, "artifacts_root": state.get("artifacts_root")}
    except Exception as e:  # keep flow alive
        state["candidate_base_json"] = {
            "Nom": "", "Prenom": "", "Email": "", "Telephone": "",
            "Ville": "", "Pays": "", "Region": "",
            "ExperienceAnnees": None, "Formation": [], "Competences": [], "Langues": []
        }
        state.setdefault("error", str(e))
        state["messages"].append({"role": "node", "name": "extract", "status": "error", "error": str(e)})
        return state


async def _score_dimension(state: CVState, config: Dict[str, Any], prompt_text: str,
                           score_key: str, evid_key: str, model_key: str = "SCORER_MODEL") -> CVState:
    _ensure_state_lists(state)
    try:
        _ensure_artifacts_root(state, config)
        llm_cfg = _get_llm_config(config)
        provider = llm_cfg.get("LLM_PROVIDER")
        model = llm_cfg.get(model_key) or llm_cfg.get("SCORER_MODEL")
        base_url = llm_cfg.get("BASE_URL")
        api_key_present = bool(llm_cfg.get("LLM_API_KEY"))

        # Render structured prompt with placeholders
        prompt = (
            prompt_text
            .replace("{{profil_poste_json}}", json.dumps(state.get("profil_poste_json", {}), ensure_ascii=False))
            .replace("{{candidate_base_json}}", json.dumps(state.get("candidate_base_json", {}), ensure_ascii=False))
        )
        _dump_text(state, f"prompt_{score_key}", prompt)
        logger.info(
            f"🧩 [{score_key}] provider={provider} model={model} base_url={base_url} api_key_present={api_key_present}"
        )

        agent = PydanticAgent(
            await get_llm_instance(provider, model, llm_cfg),
            system_prompt=prompt,
        )
        timeout_s = float(llm_cfg.get("TIMEOUT_S", 30))
        logger.info(
            f"🧩 [{score_key}] timeout_s={timeout_s} corr={state.get('correlation_id')}"
        )
        sem = _get_scorer_semaphore(llm_cfg)
        await sem.acquire()
        try:
            try:
                res = await _with_retries(lambda: asyncio.wait_for(agent.run(""), timeout=timeout_s))
            except Exception:
                res = None
                if bool(llm_cfg.get("ALLOW_OLLAMA_FALLBACK", False)):
                    fb_model = _fallback_model(llm_cfg)
                    fb = PydanticAgent(await get_llm_instance("ollama", fb_model, llm_cfg), system_prompt=prompt)
                    try:
                        res = await _with_retries(lambda: asyncio.wait_for(fb.run(""), timeout=timeout_s))
                    except Exception:
                        res = None
                if res is None:
                    state[score_key] = 0
                    state[evid_key] = ["timeout or error"]
                    _dump_json(state, score_key, {"score": None, "evidence": state[evid_key]})
                    return {score_key: state[score_key], evid_key: state[evid_key]}
        finally:
            sem.release()

        out_txt = res.data if hasattr(res, "data") else str(res)
        _dump_text(state, f"raw_{score_key}", out_txt)
        obj = _extract_json_from_text(out_txt)
        if isinstance(obj, dict):
            # Prefer dimension-specific keys (e.g., score_skills), fallback to generic
            score = obj.get(score_key)
            if score is None:
                score = obj.get("score")
            evidence = obj.get(evid_key)
            if evidence is None:
                evidence = obj.get("evidence") or []
            if isinstance(score, (int, float)):
                state[score_key] = float(score)
            else:
                state[score_key] = None
            if isinstance(evidence, list):
                state[evid_key] = [str(x) for x in evidence][:5]
            else:
                state[evid_key] = []
        else:
            # parsing failed → numeric fallback: first 0–100 integer in text
            fallback_score = None
            try:
                m = re.search(r"\b(100|[0-9]{1,2})\b", out_txt)
                if m:
                    fallback_score = int(m.group(1))
            except Exception:
                fallback_score = None
            state[score_key] = fallback_score if fallback_score is not None else 0
            state[evid_key] = ["fallback: numeric extraction" if fallback_score is not None else "invalid or non-JSON response"]
            _dump_json(state, score_key, {"score": state[score_key], "evidence": state[evid_key]})
            return {score_key: state[score_key], evid_key: state[evid_key]}

        # Persist node output
        _dump_json(state, score_key, {"score": state.get(score_key), "evidence": state.get(evid_key, [])})
        return {score_key: state.get(score_key), evid_key: state.get(evid_key, [])}
    except Exception as e:
        # Ensure we still create artifacts for observability
        state[score_key] = 0
        state[evid_key] = []
        _dump_text(state, f"raw_{score_key}_error", f"{type(e).__name__}: {e}")
        _dump_json(state, score_key, {"score": state[score_key], "evidence": []})
        return {score_key: state[score_key], evid_key: []}


async def score_skills(state: CVState, config: Dict[str, Any]) -> CVState:
    return await _score_dimension(state, config, cv_skills_prompt, "score_skills", "evidence_skills")


async def score_experience(state: CVState, config: Dict[str, Any]) -> CVState:
    return await _score_dimension(state, config, cv_experience_prompt, "score_experience", "evidence_experience")


async def score_education(state: CVState, config: Dict[str, Any]) -> CVState:
    return await _score_dimension(state, config, cv_education_prompt, "score_education", "evidence_education")


async def score_langues(state: CVState, config: Dict[str, Any]) -> CVState:
    llm_cfg = _get_llm_config(config)
    if not llm_cfg.get("ENABLE_LANGUES", True):
        state["score_langues"] = None
        state["evidence_langues"] = []
        return state
    return await _score_dimension(state, config, cv_languages_prompt, "score_langues", "evidence_langues")


async def score_localisation(state: CVState, config: Dict[str, Any]) -> CVState:
    llm_cfg = _get_llm_config(config)
    if not llm_cfg.get("ENABLE_LOCALISATION", True):
        state["score_localisation"] = None
        state["evidence_localisation"] = []
        return state
    return await _score_dimension(state, config, cv_location_prompt, "score_localisation", "evidence_localisation")


async def run_all_scorers(state: CVState, config: Dict[str, Any]) -> CVState:
    """Run all scorer nodes and merge their outputs into state before aggregation.

    This ensures `aggregate_scores` executes after all score_* fields are set,
    avoiding partial aggregation when the graph would otherwise fan-in per edge.
    """
    llm_cfg = _get_llm_config(config)

    # Launch scorers; use shallow copies to avoid mutation races between tasks.
    tasks: List[asyncio.Future] = []
    tasks.append(asyncio.create_task(score_skills(dict(state), config)))
    tasks.append(asyncio.create_task(score_experience(dict(state), config)))
    tasks.append(asyncio.create_task(score_education(dict(state), config)))
    if llm_cfg.get("ENABLE_LANGUES", True):
        tasks.append(asyncio.create_task(score_langues(dict(state), config)))
    if llm_cfg.get("ENABLE_LOCALISATION", True):
        tasks.append(asyncio.create_task(score_localisation(dict(state), config)))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    updates: Dict[str, Any] = {}
    for r in results:
        if isinstance(r, dict):
            updates.update(r)
    # Return only updates so the graph merges them cleanly
    return updates


def _safe_weight(v: Any) -> float:
    try:
        x = float(v)
        return max(0.0, x)
    except Exception:
        return 0.0


def _renormalize(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(weights.values())
    if s <= 0:
        return {k: 0.0 for k in weights}
    return {k: (v / s) for k, v in weights.items()}


async def aggregate_scores(state: CVState, config: Dict[str, Any]) -> CVState:
    """Weighted aggregation with null-handling and basic recommendation heuristics."""
    _ensure_artifacts_root(state, config)
    llm_cfg = _get_llm_config(config)
    w_cfg = llm_cfg.get("WEIGHTS", {}) or {}
    # default weights
    w = {
        "skills": _safe_weight(w_cfg.get("skills", 0.35)),
        "experience": _safe_weight(w_cfg.get("experience", 0.30)),
        "education": _safe_weight(w_cfg.get("education", 0.20)),
        "langues": _safe_weight(w_cfg.get("langues", 0.10)),
        "localisation": _safe_weight(w_cfg.get("localisation", 0.05)),
    }

    # Backfill scores from artifacts if missing (resiliency for async joins)
    try:
        corr = state.get("correlation_id") or "cv-graph"
        cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
        out_root = cfg.get("output_root")
        base_dir = state.get("artifacts_root") or (os.path.join(str(out_root), str(corr)) if out_root else os.path.join("out", "tmp_ingest", str(corr)))
        for key in ["skills", "experience", "education", "langues", "localisation"]:
            sk = f"score_{key}"
            if state.get(sk) is None:
                p = os.path.join(base_dir, f"{sk}.json")
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    v = obj.get("score")
                    if isinstance(v, (int, float)):
                        state[sk] = float(v)
    except Exception:
        pass

    # Disable weights if scorer disabled or score is None
    available: Dict[str, float] = {}
    for key, weight in w.items():
        score_key = f"score_{key}"
        score_val = state.get(score_key)
        enabled = True
        if key == "langues" and not llm_cfg.get("ENABLE_LANGUES", True):
            enabled = False
        if key == "localisation" and not llm_cfg.get("ENABLE_LOCALISATION", True):
            enabled = False
        if enabled and (isinstance(score_val, (int, float))):
            available[key] = weight

    weights = _renormalize(available)

    total = 0.0
    for key, wv in weights.items():
        sv = float(state.get(f"score_{key}") or 0.0)
        total += wv * sv

    state["score_global"] = round(total, 2) if total > 0 else None

    # Simple recommendation
    sg = state["score_global"] or 0.0
    if sg >= 75:
        state["recommandation"] = "Go"
    elif sg >= 55:
        state["recommandation"] = "Consider"
    else:
        state["recommandation"] = "No"

    state["match_commentaire"] = (
        f"Global={state['score_global']} (w={weights})"
    )

    # Build SharePoint merged dict (respect missing-value policy)
    base = state.get("candidate_base_json", {})
    nom_complet = str(base.get("nom_complet", "") or "").strip()
    prenom, nom = "", ""
    if nom_complet:
        parts = nom_complet.split()
        if len(parts) >= 2:
            prenom = parts[0]
            nom = " ".join(parts[1:])
        else:
            nom = nom_complet
    merged = {
        # identity (mapped from detailed schema)
        "Nom": nom,
        "Prenom": prenom,
        # Email/Telephone will be injected by canonical adapter; avoid empty overrides here
        "Ville": base.get("ville", ""),
        "Pays": base.get("pays", ""),
        "Region": base.get("region_text", ""),  # Region as Text
        "ExperienceAnnees": base.get("annees_exp_totale", None),
        # keys for upsert uniqueness if present
        "NomFichier": base.get("nom_fichier", ""),
        "ResumeHash": base.get("resume_hash", ""),
        # scores
        "ScoreSkills": state.get("score_skills"),
        "ScoreExperience": state.get("score_experience"),
        "ScoreEducation": state.get("score_education"),
        "ScoreLangues": state.get("score_langues"),
        "ScoreLocalisation": state.get("score_localisation"),
        "ScoreGlobal": state.get("score_global"),
        # decision
        "Recommandation": state.get("recommandation", ""),
        "MatchCommentaire": state.get("match_commentaire", ""),
    }
    # Prepend canonical SharePoint payload (single source of truth for keys)
    try:
        schema = state.get("schema_candidats") or load_candidats_schema()
        # Build extras with robust fallbacks
        cand = state.get("candidate_base_json") or {}
        cvp = state.get("cv_path") or ""
        # Derive robust ResumeHash: prefer candidate value, else file bytes, else text
        resume_hash_extra = cand.get("resume_hash") or ""
        if not resume_hash_extra:
            try:
                if cvp and os.path.exists(cvp):
                    with open(cvp, "rb") as f:
                        resume_hash_extra = hashlib.sha256(f.read()).hexdigest()
                elif state.get("cv_text"):
                    resume_hash_extra = hashlib.sha256((state.get("cv_text") or "").encode("utf-8", errors="ignore")).hexdigest()
            except Exception:
                resume_hash_extra = resume_hash_extra or ""
        extras = {
            "Title": cand.get("nom_complet") or "",
            "NomFichier": (os.path.basename(cvp) if cvp else "") or cand.get("nom_fichier") or "",
            "ResumeHash": resume_hash_extra,
            "UrlFichier": cand.get("url_fichier") or (str(Path(cvp).resolve()) if cvp else ""),
        }
        canonical = build_canonical_payload(state.get("candidate_base_json") or {}, extras=extras, schema=schema)
        merged = {**canonical, **(merged or {})}
    except Exception:
        # On any adapter error, keep existing merged structure
        pass

    # Optional LLM aggregator enrichment to fill remaining canonical fields
    try:
        agg_model = (llm_cfg or {}).get("AGGREGATOR_MODEL")
        if agg_model:
            provider = llm_cfg.get("LLM_PROVIDER")
            base_url = llm_cfg.get("BASE_URL")
            api_key_present = bool(llm_cfg.get("LLM_API_KEY"))
            # Build strict prompt instructing JSON-only output with allowed schema keys
            prompt = (
                "Tu es un agrégateur strict. À partir des données suivantes (JSON), propose des valeurs pour les champs manquants du schéma candidat. "
                "Réponds UNIQUEMENT en JSON avec des paires clé/valeur, sans texte autour. "
                "N'inclus que des clés pertinentes du schéma. Évite les hallucinations; laisse vide si inconnu.\n\n"
                f"SchemaKeys: {list((schema or {}).keys())}\n"
                f"BaseCandidate: {json.dumps(state.get('candidate_base_json') or {}, ensure_ascii=False)}\n"
                f"Scores: {json.dumps({k: state.get(k) for k in ['score_skills','score_experience','score_education','score_langues','score_localisation','score_global']}, ensure_ascii=False)}\n"
                f"CurrentMerged: {json.dumps(merged, ensure_ascii=False)}\n"
            )
            _dump_text(state, "prompt_llm_aggregate", prompt)
            agent = PydanticAgent(await get_llm_instance(provider, agg_model, llm_cfg), system_prompt=prompt)
            timeout_s = float(llm_cfg.get("TIMEOUT_S", 30))
            try:
                res = await _with_retries(lambda: asyncio.wait_for(agent.run(""), timeout=timeout_s))
                out_txt = res.data if hasattr(res, "data") else str(res)
                _dump_text(state, "raw_llm_aggregate", out_txt)
                obj = _extract_json_from_text(out_txt)
                if isinstance(obj, dict):
                    allowed = set((schema or {}).keys())
                    enriched = {k: v for k, v in obj.items() if k in allowed}
                    _dump_json(state, "llm_aggregate", enriched)
                    # Only update with non-empty values to avoid erasing deterministic fields
                    merged.update({k: v for k, v in enriched.items() if (v is not None and v != "")})
            except Exception as _e:
                _dump_text(state, "raw_llm_aggregate_error", f"{type(_e).__name__}: {_e}")
    except Exception:
        pass

    state["merged_for_sharepoint"] = merged
    _dump_json(state, "aggregate", {
        "weights": weights,
        "score_global": state["score_global"],
        "recommandation": state["recommandation"],
        "match_commentaire": state["match_commentaire"],
        "merged_for_sharepoint": merged,
    })
    logger.info(
        f"🧩 [aggregate] corr={state.get('correlation_id')} score_global={state.get('score_global')} reco={state.get('recommandation')}"
    )
    state.setdefault("messages", []).append({"role": "node", "name": "aggregate", "status": "ok"})

    # Emit Mermaid graph artifact (.mmd) for visualization
    try:
        def _fmt(v: Any) -> str:
            try:
                return str(v)
            except Exception:
                return ""
        mmd_lines = [
            "flowchart TD",
            "  A[extract_candidate_base] --> J(run_all_scorers)",
            "  J --> S1[score_skills]",
            "  J --> S2[score_experience]",
            "  J --> S3[score_education]",
            "  J --> S4[score_langues]",
            "  J --> S5[score_localisation]",
            "  S1 --> AGG[aggregate_scores]",
            "  S2 --> AGG",
            "  S3 --> AGG",
            "  S4 --> AGG",
            "  S5 --> AGG",
            f"  AGG -->|ScoreGlobal={_fmt(state.get('score_global'))}; Reco={_fmt(state.get('recommandation'))}| V[validate_merged]",
            "  V --> U[upsert_sharepoint]",
        ]
        _dump_text(state, "graph.mmd", "\n".join(mmd_lines))
    except Exception:
        pass
    return state


async def upsert_sharepoint(state: CVState, config: Dict[str, Any]) -> CVState:
    """Optionally upsert to SharePoint via external PnP script.

    Behavior:
    - If state['perform_upsert'] is truthy, write merged JSON to out/tmp_ingest/<id>/ and call pwsh script.
    - Otherwise, no-op with status="skipped".
    """
    _ensure_artifacts_root(state, config)
    perform = bool(state.get("perform_upsert"))
    corr = state.get("correlation_id") or "cv-graph"
    cfg = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    out_root = cfg.get("output_root")
    out_dir = state.get("artifacts_root") or (os.path.join(str(out_root), str(corr)) if out_root else os.path.join("out", "tmp_ingest", str(corr)))
    os.makedirs(out_dir, exist_ok=True)
    merged_path = os.path.join(out_dir, "merged_for_sharepoint.json")
    try:
        with open(merged_path, "w", encoding="utf-8") as f:
            json.dump(state.get("merged_for_sharepoint", {}), f, ensure_ascii=False, indent=2)
    except Exception as e:
        state.setdefault("error", str(e))

    if not perform:
        state["upsert_status"] = f"skipped:{merged_path}"
        state.setdefault("messages", []).append({"role": "node", "name": "upsert", "status": "skipped", "path": merged_path})
        return state

    # Execute PowerShell script if requested
    script_path = os.path.join("scripts", "upsert_candidates_pnp.ps1")
    if not os.path.exists(script_path):
        state["upsert_status"] = f"error:no_script:{script_path}"
        state.setdefault("messages", []).append({"role": "node", "name": "upsert", "status": "error", "error": "script not found"})
        return state

    try:
        import subprocess
        cmd = [
            "pwsh", "-NoLogo", "-NoProfile", "-File", script_path,
            "-InputPath", out_dir,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        ok = (res.returncode == 0)
        state["upsert_status"] = "ok" if ok else f"error:{res.returncode}"
        state.setdefault("messages", []).append({
            "role": "node", "name": "upsert",
            "status": "ok" if ok else "error",
            "stdout": res.stdout[-1000:],
            "stderr": res.stderr[-1000:],
        })
    except Exception as e:
        state["upsert_status"] = f"error:{e}"
        state.setdefault("messages", []).append({"role": "node", "name": "upsert", "status": "error", "error": str(e)})
    return state


async def validate_merged(state: CVState, config: Dict[str, Any]) -> CVState:
    """Validate merged_for_sharepoint against JSON Schema before upsert.

    Behavior:
    - Load canonical SharePoint schema from RH CV Vector mount via load_candidats_schema().
    - Filter out any extra keys not allowed by the canonical schema.
    - On validation error, set upsert to skipped and record error.
    - Writes validated_merged.json artifact.
    """
    _ensure_artifacts_root(state, config)
    merged = state.get("merged_for_sharepoint") or {}
    errors: List[str] = []

    # Load canonical SP schema (simple dict of fields -> metadata)
    try:
        from k.cv_schema_adapter import load_candidats_schema
        canon_schema = load_candidats_schema()
    except Exception as e:
        canon_schema = {}
        errors.append(f"schema_load_error:{e}")

    allowed_props: List[str] = list(canon_schema.keys()) if isinstance(canon_schema, dict) else []
    if allowed_props:
        merged = {k: v for k, v in merged.items() if (k in allowed_props or k == "Title")}

    # Simple type checks based on canonical schema's 'type' hints
    type_map = {
        "string": str,
        "number": (int, float),
        "boolean": bool,
    }
    for key, spec in (canon_schema or {}).items():
        if key not in merged:
            continue
        expected_t = spec.get("type") if isinstance(spec, dict) else None
        py_t = type_map.get(expected_t)
        if py_t and not isinstance(merged.get(key), py_t) and merged.get(key) is not None:
            errors.append(f"type:{key}:expected_{expected_t}")

    # Persist validated payload
    state["merged_for_sharepoint"] = merged
    _dump_json(state, "validated_merged", {"payload": merged, "errors": errors, "schema_source": "canonical"})

    if errors:
        state.setdefault("messages", []).append({"role": "node", "name": "validate", "status": "error", "errors": errors})
        # Block upsert by default on error
        state["perform_upsert"] = False
    else:
        state.setdefault("messages", []).append({"role": "node", "name": "validate", "status": "ok"})
    return state


_cv_flow = None

def get_cv_agentic_flow():
    global _cv_flow
    if _cv_flow is None:
        # Build StateGraph similar to archon_graph
        builder = StateGraph(CVState)
        builder.add_node("load_schema", load_schema)
        builder.add_node("extract_candidate_base", extract_candidate_base)
        # Keep individual scorer nodes available (for potential reuse),
        # but route through a join node to ensure all scores are computed before aggregation.
        builder.add_node("score_skills", score_skills)
        builder.add_node("score_experience", score_experience)
        builder.add_node("score_education", score_education)
        builder.add_node("score_langues", score_langues)
        builder.add_node("score_localisation", score_localisation)
        builder.add_node("run_all_scorers", run_all_scorers)
        builder.add_node("aggregate_scores", aggregate_scores)
        builder.add_node("validate_merged", validate_merged)
        builder.add_node("upsert_sharepoint", upsert_sharepoint)

        builder.set_entry_point("load_schema")

        # Route through join node that ensures all scorers have run
        builder.add_edge("load_schema", "extract_candidate_base")
        builder.add_edge("extract_candidate_base", "run_all_scorers")
        builder.add_edge("run_all_scorers", "aggregate_scores")

        # Aggregate → validate → upsert → END
        builder.add_edge("aggregate_scores", "validate_merged")
        builder.add_edge("validate_merged", "upsert_sharepoint")
        builder.add_edge("upsert_sharepoint", END)

        memory = MemorySaver()
        _cv_flow = builder.compile(checkpointer=memory)
        logger.info("✅ CV agentic flow initialized")
    return _cv_flow


async def run_cv_workflow(initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if initial_state is None:
        initial_state = {
            "profil_poste_json": {},
            "cv_text": "",
            "correlation_id": "cv-graph",
            "messages": [],
        }
    flow = get_cv_agentic_flow()
    # Provide required configurable keys for the checkpointer (thread_id)
    corr = initial_state.get("correlation_id") or "cv-graph"

    # Build strict llm_config from active profile and merge optional overrides
    try:
        overrides = (initial_state or {}).get("llm_overrides") or {}
    except Exception:
        overrides = {}
    llm_config = build_llm_config_from_active_profile(overrides if isinstance(overrides, dict) else None)

    logger.info(
        f"🔧 CV-Graph provider='{llm_config.get('LLM_PROVIDER')}' "
        f"extractor='{llm_config.get('EXTRACTOR_MODEL')}' scorer='{llm_config.get('SCORER_MODEL')}'"
    )
    try:
        logger.info(
            f"🔧 CV-Graph config: base_url='{llm_config.get('BASE_URL')}' api_key_present={bool(llm_config.get('LLM_API_KEY'))}"
        )
    except Exception:
        pass

    config = {"configurable": {"thread_id": str(corr), "llm_config": llm_config}}
    result = await flow.ainvoke(initial_state, config)  # type: ignore
    return result

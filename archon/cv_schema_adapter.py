# -*- coding: utf-8 -*-
"""
Adapter to map Archon CV Graph artifacts to the canonical SharePoint
"Candidats" schema and to construct a candidate list row for CSV export.

- Canonical schema path (mounted from RH CV Vector):
  /mnt/c/projects/rh_cv_vector/out/candidats_archon_schema.json

- Minimal ingestion payload keys (see rh_cv_vector/docs/schema_liste_candidats.md):
  Title, NomFichier, ResumeHash, UrlFichier, NomComplet, Email, Telephone,
  LinkedInUrl, Ville, Region, Pays

This module is intentionally side-effect free; CSV writing is implemented in
scripts that call these helpers.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA_PATH_DEFAULT = "/mnt/c/projects/rh_cv_vector/out/candidats_archon_schema.json"


def load_candidats_schema(path: str = SCHEMA_PATH_DEFAULT) -> Dict[str, Any]:
    """Load the canonical SharePoint candidates schema (if present)."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _iso_datetime(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def _first_str(values: Any) -> str:
    """Return the first non-empty string from a scalar or list of strings."""
    if values is None:
        return ""
    if isinstance(values, str):
        return values.strip()
    if isinstance(values, list):
        for v in values:
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _to_bool_optional(val: Any) -> Optional[bool]:
    """Convert common truthy/falsey strings/bools to bool or None."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"true", "vrai", "oui", "yes", "1"}:
            return True
        if s in {"false", "faux", "non", "no", "0"}:
            return False
    return None


def build_canonical_payload(
    candidate_base_json: Dict[str, Any],
    extras: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the strict minimal ingestion payload using canonical keys.

    Inputs typically come from `extract.json` and runtime context.
    - Title: fallback to NomComplet; if empty, fallback to NomFichier (sans extension).
    - Dates: ISO 8601.
    - Unknown enums: omitted; final validation happens elsewhere.
    """
    extras = extras or {}
    schema = schema or {}

    # Base fields from extraction
    nom_complet = _first_str(candidate_base_json.get("nom_complet") or candidate_base_json.get("name"))
    email = _first_str(candidate_base_json.get("email"))
    telephone = _first_str(candidate_base_json.get("telephone") or candidate_base_json.get("phone"))
    linkedin = _first_str(candidate_base_json.get("linkedin") or candidate_base_json.get("linkedin_url"))
    ville = _first_str(candidate_base_json.get("ville") or candidate_base_json.get("city"))
    # region variations encountered in extraction
    region = _first_str(
        candidate_base_json.get("region")
        or candidate_base_json.get("region_text")
        or candidate_base_json.get("state_province")
    )
    pays = _first_str(candidate_base_json.get("pays") or candidate_base_json.get("country"))

    # Extras from runtime
    nom_fichier = _first_str(extras.get("NomFichier") or candidate_base_json.get("nom_fichier"))
    resume_hash = _first_str(extras.get("ResumeHash") or candidate_base_json.get("resume_hash"))
    url_fichier = _first_str(extras.get("UrlFichier") or candidate_base_json.get("url_fichier"))

    title = _first_str(extras.get("Title")) or nom_complet or os.path.splitext(nom_fichier or "")[0]

    payload: Dict[str, Any] = {
        "Title": title,
        "NomFichier": nom_fichier,
        "ResumeHash": resume_hash,
        "UrlFichier": url_fichier,
        "NomComplet": nom_complet,
        "Email": email,
        "Telephone": telephone,
        "LinkedInUrl": linkedin,
        "Ville": ville,
        "Region": region,
        "Pays": pays,
    }

    # Optional fields when present in extraction
    # Poste / domaine / seniorité / niveau d'études
    poste_cible = _first_str(
        candidate_base_json.get("poste_cible")
        or candidate_base_json.get("poste")
        or candidate_base_json.get("job_title")
    )
    if poste_cible:
        payload["PosteCible"] = poste_cible

    domaine = _first_str(
        candidate_base_json.get("domaine")
        or candidate_base_json.get("secteur")
        or candidate_base_json.get("domain")
        or candidate_base_json.get("industry")
    )
    if domaine:
        payload["Domaine"] = domaine

    seniorite = _first_str(
        candidate_base_json.get("seniorite")
        or candidate_base_json.get("seniority")
    )
    if seniorite:
        payload["Seniorite"] = seniorite

    niveau_etudes = _first_str(
        candidate_base_json.get("niveau_etudes")
        or candidate_base_json.get("education_level")
    )
    if niveau_etudes:
        payload["NiveauEtudes"] = niveau_etudes

    # Disponibilité / relocalisation / autorisation travail
    dispo_date = candidate_base_json.get("disponibilite_date") or candidate_base_json.get("dispo_date")
    if isinstance(dispo_date, datetime):
        dispo_iso = _iso_datetime(dispo_date)
    else:
        dispo_iso = _first_str(dispo_date)
    if dispo_iso:
        payload["DispoDate"] = dispo_iso

    pret_relocaliser = _to_bool_optional(
        candidate_base_json.get("pret_relocaliser") or candidate_base_json.get("relocation")
    )
    if pret_relocaliser is not None:
        payload["PretRelocaliser"] = pret_relocaliser

    autorisation_travail = _first_str(
        candidate_base_json.get("autorisation_travail")
        or candidate_base_json.get("work_authorization")
        or candidate_base_json.get("work_permit")
    )
    if autorisation_travail:
        payload["AutorisationTravail"] = autorisation_travail

    # Additional canonical fields when present in extraction or extras
    # Notes / Statut
    notes = _first_str(candidate_base_json.get("notes") or candidate_base_json.get("appreciation") or extras.get("Notes"))
    if notes:
        payload["Notes"] = notes
    statut = _first_str(candidate_base_json.get("statut") or extras.get("Statut"))
    if statut:
        payload["Statut"] = statut

    # Langues (principal level + details)
    # Expect formats like: { lang: "français", level: "C1" } or strings; fallback to empties
    langues = candidate_base_json.get("langues")
    langue_principale = ""
    niveau_fr = ""
    niveau_en = ""
    autres_langues: List[str] = []
    if isinstance(langues, list):
        def _norm_lang(s: str) -> str:
            return (s or "").strip().lower()
        for i, item in enumerate(langues):
            if isinstance(item, dict):
                l = _first_str(item.get("lang") or item.get("language") or item.get("name"))
                lvl = _first_str(item.get("level") or item.get("niveau") or item.get("proficiency"))
                ln = _norm_lang(l)
                if i == 0 and l and not langue_principale:
                    langue_principale = l
                if ln in {"fr", "français", "francais", "french"} and lvl:
                    niveau_fr = lvl
                elif ln in {"en", "anglais", "english"} and lvl:
                    niveau_en = lvl
                elif l:
                    autres_langues.append(f"{l}{' '+lvl if lvl else ''}".strip())
            elif isinstance(item, str) and item.strip():
                if not langue_principale:
                    langue_principale = item.strip()
                else:
                    autres_langues.append(item.strip())
    if langue_principale:
        payload["LanguePrincipale"] = langue_principale
    if niveau_fr:
        payload["NiveauFrancais"] = niveau_fr
    if niveau_en:
        payload["NiveauAnglais"] = niveau_en
    if autres_langues:
        payload["AutresLangues"] = ", ".join(autres_langues[:20])

    # Rémunération
    salaire_actuel = candidate_base_json.get("salaire_actuel")
    if isinstance(salaire_actuel, (int, float)):
        payload["SalaireActuel"] = float(salaire_actuel)
    salaire_souhaite = candidate_base_json.get("salaire_souhaite")
    if isinstance(salaire_souhaite, (int, float)):
        payload["SalaireSouhaite"] = float(salaire_souhaite)

    # Contact et suivi
    dernier_contact = candidate_base_json.get("dernier_contact") or extras.get("DernierContact")
    if isinstance(dernier_contact, datetime):
        payload["DernierContact"] = _iso_datetime(dernier_contact)
    else:
        dc = _first_str(dernier_contact)
        if dc:
            payload["DernierContact"] = dc
    prochaine_action = _first_str(candidate_base_json.get("prochaine_action") or extras.get("ProchaineAction"))
    if prochaine_action:
        payload["ProchaineAction"] = prochaine_action
    dpa = candidate_base_json.get("date_prochaine_action") or extras.get("DateProchaineAction")
    if isinstance(dpa, datetime):
        payload["DateProchaineAction"] = _iso_datetime(dpa)
    else:
        dpa_s = _first_str(dpa)
        if dpa_s:
            payload["DateProchaineAction"] = dpa_s

    # Identifiants et métadonnées document
    embedding_id = _first_str(extras.get("EmbeddingId") or candidate_base_json.get("embedding_id"))
    if embedding_id:
        payload["EmbeddingId"] = embedding_id
    document_id = _first_str(extras.get("DocumentId") or candidate_base_json.get("document_id"))
    if document_id:
        payload["DocumentId"] = document_id
    page_spec = _first_str(extras.get("PageSpec") or candidate_base_json.get("page_spec"))
    if page_spec:
        payload["PageSpec"] = page_spec

    # Appréciation globale / Compétences (en Note texte)
    appreciation = _first_str(candidate_base_json.get("appreciation_globale") or candidate_base_json.get("notes_globale") or extras.get("AppreciationGlobale"))
    if appreciation:
        payload["AppreciationGlobale"] = appreciation
    competences = candidate_base_json.get("competences")
    if isinstance(competences, dict):
        tech = competences.get("tech") if isinstance(competences.get("tech"), list) else []
        soft = competences.get("soft") if isinstance(competences.get("soft"), list) else []
        mots = competences.get("mots_cles") if isinstance(competences.get("mots_cles"), list) else []
        comp_txt = ", ".join([*tech[:50], *soft[:20], *mots[:50]])
        if comp_txt:
            payload["Competences"] = comp_txt
    elif isinstance(competences, list):
        comp_txt = ", ".join([str(x) for x in competences[:100]])
        if comp_txt:
            payload["Competences"] = comp_txt

    # Date d'analyse: maintenant si non fourni
    if not payload.get("DateAnalyse"):
        payload["DateAnalyse"] = _iso_datetime(datetime.utcnow())

    # Filter out keys not in schema if schema provided
    if schema:
        allowed = set(schema.keys()) | {"Title"}  # Title may not always be listed
        payload = {k: v for k, v in payload.items() if k in allowed}

    # Final trims and drop empties to avoid Choice/URL violations
    for k, v in list(payload.items()):
        if isinstance(v, str):
            payload[k] = v.strip()
        if payload[k] in ("", None):
            payload.pop(k, None)

    # Ensure Title exists
    if "Title" not in payload:
        # Prefer full name if available, else filename stem, else 'Candidate'
        fallback = nom_complet or os.path.splitext(nom_fichier or "Candidate")[0] or "Candidate"
        payload["Title"] = fallback

    return payload

def normalize_payload_for_csv(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize payload keys to guarantee CSV-required fields are present.

    - Ensure NomComplet exists; derive from Prenom + Nom if needed.
    - Ensure Title exists; prefer Title, else NomComplet, else NomFichier stem, else 'Candidate'.
    - Ensure Ville/Region/Pays keys exist (empty string if missing).
    """
    p = dict(payload or {})

    nom = (p.get("Nom") or "").strip()
    prenom = (p.get("Prenom") or "").strip()
    nom_complet = (p.get("NomComplet") or "").strip()
    if not nom_complet:
        nom_complet = (f"{prenom} {nom}" if prenom or nom else "").strip()
        if nom_complet:
            p["NomComplet"] = nom_complet

    title = (p.get("Title") or "").strip()
    if not title:
        title = nom_complet or (p.get("NomFichier") or "").rsplit(".", 1)[0] or "Candidate"
        p["Title"] = title

    # Core location keys default to empty strings
    for k in ["Ville", "Region", "Pays"]:
        p[k] = (p.get(k) or "").strip()

    return p


def build_candidates_list_row(
    payload: Dict[str, Any],
    scoring: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Construct a row dict for the candidate list CSV from canonical payload + scores."""
    row = {
        "Title": payload.get("Title", ""),
        "NomFichier": payload.get("NomFichier", ""),
        "ResumeHash": payload.get("ResumeHash", ""),
        "NomComplet": payload.get("NomComplet", ""),
        "Email": payload.get("Email", ""),
        "Telephone": payload.get("Telephone", ""),
        "LinkedInUrl": payload.get("LinkedInUrl", ""),
        "Ville": payload.get("Ville", ""),
        "Region": payload.get("Region", ""),
        "Pays": payload.get("Pays", ""),
        "ScoreGlobal": scoring.get("score_global"),
        "ScoreSkills": scoring.get("score_skills"),
        "Recommandation": scoring.get("recommandation"),
        "MatchCommentaire": scoring.get("match_commentaire"),
        "CorrelationId": meta.get("correlation_id"),
        "Source": meta.get("source", "Archon"),
        "Skills": ", ".join(scoring.get("skills", [])[:50]) if isinstance(scoring.get("skills"), list) else scoring.get("skills", ""),
    }
    return row


CSV_HEADER_MINIMAL = [
    "Title",
    "NomFichier",
    "ResumeHash",
    "NomComplet",
    "Email",
    "Telephone",
    "LinkedInUrl",
    "Ville",
    "Region",
    "Pays",
    "ScoreGlobal",
    "ScoreSkills",
    "Recommandation",
    "MatchCommentaire",
    "CorrelationId",
    "Source",
    "Skills",
]

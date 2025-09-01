# -*- coding: utf-8 -*-
"""Candidate schema dict for aggregator fill-all defaults (canonical)."""
from __future__ import annotations
from typing import Dict, Any

# type: string|number|boolean
CANDIDATE_SCHEMA: Dict[str, Any] = {
    # Contact
    "NomComplet": {"type": "string"},
    "Email": {"type": "string"},
    "Telephone": {"type": "string"},
    "LinkedInUrl": {"type": "string"},
    "NomFichier": {"type": "string"},
    "UrlFichier": {"type": "string"},
    "ResumeHash": {"type": "string"},
    "DocumentId": {"type": "string"},
    # Location
    "Ville": {"type": "string"},
    "Pays": {"type": "string"},
    "PretRelocaliser": {"type": "boolean"},
    "AutorisationTravail": {"type": "string"},
    # Education
    "NiveauEtudes": {"type": "string"},
    "LanguePrincipale": {"type": "string"},
    "AutresLangues": {"type": "string"},
    "NiveauFrancais": {"type": "string"},
    "NiveauAnglais": {"type": "string"},
    # Job
    "PosteCible": {"type": "string"},
    "Domaine": {"type": "string"},
    "Seniorite": {"type": "string"},
    # Competences
    "Competences": {"type": "string"},
    # Availability/Comp
    "DispoDate": {"type": "string"},
    "SalaireActuel": {"type": "number"},
    "SalaireSouhaite": {"type": "number"},
    "DernierContact": {"type": "string"},
    "ProchaineAction": {"type": "string"},
    "DateProchaineAction": {"type": "string"},
    # Scoring
    "Notes": {"type": "string"},
    "AppreciationGlobale": {"type": "string"},
    # Embedding
    "EmbeddingId": {"type": "string"},
    # Meta
    "DateAnalyse": {"type": "string"},
}

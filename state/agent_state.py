"""
Définition des modèles d'état agent pour Archon utilisant Pydantic.
Remplace l'ancienne implémentation TypedDict avec validation stricte.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """
    Modèle d'état d'agent avec validation complète via Pydantic.
    Utilisé pour le suivi d'état dans les graphes LangGraph.
    """
    latest_user_message: str
    next_user_message: str = ""
    messages: List[Dict[str, str]] = Field(default_factory=list,
                                        description="Liste d'historique de messages au format {role, content}")
    scope: str = ""
    advisor_output: str = ""
    file_list: List[str] = Field(default_factory=list,
                               description="Liste des fichiers pertinents pour le contexte")
    refined_prompt: str = ""
    refined_tools: str = ""
    refined_agent: str = ""
    generated_code: Optional[str] = None
    error: Optional[str] = None
    timings: Dict[str, Any] = Field(default_factory=dict,
                                  description="Métriques de timing par nœud")

    class Config:
        """Configuration du modèle Pydantic."""
        extra = "forbid"  # Rejette les champs non définis
        validate_assignment = True  # Valide à l'assignation d'attributs
        json_schema_extra = {
            "examples": [
                {
                    "latest_user_message": "Comment puis-je implémenter une API REST en Python?",
                    "messages": [{"role": "user", "content": "Comment puis-je implémenter une API REST en Python?"}],
                    "scope": "L'utilisateur souhaite créer une API REST en Python"
                }
            ]
        }

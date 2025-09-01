"""
Utilitaires pour la migration entre l'ancienne définition TypedDict 
et le nouveau modèle Pydantic d'AgentState.
"""
from typing import Dict, Any, Union, Optional, cast

from .agent_state import AgentState as PydanticAgentState


def convert_dict_to_agent_state(state_dict: Dict[str, Any]) -> PydanticAgentState:
    """
    Convertit un dict d'état (ancien format) en modèle Pydantic AgentState.
    
    Args:
        state_dict: Dictionnaire d'état potentiellement incomplet
        
    Returns:
        Instance validée d'AgentState
    """
    # Filtrer les clés None pour éviter les erreurs de validation
    clean_dict = {k: v for k, v in state_dict.items() if v is not None}
    
    # Créer une instance validée
    return PydanticAgentState(**clean_dict)


def ensure_agent_state(state: Union[Dict[str, Any], PydanticAgentState]) -> PydanticAgentState:
    """
    Garantit qu'un état est au format AgentState Pydantic.
    Convertit si nécessaire un dict en AgentState.
    
    Args:
        state: État au format dict ou AgentState
        
    Returns:
        Instance validée d'AgentState
    """
    if isinstance(state, PydanticAgentState):
        return state
    
    return convert_dict_to_agent_state(state)


def update_agent_state(
    state: Union[Dict[str, Any], PydanticAgentState], 
    updates: Dict[str, Any]
) -> PydanticAgentState:
    """
    Met à jour un état agent avec de nouvelles valeurs.
    
    Args:
        state: État existant (dict ou AgentState)
        updates: Mises à jour à appliquer
        
    Returns:
        AgentState mis à jour et validé
    """
    # Garantir que l'état d'origine est un AgentState
    agent_state = ensure_agent_state(state)
    
    # Mettre à jour avec les nouvelles valeurs
    updated_dict = agent_state.model_dump()
    updated_dict.update(updates)
    
    # Créer une nouvelle instance validée
    return PydanticAgentState(**updated_dict)

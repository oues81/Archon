"""
Module d'instrumentation standardisée pour les graphes LangGraph d'Archon.
Fournit des décorateurs et utilitaires pour mesurer performance et métriques.
"""
import asyncio
import functools
import time
import uuid
import logging
import json
from typing import Dict, Any, Optional, Union, List, Callable, TypeVar, cast
from datetime import datetime

import logfire
from pydantic import BaseModel

from k.state.agent_state import AgentState
from k.state.state_migration import ensure_agent_state

logger = logging.getLogger(__name__)

# Type générique pour la fonction à instrumenter
T = TypeVar('T')


class NodeMetrics(BaseModel):
    """Métriques de performance d'un nœud LangGraph."""
    node_id: str
    node_name: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    error: Optional[str] = None
    llm_tokens_in: Optional[int] = None
    llm_tokens_out: Optional[int] = None
    llm_model: Optional[str] = None
    llm_provider: Optional[str] = None


def time_node(func: Callable[..., T]) -> Callable[..., T]:
    """
    Décorateur pour instrumenter un nœud LangGraph et mesurer sa performance.
    Enregistre les métriques dans l'état de l'agent sous state.timings.
    
    Args:
        func: Fonction de nœud LangGraph à instrumenter
        
    Returns:
        Fonction instrumentée
        
    Example:
        ```python
        @time_node
        async def my_langgraph_node(state: AgentState, config: dict) -> AgentState:
            # implémentation du nœud
            return state
        ```
    """
    @functools.wraps(func)
    async def wrapper(state: Union[Dict[str, Any], AgentState], *args, **kwargs) -> Union[Dict[str, Any], AgentState]:
        # Identifier le nœud
        node_name = func.__name__
        node_id = str(uuid.uuid4())
        
        # Convertir en AgentState Pydantic si nécessaire
        input_is_dict = isinstance(state, dict)
        agent_state = ensure_agent_state(state)
        
        # Banner de début
        logger.info(f"▶️ Début du nœud: {node_name}")
        
        # Initialiser les métriques
        start_time = time.time()
        success = False
        error_message = None
        
        # Initialiser la structure des timings si nécessaire
        if "timings" not in agent_state.model_dump() or not agent_state.timings:
            agent_state.timings = {}
        
        try:
            # Appel à la fonction originale
            if input_is_dict:
                result = await func(agent_state.model_dump(), *args, **kwargs)
            else:
                result = await func(agent_state, *args, **kwargs)
                
            # Marquer comme réussi
            success = True
            
            # Convertir le résultat en AgentState Pydantic si nécessaire
            result_state = ensure_agent_state(result)
            
            return result_state if not input_is_dict else result_state.model_dump()
            
        except Exception as e:
            # Capture de l'erreur
            error_message = str(e)
            logger.error(f"❌ Erreur dans le nœud {node_name}: {error_message}")
            
            # Mettre à jour l'état avec l'erreur
            agent_state.error = f"Erreur dans {node_name}: {error_message}"
            
            # Rethrow or return state with error
            if input_is_dict:
                return agent_state.model_dump()
            return agent_state
            
        finally:
            # Calculer les métriques finales
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Créer l'objet de métriques
            metrics = {
                "node_id": node_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": duration_ms,
                "success": success
            }
            
            if error_message:
                metrics["error"] = error_message
                
            # Ajouter les métriques à l'état
            if agent_state.timings is not None:
                agent_state.timings[node_name] = metrics
                
            # Logging structuré avec logfire
            try:
                formatted_time = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")
                log_method = logfire.info if success else logfire.error
                log_method(
                    f"{node_name} completed" if success else f"{node_name} failed",
                    node=node_name,
                    duration_ms=duration_ms,
                    success=success,
                    time=formatted_time,
                    **({"error": error_message} if error_message else {})
                )
            except Exception as log_error:
                logger.warning(f"Erreur lors du logging structuré: {log_error}")
                
            # Banner de fin
            status_icon = "✅" if success else "❌"
            logger.info(f"{status_icon} Fin du nœud: {node_name} ({duration_ms:.2f}ms)")
    
    return wrapper


def collect_graph_metrics(state: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    """
    Extraire et formater toutes les métriques d'un état de graphe.
    
    Args:
        state: État du graphe (dict ou AgentState)
        
    Returns:
        Métriques formatées
    """
    agent_state = ensure_agent_state(state)
    
    # Extraire les timings
    timings = agent_state.timings or {}
    
    # Calculer les métriques globales
    total_duration = 0
    node_count = 0
    error_count = 0
    
    for node_name, node_metrics in timings.items():
        if "duration_ms" in node_metrics:
            total_duration += node_metrics["duration_ms"]
            node_count += 1
        if node_metrics.get("success") is False:
            error_count += 1
            
    # Formater les résultats
    return {
        "total_duration_ms": total_duration,
        "avg_node_duration_ms": total_duration / node_count if node_count > 0 else 0,
        "node_count": node_count,
        "error_count": error_count,
        "nodes": timings
    }


def log_graph_metrics(state: Union[Dict[str, Any], AgentState]) -> None:
    """
    Logger un résumé des métriques d'un graphe complet.
    
    Args:
        state: État du graphe (dict ou AgentState)
    """
    metrics = collect_graph_metrics(state)
    
    # Construire un résumé visuel
    logger.info(
        f"📊 Résumé du graphe: {metrics['node_count']} nœuds, "
        f"{metrics['total_duration_ms']:.2f}ms au total, "
        f"{metrics['error_count']} erreurs"
    )
    
    # Détail par nœud
    for node_name, node_metrics in metrics["nodes"].items():
        status = "✅" if node_metrics.get("success", True) else "❌"
        duration = node_metrics.get("duration_ms", 0)
        logger.info(f"  {status} {node_name}: {duration:.2f}ms")
        
    # Logging structuré avec logfire
    try:
        logfire.info(
            "Graph execution complete",
            metrics=metrics
        )
    except Exception as e:
        logger.warning(f"Erreur lors du logging structuré: {e}")


def create_instrumented_node(func: Callable):
    """
    Factory pour créer un nœud LangGraph instrumenté.
    
    Args:
        func: Fonction originale du nœud
        
    Returns:
        Fonction instrumentée
    """
    @time_node
    async def instrumented_node(state, config):
        return await func(state, config)
    
    return instrumented_node

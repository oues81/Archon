"""
Module de gestion des checkpoints et reprise d'état pour les graphes LangGraph.
Permet la sauvegarde d'états intermédiaires et la reprise de l'exécution.
"""
import json
import time
import uuid
import logging
import asyncio
import os
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from k.state.agent_state import AgentState
from k.state.state_migration import ensure_agent_state

logger = logging.getLogger(__name__)


class CheckpointMetadata(BaseModel):
    """Métadonnées des checkpoints d'état."""
    id: str
    timestamp: float
    flow_id: str
    node_name: str
    state_version: str = "1.0.0"
    tags: List[str] = []


class StateCheckpoint(BaseModel):
    """Modèle complet d'un checkpoint d'état."""
    metadata: CheckpointMetadata
    state: AgentState


class CheckpointManager:
    """
    Gestionnaire de checkpoints pour les états du graphe LangGraph.
    Permet la sauvegarde/chargement d'états intermédiaires.
    """
    def __init__(self, checkpoint_dir: Optional[str] = None):
        """
        Initialiser le gestionnaire de checkpoints.
        
        Args:
            checkpoint_dir: Répertoire de stockage des checkpoints (défaut: "./checkpoints")
        """
        self.checkpoint_dir = checkpoint_dir or os.environ.get(
            "ARCHON_CHECKPOINT_DIR", "./checkpoints"
        )
        # Créer le répertoire de checkpoints s'il n'existe pas
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Gestionnaire de checkpoints initialisé - dir: {self.checkpoint_dir}")
    
    def _get_checkpoint_path(self, checkpoint_id: str) -> str:
        """
        Obtenir le chemin du fichier pour un ID de checkpoint.
        
        Args:
            checkpoint_id: ID unique du checkpoint
            
        Returns:
            Chemin complet du fichier de checkpoint
        """
        return os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")
    
    async def save_checkpoint(
        self, 
        state: Union[Dict[str, Any], AgentState], 
        flow_id: str, 
        node_name: str, 
        tags: List[str] = None
    ) -> str:
        """
        Sauvegarder l'état courant comme checkpoint.
        
        Args:
            state: État à sauvegarder (dict ou AgentState Pydantic)
            flow_id: Identifiant du workflow
            node_name: Nom du nœud en cours d'exécution
            tags: Tags optionnels pour identifier le checkpoint
            
        Returns:
            ID unique du checkpoint créé
        """
        # Convertir en AgentState Pydantic si nécessaire
        agent_state = ensure_agent_state(state)
        
        # Générer un ID unique pour ce checkpoint
        checkpoint_id = str(uuid.uuid4())
        
        # Créer les métadonnées
        metadata = CheckpointMetadata(
            id=checkpoint_id,
            timestamp=time.time(),
            flow_id=flow_id,
            node_name=node_name,
            tags=tags or []
        )
        
        # Créer l'objet checkpoint complet
        checkpoint = StateCheckpoint(
            metadata=metadata,
            state=agent_state
        )
        
        # Sauvegarder le checkpoint
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        
        # Sauvegarder de manière asynchrone pour ne pas bloquer le workflow
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: Path(checkpoint_path).write_text(
                checkpoint.model_dump_json(indent=2, exclude_unset=True)
            )
        )
        
        logger.info(f"Checkpoint créé: {checkpoint_id} pour le nœud '{node_name}' du flow '{flow_id}'")
        return checkpoint_id
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[StateCheckpoint]:
        """
        Charger un checkpoint par son ID.
        
        Args:
            checkpoint_id: ID unique du checkpoint
            
        Returns:
            Objet checkpoint avec métadonnées et état, ou None si non trouvé
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint non trouvé: {checkpoint_id}")
            return None
            
        try:
            # Charger de manière asynchrone
            loop = asyncio.get_event_loop()
            checkpoint_data = await loop.run_in_executor(
                None,
                lambda: Path(checkpoint_path).read_text()
            )
            
            # Parser le JSON et créer l'objet Pydantic
            checkpoint = StateCheckpoint.model_validate_json(checkpoint_data)
            logger.info(f"Checkpoint chargé: {checkpoint_id} (flow: {checkpoint.metadata.flow_id}, "
                      f"nœud: {checkpoint.metadata.node_name})")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du checkpoint {checkpoint_id}: {e}")
            return None
    
    async def find_latest_checkpoint(
        self, 
        flow_id: str, 
        tags: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Trouver le checkpoint le plus récent pour un workflow et tags donnés.
        
        Args:
            flow_id: Identifiant du workflow
            tags: Liste de tags pour filtrer les checkpoints
            
        Returns:
            ID du checkpoint le plus récent, ou None si aucun trouvé
        """
        checkpoints = []
        
        # Parcourir tous les fichiers de checkpoint
        for checkpoint_file in Path(self.checkpoint_dir).glob("*.json"):
            try:
                # Charger le checkpoint
                checkpoint_data = checkpoint_file.read_text()
                checkpoint = StateCheckpoint.model_validate_json(checkpoint_data)
                
                # Vérifier si le checkpoint correspond aux critères
                if checkpoint.metadata.flow_id == flow_id:
                    # Vérifier les tags si spécifiés
                    if tags:
                        if not all(tag in checkpoint.metadata.tags for tag in tags):
                            continue
                    
                    checkpoints.append((checkpoint.metadata.timestamp, checkpoint.metadata.id))
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture du checkpoint {checkpoint_file}: {e}")
        
        # Trier par timestamp (du plus récent au plus ancien)
        checkpoints.sort(reverse=True)
        
        # Retourner le plus récent s'il existe
        if checkpoints:
            latest_checkpoint_id = checkpoints[0][1]
            logger.info(f"Checkpoint le plus récent trouvé: {latest_checkpoint_id} pour le flow '{flow_id}'")
            return latest_checkpoint_id
            
        logger.info(f"Aucun checkpoint trouvé pour le flow '{flow_id}'")
        return None
    
    async def list_checkpoints(
        self, 
        flow_id: Optional[str] = None, 
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Lister les checkpoints disponibles avec filtrage optionnel.
        
        Args:
            flow_id: Filtrer par ID de workflow
            tags: Filtrer par tags
            limit: Nombre maximum de checkpoints à retourner
            
        Returns:
            Liste de métadonnées de checkpoints
        """
        checkpoints = []
        
        # Parcourir tous les fichiers de checkpoint
        for checkpoint_file in Path(self.checkpoint_dir).glob("*.json"):
            try:
                # Charger le checkpoint
                checkpoint_data = checkpoint_file.read_text()
                checkpoint = StateCheckpoint.model_validate_json(checkpoint_data)
                
                # Appliquer les filtres
                if flow_id and checkpoint.metadata.flow_id != flow_id:
                    continue
                    
                if tags and not all(tag in checkpoint.metadata.tags for tag in tags):
                    continue
                
                # Ajouter les métadonnées à la liste
                checkpoints.append(checkpoint.metadata.model_dump())
                
            except Exception as e:
                logger.warning(f"Erreur lors de la lecture du checkpoint {checkpoint_file}: {e}")
        
        # Trier par timestamp (du plus récent au plus ancien)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Limiter le nombre de résultats
        return checkpoints[:limit]
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Supprimer un checkpoint par son ID.
        
        Args:
            checkpoint_id: ID unique du checkpoint
            
        Returns:
            True si suppression réussie, False sinon
        """
        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Checkpoint non trouvé pour suppression: {checkpoint_id}")
            return False
            
        try:
            # Supprimer de manière asynchrone
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: os.unlink(checkpoint_path)
            )
            
            logger.info(f"Checkpoint supprimé: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du checkpoint {checkpoint_id}: {e}")
            return False


# Instance singleton du gestionnaire de checkpoints
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """
    Obtenir l'instance singleton du gestionnaire de checkpoints.
    
    Returns:
        Instance du gestionnaire de checkpoints
    """
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager()
    return _checkpoint_manager


async def checkpoint_state(
    state: Union[Dict[str, Any], AgentState], 
    node_name: str,
    flow_id: Optional[str] = None,
    tags: List[str] = None
) -> str:
    """
    Fonction utilitaire pour sauvegarder l'état courant.
    
    Args:
        state: État à sauvegarder
        node_name: Nom du nœud en cours d'exécution
        flow_id: Identifiant du workflow (généré si non fourni)
        tags: Tags optionnels
        
    Returns:
        ID du checkpoint créé
    """
    manager = get_checkpoint_manager()
    # Générer un ID de flow si non fourni
    actual_flow_id = flow_id or f"flow-{int(time.time())}"
    return await manager.save_checkpoint(state, actual_flow_id, node_name, tags)


async def restore_state(
    checkpoint_id: Optional[str] = None,
    flow_id: Optional[str] = None,
    tags: List[str] = None
) -> Optional[AgentState]:
    """
    Fonction utilitaire pour restaurer un état depuis un checkpoint.
    
    Args:
        checkpoint_id: ID spécifique du checkpoint (prioritaire)
        flow_id: ID du workflow pour trouver le dernier checkpoint
        tags: Tags pour filtrer les checkpoints
        
    Returns:
        État restauré, ou None si non trouvé
    """
    manager = get_checkpoint_manager()
    
    # Utiliser l'ID spécifié ou trouver le plus récent
    actual_checkpoint_id = checkpoint_id
    if not actual_checkpoint_id and flow_id:
        actual_checkpoint_id = await manager.find_latest_checkpoint(flow_id, tags)
        
    if not actual_checkpoint_id:
        logger.warning("Aucun checkpoint trouvé pour restauration")
        return None
        
    # Charger le checkpoint
    checkpoint = await manager.load_checkpoint(actual_checkpoint_id)
    if checkpoint:
        return checkpoint.state
    
    return None

"""
Exemple d'intégration de toutes les améliorations dans un graphe LangGraph.
Démontre l'utilisation des nouveaux composants:
- Modèle Pydantic AgentState
- Gestion standardisée des erreurs LLM
- Instrumentation unifiée
- Mécanisme de checkpoint/reprise
"""
import os
import json
import asyncio
import logging
import sys
from typing import Dict, Any, Optional, List, Tuple
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# Imports des nouveaux composants
from archon.state.agent_state import AgentState
from archon.state.checkpoint import checkpoint_state, restore_state
from archon.llm.error_handling import llm_error_handler
from archon.utils.instrumentation import time_node, log_graph_metrics

# Configuration du logging
logger = logging.getLogger(__name__)


# Exemples de nœuds LangGraph instrumentés avec les nouvelles fonctionnalités
@time_node
async def analyzer_node(state: AgentState, config: Dict[str, Any]) -> AgentState:
    """
    Nœud d'analyse qui détermine les besoins de l'utilisateur.
    Utilise la gestion d'erreurs LLM standardisée.
    """
    logger.info(f"Analyse de la requête utilisateur: {state.latest_user_message}")
    
    # Checkpoint avant traitement pour permettre la reprise
    checkpoint_id = await checkpoint_state(state, "analyzer_node", "example_flow")
    logger.info(f"État sauvegardé avec checkpoint ID: {checkpoint_id}")
    
    try:
        # Appel LLM avec le wrapper de gestion d'erreurs
        @llm_error_handler(provider="openai", model="gpt-4", max_retries=2)
        async def call_analysis_llm(prompt: str):
            # Simulation d'un appel LLM
            await asyncio.sleep(0.5)  # Simule la latence
            return "L'utilisateur souhaite créer une application web avec une base de données."
        
        # Appel au LLM avec gestion d'erreurs
        analysis_result, metrics = await call_analysis_llm(state.latest_user_message)
        
        # Mettre à jour l'état avec le résultat
        updated_state = state.model_copy()
        updated_state.scope = analysis_result
        
        # Ajouter les métriques LLM aux timings
        if "llm_metrics" not in updated_state.timings:
            updated_state.timings["llm_metrics"] = []
            
        updated_state.timings["llm_metrics"].append({
            "provider": metrics.provider,
            "model": metrics.model,
            "duration_ms": metrics.duration_ms,
            "success": metrics.success,
            "tokens_in": metrics.request_tokens,
            "tokens_out": metrics.response_tokens
        })
        
        return updated_state
        
    except Exception as e:
        # En cas d'erreur, l'état contient déjà l'information d'erreur grâce au décorateur time_node
        logger.error(f"Erreur dans analyzer_node: {e}")
        state.error = f"Erreur d'analyse: {str(e)}"
        return state


@time_node
async def planner_node(state: AgentState, config: Dict[str, Any]) -> AgentState:
    """
    Nœud de planification qui définit les étapes nécessaires.
    """
    logger.info(f"Planification basée sur l'analyse: {state.scope}")
    
    # Checkpoint avant traitement
    checkpoint_id = await checkpoint_state(state, "planner_node", "example_flow")
    
    try:
        # Simulation d'un traitement de planification
        await asyncio.sleep(0.3)
        
        # Mettre à jour l'état
        updated_state = state.model_copy()
        updated_state.refined_prompt = (
            "Étapes de création d'une application web avec base de données:\n"
            "1. Définir le schéma de la base de données\n"
            "2. Créer le backend API avec FastAPI\n"
            "3. Implémenter le frontend avec React\n"
            "4. Configurer le déploiement Docker"
        )
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Erreur dans planner_node: {e}")
        state.error = f"Erreur de planification: {str(e)}"
        return state


@time_node
async def generator_node(state: AgentState, config: Dict[str, Any]) -> AgentState:
    """
    Nœud de génération qui produit du code basé sur la planification.
    """
    logger.info(f"Génération de code selon le plan")
    
    # Checkpoint avant traitement
    checkpoint_id = await checkpoint_state(state, "generator_node", "example_flow")
    
    try:
        # Appel LLM avec le wrapper de gestion d'erreurs
        @llm_error_handler(provider="openai", model="gpt-4", max_retries=2)
        async def call_generation_llm(prompt: str):
            # Simulation d'un appel LLM
            await asyncio.sleep(0.7)  # Simule la latence
            return """```python
from fastapi import FastAPI, Depends
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Configuration de la base de données
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Modèle de données
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)

# Créer les tables
Base.metadata.create_all(bind=engine)

# Application FastAPI
app = FastAPI()

# Dépendance pour obtenir une session DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users/")
def create_user(name: str, email: str, db: Session = Depends(get_db)):
    user = User(name=name, email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.get("/users/")
def read_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users
```"""
        
        # Appel au LLM avec gestion d'erreurs
        code_result, metrics = await call_generation_llm(state.refined_prompt)
        
        # Mettre à jour l'état avec le résultat
        updated_state = state.model_copy()
        updated_state.generated_code = code_result
        
        # Ajouter les métriques LLM aux timings
        if "llm_metrics" not in updated_state.timings:
            updated_state.timings["llm_metrics"] = []
            
        updated_state.timings["llm_metrics"].append({
            "provider": metrics.provider,
            "model": metrics.model,
            "duration_ms": metrics.duration_ms,
            "success": metrics.success,
            "tokens_in": metrics.request_tokens,
            "tokens_out": metrics.response_tokens
        })
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Erreur dans generator_node: {e}")
        state.error = f"Erreur de génération: {str(e)}"
        return state


async def should_end(state: AgentState) -> str:
    """
    Fonction de condition pour déterminer si le workflow doit se terminer.
    """
    # Terminer si une erreur est survenue ou si le code a été généré
    if state.error is not None:
        logger.info("Workflow terminé avec erreur")
        return "end"
        
    if state.generated_code is not None:
        logger.info("Workflow terminé avec succès")
        return "end"
        
    # Continuer avec le workflow par défaut
    return "continue"


async def build_enhanced_graph():
    """
    Construit le graphe LangGraph avec les améliorations.
    """
    # Créer le graphe avec le nouveau modèle Pydantic
    builder = StateGraph(AgentState)
    
    # Ajouter les nœuds
    builder.add_node("analyzer", analyzer_node)
    builder.add_node("planner", planner_node)
    builder.add_node("generator", generator_node)
    
    # Définir le point d'entrée
    builder.set_entry_point("analyzer")
    
    # Définir les transitions
    builder.add_edge("analyzer", "planner")
    builder.add_edge("planner", "generator")
    
    # Ajouter la condition de fin
    builder.add_conditional_edges(
        "generator",
        should_end,
        {
            "end": END,
            "continue": "analyzer"  # Boucle si nécessaire
        }
    )
    
    # Compiler le graphe
    graph = builder.compile()
    
    return graph


async def run_example(user_input: str, resume_from: Optional[str] = None):
    """
    Exécute le graphe avec les améliorations.
    
    Args:
        user_input: Message de l'utilisateur
        resume_from: ID de checkpoint pour reprendre un workflow
    """
    # Construire le graphe
    graph = await build_enhanced_graph()
    
    # Créer l'état initial ou restaurer depuis un checkpoint
    if resume_from:
        logger.info(f"Reprise du workflow depuis checkpoint: {resume_from}")
        initial_state = await restore_state(checkpoint_id=resume_from)
        if initial_state is None:
            logger.error(f"Impossible de restaurer le checkpoint: {resume_from}")
            initial_state = AgentState(
                latest_user_message=user_input,
                messages=[{"role": "user", "content": user_input}]
            )
    else:
        # Nouvel état
        initial_state = AgentState(
            latest_user_message=user_input,
            messages=[{"role": "user", "content": user_input}]
        )
    
    # Exécuter le graphe
    try:
        logger.info(f"Démarrage du workflow pour: '{user_input}'")
        final_state = await graph.ainvoke(initial_state)
        
        # Logger les métriques finales
        log_graph_metrics(final_state)
        
        # Vérifier le résultat
        if final_state.error:
            logger.error(f"Workflow terminé avec erreur: {final_state.error}")
            return {"status": "error", "error": final_state.error, "state": final_state.model_dump()}
            
        logger.info("Workflow terminé avec succès")
        return {
            "status": "success", 
            "result": {
                "scope": final_state.scope,
                "plan": final_state.refined_prompt,
                "code": final_state.generated_code
            },
            "metrics": final_state.timings,
            "state": final_state.model_dump()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du workflow: {e}")
        return {"status": "error", "error": str(e)}


# Point d'entrée pour l'exécution en standalone
async def main():
    """Point d'entrée pour l'exécution du script."""
    # Configurer le logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Exemple d'utilisation
    user_query = "Je veux créer une application web pour gérer mes contacts avec une base de données."
    
    print(f"Exécution du workflow pour: '{user_query}'")
    result = await run_example(user_query)
    
    if result["status"] == "success":
        print("\n✅ WORKFLOW TERMINÉ AVEC SUCCÈS")
        print(f"\nSCOPE:\n{result['result']['scope']}")
        print(f"\nPLAN:\n{result['result']['plan']}")
        print(f"\nCODE:\n{result['result']['code']}")
        
        # Afficher les métriques de performance
        print("\nMÉTRIQUES DE PERFORMANCE:")
        for node_name, metrics in result["metrics"].items():
            if node_name != "llm_metrics":
                print(f"  • {node_name}: {metrics['duration_ms']:.2f}ms")
    else:
        print(f"\n❌ ERREUR: {result['error']}")
    
    # Exemple de reprise après erreur
    print("\n🔄 SIMULATION D'UNE REPRISE DEPUIS UN CHECKPOINT")
    
    # Sauvegarder l'état final comme checkpoint pour démonstration
    checkpoint_id = await checkpoint_state(
        result["state"], 
        "demo_checkpoint", 
        "example_flow", 
        ["demo"]
    )
    
    print(f"État sauvegardé avec checkpoint ID: {checkpoint_id}")
    print(f"Reprise du workflow depuis checkpoint: {checkpoint_id}")
    
    # Reprendre depuis le checkpoint
    resumed_result = await run_example("", resume_from=checkpoint_id)
    
    if resumed_result["status"] == "success":
        print("\n✅ WORKFLOW REPRIS AVEC SUCCÈS")
    else:
        print(f"\n❌ ERREUR LORS DE LA REPRISE: {resumed_result['error']}")


if __name__ == "__main__":
    asyncio.run(main())

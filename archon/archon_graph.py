"""
Archon Graph - Agent Workflow Management
Utilizes a unified LLM provider to support multiple backends (Ollama, OpenAI, OpenRouter)
"""
import json
import logging
import os
import sys
import traceback
from api.profiles import router as profiles_router
from datetime import datetime
from typing import Annotated, List, Any, Optional, Dict, Union
from typing_extensions import TypedDict

# Configuration OpenRouter simplifiée
logging.info("🔧 Configuration OpenRouter simplifiée")

# Logger Configuration
logger = logging.getLogger(__name__)

# Path Configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Unified LLM Provider Import
from archon.llm_provider import llm_provider

# Pydantic AI Compatibility Imports
from pydantic_ai import RunContext, Agent as PydanticAgent, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
import httpx

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    SystemPromptPart,
    UserPromptPart
)


def get_llm_instance(model_name: str):
    """Crée et retourne une instance LLM configurée pour pydantic-ai."""
    provider_name = llm_provider.config.provider.lower()

    if provider_name == "ollama":
        logger.info(f"Configuration d'Ollama via l'API compatible OpenAI: {llm_provider.config.base_url}")
        ollama_provider = OpenAIProvider(base_url=f"{llm_provider.config.base_url}/v1")
        return OpenAIModel(model_name=model_name, provider=ollama_provider)
    elif provider_name == "openrouter":
        logger.info(f"🔧 Configuration d'OpenRouter via l'API compatible OpenAI")
        if not llm_provider.config.api_key:
            raise ValueError("❌ La clé API OpenRouter n'est pas configurée.")

        # Journaliser la clé API (version masquée) pour le débogage
        api_key = llm_provider.config.api_key
        masked_key = api_key[:6] + "*****" + api_key[-4:] if len(api_key) > 10 else "***"
        logger.info(f"🔑 Utilisation de la clé API OpenRouter: {masked_key}")

        try:
            # Utiliser directement la configuration de llm_provider.py
            # Cette approche garantit que les en-têtes d'authentification sont correctement configurés
            from openai import AsyncOpenAI
            
            # Créer un client OpenAI avec l'authentification correcte
            openai_client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Archon"
                }
            )
            
            # Créer le provider avec ce client
            openrouter_provider = OpenAIProvider(
                openai_client=openai_client
            )
            
            logger.info(f"✅ Provider OpenRouter correctement initialisé avec le modèle {model_name}")
            model = OpenAIModel(model_name=model_name, provider=openrouter_provider)
            
            # Configurer les en-têtes supplémentaires pour OpenRouter
            # Dans la version 0.4.7 de pydantic-ai, les en-têtes doivent être configurés différemment
            # Nous passons directement les en-têtes au client HTTP du fournisseur
            if hasattr(openrouter_provider.client, "http_client") and hasattr(openrouter_provider.client.http_client, "headers"):
                openrouter_provider.client.http_client.headers.update({
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "Archon"
                })
                logger.info("✅ En-têtes supplémentaires configurés pour OpenRouter")
            
            return model
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation du provider OpenRouter: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    else:
        raise ValueError(f"Fournisseur LLM non supporté pour pydantic-ai: {provider_name}")


# LangGraph Imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.state import START
    from langgraph.checkpoint.memory import MemorySaver
except ImportError as e:
    logger.warning(f"Could not import LangGraph: {e}")
    class StateGraph:
        def __init__(self, *args, **kwargs): pass
    END = None
    START = None
    MemorySaver = object

# Prompt Imports
from archon.agent_prompts import (
    prompt_refiner_agent_prompt, advisor_prompt, coder_prompt_with_examples
)

# Log models on startup
logger.info(f"LLM Provider: {llm_provider.config.provider}")
logger.info(f"Reasoner Model: {llm_provider.config.reasoner_model}")
logger.info(f"Primary Model: {llm_provider.config.primary_model}")

# Agent State Definition
class AgentState(TypedDict):
    latest_user_message: str
    next_user_message: str
    messages: List[Any]
    scope: str
    advisor_output: str
    file_list: List[str]
    refined_prompt: str
    refined_tools: str
    refined_agent: str
    generated_code: Optional[str] = None
    error: Optional[str] = None

def define_scope_with_reasoner(state: AgentState) -> AgentState:
    """Définit la portée avec l'agent reasoner"""
    print("🔍 REASONER - Starting with model:", llm_provider.config.reasoner_model)
    logger.info("="*50)
    logger.info("🔍 REASONER STARTING")
    logger.info(f"🔍 Modèle: {llm_provider.config.reasoner_model}")
    logger.info("="*50)
    """Définit la portée avec l'agent reasoner"""
    logger.info("="*50)
    logger.info("🔍 REASONER STARTING")
    logger.info(f"🔍 Modèle: {llm_provider.config.reasoner_model}")
    logger.info("="*50)
    """Defines the project scope using a reasoner agent"""
    try:
        logger.info("---STEP: Defining scope with reasoner agent---")
        
        # Log du modèle utilisé pour le reasoner
        llm_model = llm_provider.config.reasoner_model
        llm_provider_name = llm_provider.config.provider.lower()
        logger.info(f"🧠 REASONER - Modèle: {llm_provider_name}:{llm_model}")
        logger.info(f"🧠 REASONER - Message utilisateur: {state['latest_user_message']}")
        
        if 'scope' not in state:
            state['scope'] = ""
        
        reasoner_agent = PydanticAgent(
            get_llm_instance(llm_model),
            system_prompt=prompt_refiner_agent_prompt
        )
        
        logger.info("🔍 REASONER - Envoi de la requête...")
        result = reasoner_agent.run_sync(state['latest_user_message'])
        
        # Extraire le contenu du résultat
        full_response = result.content if hasattr(result, 'content') else str(result)

        logger.info(f"🔍 REASONER - Réponse complète reçue: {full_response[:200]}...")
        state['scope'] = full_response

        logger.info(f"🔍 Scope state: {state.get('scope', 'NOT SET')}")
        return state
        
    except Exception as e:
        error_msg = f"Error in define_scope: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state['error'] = error_msg
        return state

def advisor_with_examples(state: AgentState) -> AgentState:
    """Génère des conseils avec l'agent advisor"""
    print("💡 ADVISOR - Starting with model:", llm_provider.config.primary_model)
    logger.info("="*50)
    logger.info("💡 ADVISOR STARTING")
    logger.info(f"💡 Modèle: {llm_provider.config.primary_model}")
    logger.info("="*50)
    """Génère des conseils avec l'agent advisor"""
    logger.info("="*50)
    logger.info("💡 ADVISOR STARTING")
    logger.info(f"💡 Modèle: {llm_provider.config.primary_model}")
    logger.info("="*50)
    """Generates advice and examples using the advisor agent"""
    try:
        logger.info("---STEP: Generating advice with advisor agent---")
        
        # Log détaillé pour l'advisor
        llm_model = llm_provider.config.primary_model
        llm_provider_name = llm_provider.config.provider.lower()
        logger.info(f"💡 ADVISOR - Modèle: {llm_provider_name}:{llm_model}")
        
        advisor = PydanticAgent(
            get_llm_instance(llm_model),
            system_prompt=advisor_prompt
        )
        
        logger.info("💡 ADVISOR - Envoi de la requête...")
        result = advisor.run_sync("Generate advice based on the following scope", 
                                  deps={'scope': state['scope']})
        
        # Extraire le contenu du résultat
        full_response = result.content if hasattr(result, 'content') else str(result)

        logger.info(f"💡 ADVISOR - Réponse complète reçue: {full_response[:200]}...")
        state['advisor_output'] = full_response
        logger.info("Advice generated.")
        return state

    except Exception as e:
        error_msg = f"Error in advisor: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state['error'] = error_msg
        return state

def coder_agent(state: AgentState) -> AgentState:
    """Génère le code avec l'agent coder"""
    print("⚡ CODER - Starting with model:", llm_provider.config.primary_model)
    logger.info("="*50)
    logger.info("⚡ CODER STARTING")
    logger.info(f"⚡ Modèle: {llm_provider.config.primary_model}")
    logger.info("="*50)
    """Génère le code avec l'agent coder"""
    logger.info("="*50)
    logger.info("⚡ CODER STARTING")
    logger.info(f"⚡ Modèle: {llm_provider.config.primary_model}")
    logger.info("="*50)
    """Generates the final code using the coder agent"""
    try:
        logger.info("---STEP: Generating code with coder agent---")
        
        # Log détaillé pour le coder
        llm_model = llm_provider.config.primary_model
        llm_provider_name = llm_provider.config.provider.lower()
        logger.info(f"⚡ CODER - Modèle: {llm_provider_name}:{llm_model}")
        
        # Vérification de l'existence des clés dans l'état
        scope = state.get('scope', '')
        advisor_output = state.get('advisor_output', '')
        
        if not scope:
            logger.warning("⚡ CODER - Attention: La clé 'scope' est vide ou manquante")
            scope = "Aucun scope défini. Veuillez fournir plus d'informations."
            
        if not advisor_output:
            logger.warning("⚡ CODER - Attention: La clé 'advisor_output' est vide ou manquante")
            advisor_output = "Aucune recommandation de l'advisor. Utilisez le scope pour générer le code."
            
        # Mise à jour de l'état pour garantir que ces clés existent
        state['scope'] = scope
        state['advisor_output'] = advisor_output
        
        logger.info(f"⚡ CODER - Scope: {scope[:200]}...")
        logger.info(f"⚡ CODER - Advisor Output: {advisor_output[:200]}...")
        
        coder = PydanticAgent(
            get_llm_instance(llm_model),
            system_prompt=coder_prompt_with_examples
        )
        
        logger.info("⚡ CODER - Envoi de la requête...")
        result = coder.run_sync("Generate code based on scope and advisor output", 
                                  deps={'scope': state['scope'], 'advisor_output': state['advisor_output']})
        
        # Extraire le contenu du résultat
        full_response = result.content if hasattr(result, 'content') else str(result)

        logger.info(f"⚡ CODER - Réponse complète reçue: {full_response[:200]}...")
        state['generated_code'] = full_response
        logger.info("Code generated.")
        return state
        
    except Exception as e:
        error_msg = f"Error in coder: {str(e)}"
        logger.error(error_msg, exc_info=True)
        state['error'] = error_msg
        return state

# Initialize the StateGraph and build the workflow
builder = StateGraph(AgentState)
builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
builder.add_node("advisor_with_examples", advisor_with_examples)
builder.add_node("coder_agent", coder_agent)
builder.set_entry_point("define_scope_with_reasoner")
builder.add_edge("define_scope_with_reasoner", "advisor_with_examples")
builder.add_edge("advisor_with_examples", "coder_agent")
builder.add_edge("coder_agent", END)
memory = MemorySaver()
agentic_flow = builder.compile(checkpointer=memory)

if __name__ == '__main__':
    try:
        initial_state = {
            'latest_user_message': 'Hello, can you help me create an AI agent?',
            'next_user_message': '',
            'messages': [],
            'scope': '',
            'advisor_output': '',
            'file_list': [],
            'refined_prompt': '',
            'refined_tools': '',
            'refined_agent': ''
        }
        
        print("Starting agentic flow execution...")
        result = agentic_flow.invoke(initial_state)
        
        print("\nExecution Result:")
        print(f"- Latest Message: {result.get('latest_user_message', '')}")
        print(f"- Scope Defined: {bool(result.get('scope', ''))}")
        print(f"- Code Generated: {bool(result.get('generated_code', ''))}")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {str(e)}")
        print(f"[ERROR] Type: {type(e).__name__}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")

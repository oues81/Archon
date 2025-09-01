"""
Generalist Agent - Un agent PM/orchestrateur généraliste robuste pour Archon
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from k.core.utils.utils import get_env_var, build_llm_config_from_active_profile
from k.models import OllamaModel
from k.llm import LLMProvider, LLMConfig
from k.schemas import PydanticAIDeps

# Configurer le logging
logger = logging.getLogger(__name__)

# Prompt principal de l'agent généraliste
GENERALIST_PROMPT = """
# Generalist Agent - PM/Orchestrator

## Rôle et Objectif
Vous êtes un agent généraliste chargé d'orchestrer des solutions et de planifier des tâches complexes. Votre objectif est d'analyser les demandes, de formuler des plans d'action efficaces, et de coordonner l'utilisation d'autres outils et agents spécialisés.

## Compétences Principales
- **Analyse de problèmes**: Comprendre les demandes et les décomposer en tâches gérables
- **Planification stratégique**: Élaborer des séquences logiques d'actions
- **Coordination d'outils**: Identifier et orchestrer les outils/agents appropriés
- **Documentation**: Générer des plans clairs et des rapports compréhensibles
- **Gestion de contexte**: Maintenir le contexte à travers des interactions multiples

## Limites et Contraintes
- Vous ne pouvez pas exécuter directement du code, mais vous pouvez le planifier et l'expliquer
- Vous devez toujours respecter les règles de sécurité et de confidentialité des données
- En cas d'incertitude, demandez des clarifications avant de procéder

## Format de Sortie
Votre réponse doit être structurée comme suit:

```json
{
  "analyse": {
    "objectif": "Objectif principal identifié",
    "contraintes": ["Contrainte 1", "Contrainte 2"],
    "contexte_pertinent": ["Élément de contexte 1", "Élément de contexte 2"]
  },
  "plan": {
    "étapes": [
      {"ordre": 1, "action": "Description de l'étape", "outil": "Nom de l'outil/agent à utiliser"},
      {"ordre": 2, "action": "Description de l'étape", "outil": "Nom de l'outil/agent à utiliser"}
    ]
  },
  "recommandations": {
    "outils": ["Nom outil 1", "Nom outil 2"],
    "considérations": ["Considération importante 1", "Considération importante 2"]
  }
}
```

## Règles de qualité
- Soyez précis et concis
- Ne faites pas de suppositions non fondées
- Priorisez toujours les solutions les plus directes et efficaces
- Documentez clairement votre raisonnement
- Adaptez votre niveau de détail en fonction de la complexité du problème
"""

class GeneralistAgent:
    """
    Agent généraliste polyvalent pour l'orchestration de tâches et la planification.
    
    Utilise la bibliothèque pydantic_ai pour créer un agent robuste capable de:
    - Analyser des demandes complexes
    - Planifier des séquences d'actions
    - Orchestrer l'utilisation d'autres outils et agents
    - Gérer le contexte à travers des interactions multiples
    """

    def __init__(
        self,
        profile: Optional[Dict[str, Any]] = None,
        custom_model: Optional[Any] = None,
        custom_prompt: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        retries: int = 3
    ):
        """
        Initialise l'agent généraliste.
        
        Args:
            profile: Configuration du profil LLM (provider, modèle, etc.)
            custom_model: Modèle personnalisé à utiliser à la place de celui du profil
            custom_prompt: Prompt système personnalisé à utiliser
            tools: Liste d'outils à mettre à disposition de l'agent
            retries: Nombre de tentatives en cas d'échec
        """
        self.profile = profile or build_llm_config_from_active_profile()
        self.prompt = custom_prompt or GENERALIST_PROMPT
        self.tools = tools or []
        self.retries = retries
        
        # Initialiser le modèle LLM
        self.model = custom_model or self._initialize_model()
        
        # Créer l'agent pydantic_ai
        self.agent = Agent(
            model=self.model,
            system_prompt=self.prompt,
            tools=self.tools,
            retries=self.retries,
            deps_type=PydanticAIDeps
        )
        
        logger.info("Agent généraliste initialisé avec succès")

    def _initialize_model(self) -> Any:
        """
        Initialise le modèle LLM basé sur la configuration du profil.
        Gère les fallbacks et les différents providers.
        
        Returns:
            Une instance de modèle compatible avec pydantic_ai
        """
        provider = self.profile.get("LLM_PROVIDER", "openrouter").lower()
        model_name = self.profile.get("GENERALIST_MODEL") or self.profile.get("PRIMARY_MODEL")
        api_key = self.profile.get("LLM_API_KEY", "")
        base_url = self.profile.get("BASE_URL", "")
        
        logger.info(f"Initialisation du modèle pour l'agent généraliste: {provider}/{model_name}")
        
        try:
            # Import conditionnel des modèles spécifiques en fonction du provider
            if provider == "anthropic":
                try:
                    from pydantic_ai.models.anthropic import AnthropicModel
                    return AnthropicModel(model_name, api_key=api_key)
                except ImportError:
                    logger.warning("Module AnthropicModel non disponible, fallback sur modèle générique")
            
            elif provider == "openai":
                try:
                    from pydantic_ai.models.openai import OpenAIModel
                    kwargs = {"api_key": api_key}
                    if base_url:
                        kwargs["base_url"] = base_url
                    
                    # Import OpenAI de manière conditionnelle
                    try:
                        from openai import AsyncOpenAI
                        openai_client = AsyncOpenAI(**kwargs)
                        return OpenAIModel(model_name=model_name, openai_client=openai_client)
                    except ImportError:
                        logger.warning("Module AsyncOpenAI non disponible")
                        return OpenAIModel(model_name=model_name, **kwargs)
                except ImportError:
                    logger.warning("Module OpenAIModel non disponible, fallback sur modèle générique")
            
            elif provider == "openrouter":
                try:
                    from pydantic_ai.models.openai import OpenAIModel
                    kwargs = {
                        "api_key": api_key,
                        "base_url": base_url or "https://openrouter.ai/api/v1"
                    }
                    
                    # Headers optionnels pour OpenRouter
                    headers = {}
                    ref = self.profile.get("OPENROUTER_REFERRER")
                    xtitle = self.profile.get("OPENROUTER_X_TITLE")
                    if ref:
                        headers["HTTP-Referer"] = ref
                    if xtitle:
                        headers["X-Title"] = xtitle
                    
                    if headers:
                        kwargs["default_headers"] = headers
                    
                    # Import OpenAI de manière conditionnelle
                    try:
                        from openai import AsyncOpenAI
                        openai_client = AsyncOpenAI(**kwargs)
                        return OpenAIModel(model_name=model_name, openai_client=openai_client)
                    except ImportError:
                        logger.warning("Module AsyncOpenAI non disponible")
                        return OpenAIModel(model_name=model_name, **kwargs)
                except ImportError:
                    logger.warning("Module OpenAIModel non disponible, fallback sur modèle générique")
            
            # Fallback sur Ollama
            ollama_allowed = self.profile.get("ALLOW_OLLAMA_FALLBACK", False)
            if provider == "ollama" or ollama_allowed:
                ollama_model = self.profile.get("OLLAMA_MODEL", "llama3")
                ollama_url = self.profile.get("OLLAMA_BASE_URL", "http://localhost:11434")
                return OllamaModel(model_name=ollama_model, base_url=ollama_url)
                
            # Si aucun provider valide n'est trouvé, lever une exception
            raise ValueError(f"Provider LLM non supporté ou non configuré: {provider}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du modèle LLM: {str(e)}")
            raise
    
    async def run(
        self, 
        message: str, 
        file_context: Optional[List[str]] = None,
        deps: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Exécute l'agent généraliste sur un message avec contexte optionnel.
        
        Args:
            message: Le message utilisateur à traiter
            file_context: Liste optionnelle de chemins de fichiers à utiliser comme contexte
            deps: Dépendances supplémentaires à passer à l'agent
            
        Returns:
            Résultat de l'exécution de l'agent
        """
        try:
            # Préparer le message avec le contexte des fichiers si fourni
            enhanced_message = message
            if file_context and len(file_context) > 0:
                file_context_str = "\n\n".join([f"Fichier: {path}" for path in file_context])
                enhanced_message = f"{message}\n\nContexte des fichiers:\n{file_context_str}"
            
            # Exécuter l'agent avec gestion des erreurs
            logger.info(f"Exécution de l'agent généraliste sur: {message[:100]}...")
            
            # Préparer les dépendances
            context_deps = deps or {}
            if file_context:
                context_deps["file_list"] = file_context
                
            # Exécuter l'agent avec retries
            response = await self.agent.run(enhanced_message, deps=context_deps)
            logger.info("Exécution de l'agent généraliste réussie")
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'agent généraliste: {str(e)}")
            raise

    def get_system_prompt(self) -> str:
        """
        Récupère le prompt système actuel de l'agent.
        
        Returns:
            Le prompt système actuel
        """
        return self.prompt

    def set_system_prompt(self, new_prompt: str) -> None:
        """
        Met à jour le prompt système de l'agent.
        
        Args:
            new_prompt: Le nouveau prompt système à utiliser
        """
        self.prompt = new_prompt
        # Réinitialiser l'agent avec le nouveau prompt
        self.agent = Agent(
            model=self.model,
            system_prompt=new_prompt,
            tools=self.tools,
            retries=self.retries,
            deps_type=PydanticAIDeps
        )
        logger.info("Prompt système de l'agent généraliste mis à jour")


# Fonction de création d'agent pour faciliter l'instanciation
def create_generalist_agent(
    profile: Optional[Dict[str, Any]] = None,
    custom_model: Optional[Any] = None,
    custom_prompt: Optional[str] = None,
    tools: Optional[List[Any]] = None
) -> GeneralistAgent:
    """
    Crée et retourne un agent généraliste configuré.
    
    Args:
        profile: Configuration du profil LLM
        custom_model: Modèle personnalisé à utiliser
        custom_prompt: Prompt système personnalisé
        tools: Liste d'outils à mettre à disposition
        
    Returns:
        Une instance configurée de GeneralistAgent
    """
    return GeneralistAgent(
        profile=profile,
        custom_model=custom_model,
        custom_prompt=custom_prompt,
        tools=tools
    )

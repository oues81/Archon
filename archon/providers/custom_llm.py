from litellm import completion
from typing import Union

def get_llm_model(provider: str, settings: dict) -> completion:
    """
    Retourne un modèle LLM configuré selon le fournisseur
    Args:
        provider: Le fournisseur du modèle (openai, anthropic, etc.)
        settings: Les paramètres de configuration
    Returns:
        Un objet de modèle LLM configuré
    """
    return completion(provider=provider, **settings)

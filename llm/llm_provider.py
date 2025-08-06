from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    """Configuration de base pour un fournisseur LLM"""
    provider_name: str = Field(..., alias="provider")
    model_name: str = Field(..., alias="model") 
    api_key: str | None = None
    base_url: str | None = None

class LLMProvider:
    """Classe de base pour les fournisseurs LLM"""
    def __init__(self, config: LLMConfig):
        self.config = config

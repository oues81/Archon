from pydantic import BaseModel
from typing import Any

class PydanticAIDeps(BaseModel):
    supabase: Any
    embedding_client: Any
    reasoner_output: str
    advisor_output: str

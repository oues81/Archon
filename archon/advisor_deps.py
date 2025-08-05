from dataclasses import dataclass
from typing import Optional

@dataclass
class AdvisorDeps:
    model_config: Optional[dict] = None
    tools: Optional[list] = None

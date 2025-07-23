"""
Custom JSON encoder for handling Pydantic AI specific types.
"""
import json
from typing import Any, Dict, Union
import logging

logger = logging.getLogger(__name__)

class ToolDefinitionEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles ToolDefinition objects."""
    
    def default(self, obj: Any) -> Union[Dict[str, Any], str]:
        try:
            # Try to get the module dynamically to avoid import errors
            ToolDefinition = self._get_tool_definition_class()
            if ToolDefinition and isinstance(obj, ToolDefinition):
                return self._encode_tool_definition(obj)
            return super().default(obj)
        except Exception as e:
            logger.warning(f"Error in custom JSON encoder: {e}")
            return str(obj)
    
    def _get_tool_definition_class(self):
        """Dynamically import ToolDefinition class if available."""
        try:
            from pydantic_ai.tools import ToolDefinition
            return ToolDefinition
        except ImportError:
            return None
    
    def _encode_tool_definition(self, obj) -> Dict[str, Any]:
        """Encode a ToolDefinition object to a dictionary."""
        return {
            'name': getattr(obj, 'name', ''),
            'description': getattr(obj, 'description', ''),
            'parameters': getattr(obj, 'parameters_json_schema', {}),
            'type': 'tool_definition'
        }

# Sauvegarder la fonction d'origine
_original_dumps = json.dumps

def dumps(obj: Any, **kwargs) -> str:
    """Custom dumps function that uses our custom encoder."""
    # Utiliser notre encodeur personnalisé si aucun autre n'est spécifié
    if 'cls' not in kwargs:
        kwargs['cls'] = ToolDefinitionEncoder
    # Utiliser la fonction d'origine pour éviter la récursion
    return _original_dumps(obj, **kwargs)

# Ne pas patcher json.dumps ici pour éviter les problèmes d'importation circulaire
# L'importation sera gérée dans advisor_agent.py

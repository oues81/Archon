"""
Package de configuration pour Archon AI.
Contient les configurations pour les modèles LLM.
"""
import sys as _sys

# Expose archon.config.model_config en alias du module réel archon.archon.config.model_config
try:
    from k.config.model_config import ModelConfig as _ModelConfig
    import k.config.model_config as _model_config_mod
    _sys.modules[__name__ + ".model_config"] = _model_config_mod
    ModelConfig = _ModelConfig
    __all__ = ["ModelConfig"]
except Exception:  # En cas de changement d'arborescence, ne pas casser l'import
    __all__ = []

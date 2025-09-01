# Shim de compatibilité pour remplacer l'agent advisor par l'agent généraliste
# Permet de maintenir la compatibilité des imports existants
from k.services.agents.generalist_agent import GeneralistAgent, create_generalist_agent

# Classe de compatibilité pour maintenir l'API existante
class AdvisorAgent(GeneralistAgent):
    """Agent advisor basé sur l'agent généraliste."""
    pass

# Exporter les symboles pour maintenir la compatibilité
__all__ = ['AdvisorAgent', 'create_generalist_agent']

"""
Deprecated: BaseAgent has been removed.

Raison: la sélection et l'orchestration des modèles se font désormais via les
profils (env_vars.json) et le reasoner/dispatcher. Ce module est volontairement
supprimé pour éviter toute divergence.

Si vous voyez cette erreur, mettez à jour le code appelant pour utiliser le
reasoner/dispatcher ou les fabriques fondées sur les profils (p.ex. ModelFactory).
"""

raise ImportError(
    "archon.agents.base_agent a été retiré. Utilisez le reasoner/dispatcher et"
    " les profils (env_vars.json) pour sélectionner les modèles."
)

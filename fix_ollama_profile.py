import json

with open('/app/workbench/env_vars.json', 'r') as f:
    config = json.load(f)

# Corriger les noms de modèles dans le profil Ollama
config['profiles']['ollama_light_cpu_phi3']['LLM_MODEL'] = 'phi3:mini'
config['profiles']['ollama_light_cpu_phi3']['REASONER_MODEL'] = 'phi3:mini'
config['profiles']['ollama_light_cpu_phi3']['PRIMARY_MODEL'] = 'phi3:mini'

with open('/app/workbench/env_vars.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Profil ollama corrigé avec les bons noms de modèles')

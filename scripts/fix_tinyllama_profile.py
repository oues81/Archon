import json

with open('/app/workbench/env_vars.json', 'r') as f:
    config = json.load(f)

# Corriger le profil tinyllama avec les bons noms de modèles
config['profiles']['ollama_light_cpu_tinyllama']['LLM_MODEL'] = 'tinyllama:latest'
config['profiles']['ollama_light_cpu_tinyllama']['REASONER_MODEL'] = 'tinyllama:latest'  
config['profiles']['ollama_light_cpu_tinyllama']['PRIMARY_MODEL'] = 'tinyllama:latest'

# Changer vers ce profil
config['current_profile'] = 'ollama_light_cpu_tinyllama'

with open('/app/workbench/env_vars.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Profil ollama_light_cpu_tinyllama corrigé et activé')

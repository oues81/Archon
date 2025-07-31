import json

with open('/app/workbench/env_vars.json', 'r') as f:
    config = json.load(f)

config['current_profile'] = 'ollama_light_cpu_tinyllama'

with open('/app/workbench/env_vars.json', 'w') as f:
    json.dump(config, f, indent=2)

print('Profil changé vers Ollama TinyLlama')

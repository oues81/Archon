import re

# Lire le fichier
with open('/app/archon/llm_provider.py', 'r') as f:
    content = f.read()

# Corriger la configuration Ollama pour utiliser BASE_URL au lieu de OLLAMA_BASE_URL
old_line = 'self.config.base_url = get_env_from_profile("OLLAMA_BASE_URL", "http://localhost:11434")'
new_line = 'self.config.base_url = get_env_from_profile("BASE_URL", "http://host.docker.internal:11434")'

content = content.replace(old_line, new_line)

# Écrire le fichier corrigé
with open('/app/archon/llm_provider.py', 'w') as f:
    f.write(content)

print('Configuration Ollama corrigée dans llm_provider.py')

import os
import json
from pathlib import Path

# Chemin vers le fichier de configuration
workbench_dir = Path("/app/workbench")
env_vars_file = workbench_dir / "env_vars.json"

print(f"Vérification de l'accès au fichier: {env_vars_file}")
print(f"Le fichier existe: {env_vars_file.exists()}")
print(f"Le fichier est lisible: {os.access(env_vars_file, os.R_OK)}")

# Essayer de lire le fichier
try:
    with open(env_vars_file, 'r') as f:
        config = json.load(f)
        print("Contenu du fichier:")
        print(json.dumps(config, indent=2))
        
        # Vérifier le profil actif
        current_profile = config.get('current_profile')
        print(f"\nProfil actif: {current_profile}")
        
        if current_profile in config.get('profiles', {}):
            profile = config['profiles'][current_profile]
            print(f"\nConfiguration du profil {current_profile}:")
            for key, value in profile.items():
                print(f"{key}: {value}")
        else:
            print(f"Le profil {current_profile} n'existe pas dans la configuration.")
            
except Exception as e:
    print(f"Erreur lors de la lecture du fichier: {e}")

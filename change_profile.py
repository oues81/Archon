#!/usr/bin/env python3
"""
Script utilitaire pour changer le profil actif d'Archon.
Usage:
    python change_profile.py <nom_du_profil>
"""
import json
import sys
from pathlib import Path

def switch_profile(profile_name: str, config_path: str = '/app/workbench/env_vars.json'):
    """Change le profil actif dans le fichier de configuration"""
    try:
        config_path = Path(config_path)
        
        # Charger la configuration existante
        if not config_path.exists():
            print(f"❌ Fichier de configuration introuvable: {config_path}")
            return False
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Vérifier que le profil existe
        if profile_name not in config.get('profiles', {}):
            available = list(config.get('profiles', {}).keys())
            print(f"❌ Le profil '{profile_name}' n'existe pas. Profils disponibles: {', '.join(available)}")
            return False
        
        # Mettre à jour le profil actif
        current = config.get('current_profile')
        if current == profile_name:
            print(f"ℹ️  Le profil est déjà sur: {profile_name}")
            return True
            
        config['current_profile'] = profile_name
        
        # Sauvegarder les changements
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Profil changé avec succès: {current} → {profile_name}")
        return True
        
    except json.JSONDecodeError:
        print(f"❌ Erreur: Le fichier de configuration n'est pas un JSON valide: {config_path}")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du changement de profil: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python change_profile.py <nom_du_profil>")
        print("\nExemples:")
        print("  python change_profile.py openrouter_models")
        print("  python change_profile.py ollama_light_cpu_phi4")
        sys.exit(1)
    
    profile_name = sys.argv[1]
    success = switch_profile(profile_name)
    sys.exit(0 if success else 1)

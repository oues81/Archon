#!/usr/bin/env python3
"""
Module de gestion dynamique des profils pour Archon
Permet de changer de profil sans redémarrer le service
"""
import os
import json
import logging
import typer
from pathlib import Path
from typing import Dict, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Gestionnaire dynamique de profils Archon")

def get_config_path() -> Path:
    """Retourne le chemin du fichier de configuration"""
    return Path(os.getenv('ARCHON_CONFIG', '/app/workbench/env_vars.json'))

def load_config() -> Dict:
    """Charge la configuration complète"""
    config_path = get_config_path()
    if not config_path.exists():
        return {"profiles": {}, "current_profile": None}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Le fichier de configuration n'est pas un JSON valide: {config_path}")
        return {"profiles": {}, "current_profile": None}

def update_environment(profile_config: Dict) -> bool:
    """Met à jour les variables d'environnement avec la configuration du profil"""
    try:
        # Mettre à jour les variables d'environnement
        for key, value in profile_config.items():
            os.environ[key] = str(value)
            logger.info(f"Variable mise à jour: {key}={value}")
        
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour des variables d'environnement: {e}")
        return False

@app.command()
def switch(profile_name: str):
    """Change dynamiquement le profil actif et met à jour les variables d'environnement"""
    config = load_config()
    
    if not config.get('profiles'):
        logger.error("Aucun profil configuré.")
        raise typer.Exit(1)
    
    if profile_name not in config['profiles']:
        logger.error(f"Le profil '{profile_name}' n'existe pas")
        logger.info("Profils disponibles:")
        for name in config['profiles'].keys():
            logger.info(f"- {name}")
        raise typer.Exit(1)
    
    # Mettre à jour le profil actif dans le fichier de configuration
    config['current_profile'] = profile_name
    
    # Sauvegarder la configuration
    try:
        with open(get_config_path(), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde: {e}")
        raise typer.Exit(1)
    
    # Mettre à jour les variables d'environnement
    profile_config = config['profiles'][profile_name]
    if update_environment(profile_config):
        logger.info(f"✅ Profil changé dynamiquement vers: {profile_name}")
        logger.info(f"✅ Variables d'environnement mises à jour")
        
        # Afficher la configuration actuelle
        logger.info(f"Configuration actuelle:")
        for key, value in profile_config.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.error(f"❌ Échec de la mise à jour des variables d'environnement")
        raise typer.Exit(1)

@app.command()
def current():
    """Affiche le profil actif et les variables d'environnement associées"""
    config = load_config()
    current_profile = config.get('current_profile')
    
    if not current_profile:
        logger.error("Aucun profil actif")
        raise typer.Exit(1)
    
    profile_config = config.get('profiles', {}).get(current_profile)
    if not profile_config:
        logger.error(f"Configuration du profil {current_profile} non trouvée")
        raise typer.Exit(1)
    
    logger.info(f"Profil actif: {current_profile}")
    logger.info("Configuration:")
    for key, value in profile_config.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Variables d'environnement actuelles:")
    for key in profile_config.keys():
        env_value = os.environ.get(key, "Non définie")
        logger.info(f"  {key}: {env_value}")

if __name__ == "__main__":
    app()

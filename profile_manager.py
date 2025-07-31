#!/usr/bin/env python3
"""
Gestionnaire de profils pour Archon
Permet de gérer les profils de configuration via la ligne de commande.
"""
import json
import typer
from pathlib import Path
from typing import Dict, List, Optional
import os

app = typer.Typer(help="Gestionnaire de profils Archon")

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
        typer.echo(f"❌ Erreur: Le fichier de configuration n'est pas un JSON valide: {config_path}", err=True)
        raise typer.Exit(1)

def save_config(config: Dict) -> None:
    """Sauvegarde la configuration"""
    config_path = get_config_path()
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        typer.echo(f"❌ Erreur lors de la sauvegarde: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def list():
    """Liste tous les profils disponibles"""
    config = load_config()
    current = config.get('current_profile')
    
    if not config.get('profiles'):
        typer.echo("Aucun profil configuré.")
        return
    
    typer.echo("Profils disponibles:")
    for name in config['profiles'].keys():
        prefix = "✓ " if name == current else "  "
        typer.echo(f"{prefix}{name}")
    
    if current:
        typer.echo(f"\nProfil actif: {current}")

@app.command()
def use(profile_name: str):
    """Change le profil actif"""
    config = load_config()
    
    if not config.get('profiles'):
        typer.echo("Aucun profil configuré.", err=True)
        raise typer.Exit(1)
    
    if profile_name not in config['profiles']:
        typer.echo(f"Erreur: Le profil '{profile_name}' n'existe pas", err=True)
        typer.echo("\nProfils disponibles:")
        for name in config['profiles'].keys():
            typer.echo(f"- {name}")
        raise typer.Exit(1)
    
    if config.get('current_profile') == profile_name:
        typer.echo(f"Le profil est déjà sur: {profile_name}")
        return
    
    config['current_profile'] = profile_name
    save_config(config)
    typer.echo(f"✅ Profil changé avec succès vers: {profile_name}")

@app.command()
def show(profile_name: Optional[str] = None):
    """Affiche la configuration d'un profil"""
    config = load_config()
    
    if not profile_name:
        profile_name = config.get('current_profile')
        if not profile_name:
            typer.echo("Aucun profil actif", err=True)
            raise typer.Exit(1)
    
    if profile_name not in config.get('profiles', {}):
        typer.echo(f"Erreur: Le profil '{profile_name}' n'existe pas", err=True)
        raise typer.Exit(1)
    
    profile = config['profiles'][profile_name]
    typer.echo(f"Configuration du profil: {profile_name}")
    typer.echo(json.dumps(profile, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    app()

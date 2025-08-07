#!/usr/bin/env python3
"""
Gestionnaire de profils pour Archon
Permet de gérer les profils de configuration via la ligne de commande.
"""
import json
import typer
from pathlib import Path
from typing import Dict, List, Optional, Any
import os
import sys
from importlib import import_module

# Ajouter le répertoire parent au path pour les imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Importer le LLMProvider
try:
    from archon.llm import LLMProvider, LLMConfig
    from archon.llm import get_config_path as get_llm_config_path
    LLM_PROVIDER_AVAILABLE = True
except ImportError:
    LLM_PROVIDER_AVAILABLE = False
    typer.echo("⚠️  Le module llm_provider n'est pas disponible. Certaines fonctionnalités seront limitées.", err=True)

app = typer.Typer(help="Gestionnaire de profils Archon")

def get_config_path() -> Path:
    """Retourne le chemin du fichier de configuration"""
    # Chemin par défaut si la variable d'environnement n'est pas définie
    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'workbench',
        'env_vars.json'
    )
    return Path(os.getenv('ARCHON_CONFIG', default_path))

def get_default_profiles() -> Dict[str, Any]:
    """Retourne les profils par défaut intégrés au code"""
    return {
        "current_profile": "openai_default",
        "profiles": {
            "ollama_default": {
                "LLM_PROVIDER": "ollama",
                "OLLAMA_BASE_URL": "http://localhost:11434/v1",
                "PRIMARY_MODEL": "llama2",
                "REASONER_MODEL": "llama2",
                "CODER_MODEL": "codellama:7b",
                "AGENT_MODE": "non-interactive"
            },
            "openai_default": {
                "LLM_PROVIDER": "openai",
                "LLM_API_KEY": "",
                "PRIMARY_MODEL": "gpt-4",
                "REASONER_MODEL": "gpt-4",
                "CODER_MODEL": "gpt-4",
                "AGENT_MODE": "non-interactive"
            },
            "openrouter_default": {
                "LLM_PROVIDER": "openrouter",
                "LLM_API_KEY": "",
                "BASE_URL": "https://openrouter.ai/api/v1",
                "REASONER_MODEL": "anthropic/claude-2",
                "PRIMARY_MODEL": "anthropic/claude-2",
                "CODER_MODEL": "anthropic/claude-2",
                "AGENT_MODE": "non-interactive"
            }
        }
    }

def load_config(create_if_missing: bool = True) -> Dict:
    """Charge la configuration complète"""
    config_path = get_config_path()
    
    # Si le fichier n'existe pas, on le crée avec les valeurs par défaut
    if not config_path.exists():
        if create_if_missing:
            default_config = get_default_profiles()
            # Créer le répertoire parent si nécessaire
            config_path.parent.mkdir(parents=True, exist_ok=True)
            save_config(default_config)
            return default_config
        return {"profiles": {}, "current_profile": None}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
            # S'assurer que la structure est valide
            if not isinstance(config, dict):
                raise json.JSONDecodeError("Configuration invalide", "", 0)
                
            if "profiles" not in config:
                config["profiles"] = {}
                
            if "current_profile" not in config:
                # Essayer de déterminer un profil courant par défaut
                if "openai_default" in config["profiles"]:
                    config["current_profile"] = "openai_default"
                elif config["profiles"]:
                    # Prendre le premier profil disponible
                    config["current_profile"] = next(iter(config["profiles"].keys()))
                else:
                    config["current_profile"] = None
            
            return config
            
    except json.JSONDecodeError as e:
        typer.echo(f"❌ Erreur: Le fichier de configuration n'est pas un JSON valide: {config_path}", err=True)
        typer.echo(f"Détail de l'erreur: {str(e)}", err=True)
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
def list(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Afficher plus de détails sur chaque profil")
):
    """Liste tous les profils disponibles"""
    config = load_config()
    current = config.get('current_profile')
    
    if not config.get('profiles'):
        typer.echo("Aucun profil configuré.")
        return
    
    typer.echo("\nProfils disponibles :")
    for name, profile in config['profiles'].items():
        prefix = "* " if name == current else "  "
        
        if verbose:
            provider = profile.get('provider', 'inconnu')
            model = profile.get('model', 'non spécifié')
            typer.echo(f"{prefix}{name} (Provider: {provider}, Modèle: {model})")
        else:
            typer.echo(f"{prefix}{name}")
    
    if current:
        typer.echo(f"\nProfil actif : {current}")
        
        if verbose and current in config['profiles']:
            typer.echo("\nDétails du profil actif :")
            for key, value in config['profiles'][current].items():
                if key == 'api_key' and value:
                    value = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '***'
                typer.echo(f"  {key}: {value}")
    
    if verbose:
        typer.echo("\nChemin de configuration : " + str(get_config_path()))

@app.command()
def use(
    profile_name: str,
    reload_llm: bool = typer.Option(
        True, 
        help="Recharge le fournisseur LLM après le changement de profil"
    )
):
    """Change le profil actif"""
    config = load_config()
    
    if profile_name not in config['profiles']:
        typer.echo(f"❌ Le profil '{profile_name}' n'existe pas.", err=True)
        raise typer.Exit(1)
        
    config['current_profile'] = profile_name
    save_config(config)
    
    # Recharger le fournisseur LLM si demandé et disponible
    if reload_llm and LLM_PROVIDER_AVAILABLE:
        try:
            provider = LLMProvider()
            if provider.reload_profile(profile_name):
                typer.echo(f"✅ Profil changé pour : {profile_name} (fournisseur LLM rechargé)")
            else:
                typer.echo(f"⚠️  Profil changé pour : {profile_name} (mais échec du rechargement du fournisseur LLM)", err=True)
        except Exception as e:
            typer.echo(f"⚠️  Profil changé pour : {profile_name} (mais erreur lors du rechargement du fournisseur LLM: {e})", err=True)
    else:
        typer.echo(f"✅ Profil changé pour : {profile_name}")
        if not LLM_PROVIDER_AVAILABLE:
            typer.echo("  ℹ️  Le fournisseur LLM n'a pas été rechargé car il n'est pas disponible.", err=True)

@app.command()
def show(
    profile_name: Optional[str] = typer.Argument(
        None, 
        help="Nom du profil à afficher. Si non spécifié, affiche le profil actif."
    ),
    show_sensitive: bool = typer.Option(
        False, 
        "--show-sensitive", "-s", 
        help="Afficher les informations sensibles comme les clés API"
    )
):
    """Affiche la configuration d'un profil"""
    config = load_config()
    
    if not profile_name:
        profile_name = config.get('current_profile')
        if not profile_name:
            typer.echo("❌ Aucun profil actif. Spécifiez un nom de profil.", err=True)
            raise typer.Exit(1)
    
    if profile_name not in config['profiles']:
        typer.echo(f"❌ Le profil '{profile_name}' n'existe pas.", err=True)
        raise typer.Exit(1)
    
    profile_data = config['profiles'][profile_name].copy()
    
    # Masquer les informations sensibles si demandé
    if not show_sensitive:
        for key in ['api_key', 'password', 'secret', 'token']:
            if key in profile_data and profile_data[key]:
                value = str(profile_data[key])
                profile_data[key] = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '***'
    
    # Afficher les informations du profil
    typer.echo(f"\n📋 Profil : {typer.style(profile_name, fg=typer.colors.BRIGHT_CYAN, bold=True)}")
    typer.echo("=" * (len(profile_name) + 10))
    
    # Afficher les informations de base
    typer.echo("\n🔧 Configuration :")
    for key, value in profile_data.items():
        if key not in ['api_key', 'password', 'secret', 'token'] or show_sensitive:
            typer.echo(f"  {key}: {typer.style(str(value), fg=typer.colors.GREEN)}")
    
    # Afficher les informations sensibles si demandé
    if any(key in profile_data for key in ['api_key', 'password', 'secret', 'token']) and not show_sensitive:
        typer.echo("\n🔒 Informations sensibles masquées. Utilisez --show-sensitive pour les afficher.")
    
    # Afficher des informations supplémentaires si le fournisseur LLM est disponible
    if LLM_PROVIDER_AVAILABLE and profile_name == config.get('current_profile'):
        try:
            provider = LLMProvider()
            if provider.get_current_profile() == profile_name:
                models = provider.get_available_models()
                if models:
                    typer.echo("\n🤖 Modèles disponibles :")
                    for model_type, model_name in models.items():
                        typer.echo(f"  {model_type}: {typer.style(model_name, fg=typer.colors.BLUE)}")
        except Exception as e:
            typer.echo(f"\n⚠️  Impossible de récupérer les informations du fournisseur LLM: {e}", err=True)
    
    typer.echo()  # Ligne vide à la fin
    typer.echo(json.dumps(profile_data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    app()

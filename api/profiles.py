"""
Endpoints API pour la gestion des profils de configuration
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
import os
from typing import Dict, List, Optional

router = APIRouter()

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
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Erreur de décodage JSON: {e}")

def save_config(config: Dict) -> None:
    """Sauvegarde la configuration"""
    config_path = get_config_path()
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde: {e}")

@router.get("/profiles", response_model=Dict[str, List[str]])
async def list_profiles():
    """Liste tous les profils disponibles"""
    try:
        config = load_config()
        return {
            "profiles": list(config.get("profiles", {}).keys()),
            "current": config.get("current_profile")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/profiles/switch/{profile_name}")
async def switch_profile(profile_name: str):
    """Change le profil actif"""
    try:
        config = load_config()
        
        if profile_name not in config.get('profiles', {}):
            available = list(config.get('profiles', {}).keys())
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Le profil '{profile_name}' n'existe pas",
                    "available_profiles": available
                }
            )
        
        current = config.get('current_profile')
        if current == profile_name:
            return {
                "status": "success",
                "message": f"Le profil est déjà sur: {profile_name}",
                "profile": profile_name
            }
        
        config['current_profile'] = profile_name
        save_config(config)
        
        return {
            "status": "success",
            "message": f"Profil changé avec succès: {current} → {profile_name}",
            "previous_profile": current,
            "current_profile": profile_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profiles/current", response_model=Dict)
async def get_current_profile():
    """Récupère le profil actif"""
    try:
        config = load_config()
        current = config.get('current_profile')
        
        if not current:
            raise HTTPException(status_code=404, detail="Aucun profil actif")
        
        if current not in config.get('profiles', {}):
            raise HTTPException(
                status_code=500,
                detail=f"Le profil actif '{current}' n'existe plus dans la configuration"
            )
        
        return {
            "profile": current,
            "config": config['profiles'][current]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

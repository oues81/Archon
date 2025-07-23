#!/usr/bin/env python3
"""
Script pour construire et exécuter les conteneurs Docker d'Archon.
Gère le cycle de vie complet des conteneurs avec nettoyage approprié.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Configuration
MIN_DOCKER_VERSION = (20, 10, 0)
MIN_PYTHON_VERSION = (3, 10, 0)
DEFAULT_COMPOSE_FILE = "docker-compose.yml"

def check_python_version() -> None:
    """Vérifie que la version de Python est compatible."""
    if sys.version_info < MIN_PYTHON_VERSION:
        print(f"Erreur: Python {'.'.join(str(v) for v in MIN_PYTHON_VERSION)}+ est requis")
        sys.exit(1)

def check_docker_version() -> Tuple[bool, str]:
    """Vérifie que Docker est installé et à une version compatible."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        version_str = result.stdout.strip().split()[2].rstrip(",")
        version = tuple(map(int, version_str.split(".")[:3]))
        
        if version < MIN_DOCKER_VERSION:
            return False, f"Docker {'.'.join(map(str, MIN_DOCKER_VERSION))}+ est requis"
        return True, f"Docker {version_str}"
    except (subprocess.SubprocessError, FileNotFoundError):
        return False, "Docker n'est pas installé ou n'est pas dans le PATH"

def parse_arguments() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Gestionnaire de déploiement Archon")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force la reconstruction des images Docker"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Nettoie les ressources Docker avant de démarrer"
    )
    parser.add_argument(
        "--compose-file",
        default=DEFAULT_COMPOSE_FILE,
        help=f"Fichier docker-compose à utiliser (par défaut: {DEFAULT_COMPOSE_FILE})"
    )
    return parser.parse_args()

def run_command(
    command: List[str],
    cwd: Optional[Union[str, Path]] = None,
    suppress_errors: bool = False,
    capture_output: bool = False
) -> Tuple[int, str]:
    """Exécute une commande et retourne le code de sortie et la sortie.
    
    Args:
        command: Commande à exécuter
        cwd: Répertoire de travail
        suppress_errors: Si True, n'affiche pas d'erreur en cas d'échec
        capture_output: Si True, capture la sortie au lieu de l'afficher
        
    Returns:
        Tuple (code_retour, sortie)
    """
    cwd_str = str(cwd) if cwd else os.getcwd()
    print(f"Exécution dans {cwd_str}: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.STDOUT if capture_output else None,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=cwd_str
        )
        
        output = []
        if process.stdout:
            for line in process.stdout:
                line = line.strip()
                if not capture_output:
                    print(line)
                output.append(line)
        
        process.wait()
        output_str = '\n'.join(output)
        
        if process.returncode != 0 and not suppress_errors:
            print(f"Erreur: La commande a échoué avec le code {process.returncode}")
            if capture_output:
                print(output_str)
        
        return process.returncode, output_str
        
    except Exception as e:
        if not suppress_errors:
            print(f"Erreur lors de l'exécution de la commande: {e}")
        return 1, str(e)

def check_command(command: str) -> bool:
    """Vérifie si une commande est disponible."""
    try:
        subprocess.run(
            [command, "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def cleanup_resources(compose_file: str, compose_dir: Path) -> None:
    """Nettoie les ressources Docker."""
    print("\n=== Nettoyage des ressources Docker... ===")
    
    # Arrêt et suppression des conteneurs
    run_command(
        ["docker", "compose", "-f", compose_file, "down", "--remove-orphans", "--volumes"],
        cwd=compose_dir
    )
    
    # Nettoyage des ressources inutilisées
    run_command(["docker", "system", "prune", "-f"])

def build_images(compose_file: str, compose_dir: Path, force_rebuild: bool = False) -> bool:
    """Construit les images Docker requises."""
    print("\n=== Construction des images Docker ===")
    
    # Vérifier si le fichier docker-compose existe
    if not (compose_dir / compose_file).exists():
        print(f"Erreur: Le fichier {compose_file} est introuvable dans {compose_dir}")
        return False
    
    # Options de construction
    build_cmd = ["docker", "compose", "-f", compose_file, "build"]
    if force_rebuild:
        build_cmd.append("--no-cache")
    
    # Construction des images
    return_code, _ = run_command(build_cmd, cwd=compose_dir)
    return return_code == 0

def start_services(compose_file: str, compose_dir: Path) -> bool:
    """Démarre les services Docker."""
    print("\n=== Démarrage des services ===")
    
    # Démarrer les services en arrière-plan
    return_code, _ = run_command(
        ["docker", "compose", "-f", compose_file, "up", "-d", "archon"],
        cwd=compose_dir
    )
    
    if return_code != 0:
        print("Erreur lors du démarrage des services")
        return False
    
    # Attendre que le service soit prêt
    print("\n=== Attente du démarrage du service... ===")
    time.sleep(10)
    
    # Afficher les logs initiaux
    print("\n=== Logs initiaux du service ===")
    run_command(
        ["docker", "compose", "-f", compose_file, "logs", "--tail=50", "archon"],
        cwd=compose_dir
    )
    
    return True

def check_service_health(compose_file: str, compose_dir: Path) -> bool:
    """Vérifie l'état de santé du service."""
    print("\n=== Vérification de l'état du service ===")
    
    # Vérifier que le conteneur est en cours d'exécution
    return_code, output = run_command(
        ["docker", "ps", "--filter", "name=archon", "--format", "{{.Status}}"],
        capture_output=True
    )
    
    if return_code != 0 or not output:
        print("Erreur: Le service Archon ne semble pas être en cours d'exécution")
        return False
    
    print(f"État du service: {output.strip()}")
    return "up" in output.lower()

def main() -> int:
    """Fonction principale."""
    # Vérifications initiales
    check_python_version()
    
    docker_ok, docker_msg = check_docker_version()
    if not docker_ok:
        print(f"Erreur: {docker_msg}")
        return 1
    
    # Analyse des arguments
    args = parse_arguments()
    
    # Charger les variables d'environnement à partir du fichier .env
    # Cela garantit que docker-compose peut voir les variables comme LLM
    load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

    # Obtenir le chemin du répertoire du script
    script_dir = Path(__file__).parent.resolve()
    compose_dir = script_dir.parent
    compose_file = args.compose_file
    
    # Nettoyage si demandé
    if args.clean:
        cleanup_resources(compose_file, compose_dir)
    
    # Construction des images
    if not build_images(compose_file, compose_dir, args.rebuild):
        return 1
    
    # Démarrage des services
    if not start_services(compose_file, compose_dir):
        return 1
    
    # Vérification de l'état du service
    if not check_service_health(compose_file, compose_dir):
        print("\n[ATTENTION] Le service ne semble pas être en bonne santé")
        print("Veuillez vérifier les logs avec: docker compose logs -f archon")
        return 1
    
    # Affichage des informations de connexion
    print("\n=== Service Archon démarré avec succès! ===")
    print("Interface utilisateur: http://localhost:8501")
    print("\nCommandes utiles:")
    print("  Voir les logs:        docker compose logs -f archon")
    print("  Arrêter le service:   docker compose down")
    print("  Accès au shell:      docker compose exec archon /bin/bash")
    
    return 0

if __name__ == "__main__":
    # Ensure the script is run from the correct directory context if needed
    # but paths are absolute, so it should be fine.
    sys.exit(main())
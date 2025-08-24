#!/usr/bin/env python3
"""
Script de benchmark pour comparer les performances des différents profils LLM
"""
import time
import json
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics
import typer
from datetime import datetime

app = typer.Typer()

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
        typer.echo(f"❌ Erreur de décodage JSON: {e}", err=True)
        raise typer.Exit(1)

def switch_profile(profile_name: str) -> bool:
    """Change le profil actif"""
    try:
        response = requests.post(
            "http://localhost:8100/api/profiles/switch/" + profile_name,
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        typer.echo(f"❌ Erreur lors du changement de profil: {e}", err=True)
        return False

def test_profile(profile_name: str, prompt: str, num_requests: int = 3) -> Dict:
    """Teste un profil avec plusieurs requêtes et retourne les métriques"""
    typer.echo(f"\n🔍 Test du profil: {profile_name}")
    
    # Changer de profil
    if not switch_profile(profile_name):
        return {
            "profile": profile_name,
            "status": "error",
            "error": "Impossible de changer de profil"
        }
    
    # Données de test
    latencies = []
    tokens_per_second = []
    success_count = 0
    
    for i in range(num_requests):
        try:
            start_time = time.time()
            
            # Envoyer une requête de test
            response = requests.post(
                "http://localhost:8100/invoke",
                json={
                    "message": prompt,
                    "thread_id": f"benchmark-{profile_name}-{i}",
                    "is_first_message": True
                },
                timeout=30
            )
            
            # Calculer les métriques
            end_time = time.time()
            latency = end_time - start_time
            latencies.append(latency)
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '')
                tokens = len(content.split())  # Estimation grossière des tokens
                
                if tokens > 0:
                    tps = tokens / latency
                    tokens_per_second.append(tps)
                    success_count += 1
                    
                    typer.echo(f"  ✅ Requête {i+1}: {latency:.2f}s, {tps:.1f} tokens/s")
                else:
                    typer.echo(f"  ⚠️  Réponse vide pour la requête {i+1}")
            else:
                typer.echo(f"  ❌ Erreur {response.status_code} pour la requête {i+1}: {response.text}")
                
        except Exception as e:
            typer.echo(f"  ❌ Exception pour la requête {i+1}: {str(e)}")
    
    # Calculer les statistiques
    stats = {
        "profile": profile_name,
        "total_requests": num_requests,
        "successful_requests": success_count,
        "success_rate": success_count / num_requests if num_requests > 0 else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    if latencies:
        stats.update({
            "avg_latency": statistics.mean(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "latency_stddev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        })
    
    if tokens_per_second:
        stats.update({
            "avg_tokens_per_second": statistics.mean(tokens_per_second),
            "max_tokens_per_second": max(tokens_per_second),
        })
    
    return stats

def save_benchmark_results(results: List[Dict], output_file: Optional[str] = None) -> str:
    """Sauvegarde les résultats du benchmark dans un fichier"""
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    return output_file

@app.command()
def run(
    prompt: str = typer.Option(
        "Peux-tu me donner un résumé des dernières avancées en intelligence artificielle en 3-5 phrases ?",
        "--prompt", "-p",
        help="Prompt à utiliser pour le benchmark"
    ),
    num_requests: int = typer.Option(
        3,
        "--requests", "-r",
        help="Nombre de requêtes à envoyer par profil"
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Fichier de sortie pour les résultats (par défaut: benchmark_<timestamp>.json)"
    )
):
    """Exécute un benchmark sur tous les profils disponibles"""
    typer.echo(f"🚀 Démarrage du benchmark avec le prompt: {prompt}")
    
    # Charger la configuration pour obtenir la liste des profils
    config = load_config()
    profiles = config.get('profiles', {})
    
    if not profiles:
        typer.echo("❌ Aucun profil trouvé dans la configuration", err=True)
        raise typer.Exit(1)
    
    # Sauvegarder le profil actuel pour le restaurer à la fin
    current_profile = config.get('current_profile')
    
    try:
        # Tester chaque profil
        results = []
        for profile_name in profiles.keys():
            result = test_profile(profile_name, prompt, num_requests)
            results.append(result)
            
            # Afficher un résumé pour ce profil
            if 'error' in result:
                typer.echo(f"\n❌ Erreur avec le profil {profile_name}: {result['error']}")
            else:
                typer.echo(f"\n📊 Résumé pour {profile_name}:")
                typer.echo(f"  - Taux de réussite: {result['success_rate']*100:.1f}%")
                
                if 'avg_latency' in result:
                    typer.echo(f"  - Latence moyenne: {result['avg_latency']:.2f}s")
                    typer.echo(f"  - Latence min/max: {result['min_latency']:.2f}s / {result['max_latency']:.2f}s")
                
                if 'avg_tokens_per_second' in result:
                    typer.echo(f"  - Débit moyen: {result['avg_tokens_per_second']:.1f} tokens/s")
        
        # Sauvegarder les résultats
        output_path = save_benchmark_results(results, output_file)
        typer.echo(f"\n📝 Résultats sauvegardés dans: {output_path}")
        
        # Afficher le classement
        successful_results = [r for r in results if 'error' not in r and 'avg_latency' in r]
        if successful_results:
            typer.echo("\n🏆 Classement par vitesse (latence moyenne):")
            for i, result in enumerate(sorted(successful_results, key=lambda x: x['avg_latency'])):
                typer.echo(f"  {i+1}. {result['profile']}: {result['avg_latency']:.2f}s")
        
    finally:
        # Restaurer le profil initial
        if current_profile and current_profile in profiles:
            typer.echo(f"\n🔄 Restauration du profil initial: {current_profile}")
            switch_profile(current_profile)

if __name__ == "__main__":
    app()

#!/usr/bin/env python3
"""
Patch pour corriger le bug PydanticAI/OpenRouter datetime
"""
import re

def patch_pydantic_ai_datetime_bug():
    """Corrige le bug de validation datetime dans PydanticAI pour OpenRouter"""
    
    # Fichier à patcher
    openai_model_file = '/usr/local/lib/python3.10/site-packages/pydantic_ai/models/openai.py'
    
    print("🔧 PATCH: Correction du bug datetime PydanticAI/OpenRouter")
    
    try:
        with open(openai_model_file, 'r') as f:
            content = f.read()
        
        # Chercher la ligne problématique
        old_line = 'timestamp = number_to_datetime(response.created)'
        
        if old_line in content:
            # Corriger pour gérer le cas où response.created est None
            new_line = '''timestamp = number_to_datetime(response.created) if response.created is not None else datetime.now(timezone.utc)'''
            
            content = content.replace(old_line, new_line)
            
            # Ajouter l'import nécessaire
            if 'from datetime import datetime, timezone' not in content:
                content = content.replace(
                    'from datetime import datetime',
                    'from datetime import datetime, timezone'
                )
            
            # Sauvegarder le fichier patché
            with open(openai_model_file, 'w') as f:
                f.write(content)
            
            print("✅ Patch appliqué avec succès!")
            print("  - response.created=None géré")
            print("  - Fallback vers datetime.now(timezone.utc)")
            
        else:
            print("❌ Ligne à patcher non trouvée")
            
    except Exception as e:
        print(f"❌ Erreur lors du patch: {e}")

if __name__ == "__main__":
    patch_pydantic_ai_datetime_bug()

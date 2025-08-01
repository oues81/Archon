import traceback
import sys
import os
import json
from datetime import datetime

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Appliquer le correctif pour TypedDict
try:
    from patch_typing import *
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Optional, Dict, Any, List

# Importer agentic_flow depuis le bon emplacement
from .archon_graph import agentic_flow
try:
    from langgraph.types import Command
except ImportError:
    # Fallback implementation if langgraph.types.Command is not available
    class Command:
        """Dummy implementation of Command for compatibility"""
        pass
        
from utils.utils import write_to_log
    
# Création de l'application principale
app = FastAPI()

# Importer et inclure le routeur des profils
from api.profiles import router as profiles_router
app.include_router(profiles_router, prefix="/api")

class InvokeRequest(BaseModel):
    message: str
    thread_id: str
    is_first_message: bool = False
    config: Optional[Dict[str, Any]] = None

def extract_content_from_result(result_obj):
    """Fonction utilitaire pour extraire le contenu d'un objet result de manière sûre."""
    if result_obj is None:
        return ""
    
    # Si c'est déjà une chaîne, la retourner
    if isinstance(result_obj, str):
        return result_obj
    
    # Essayer d'extraire l'attribut content
    if hasattr(result_obj, 'content'):
        content = result_obj.content
        if isinstance(content, str):
            return content
        elif hasattr(content, 'content'):  # Parfois content est lui-même un objet
            return str(content.content)
        else:
            return str(content)
    
    # Essayer d'extraire l'attribut data
    if hasattr(result_obj, 'data'):
        data = result_obj.data
        if isinstance(data, str):
            return data
        else:
            return str(data)
    
    # Essayer d'extraire l'attribut text
    if hasattr(result_obj, 'text'):
        return str(result_obj.text)
    
    # Essayer d'extraire l'attribut message
    if hasattr(result_obj, 'message'):
        message = result_obj.message
        if isinstance(message, str):
            return message
        elif hasattr(message, 'content'):
            return str(message.content)
        else:
            return str(message)
    
    # Si c'est un dictionnaire, essayer d'extraire les clés communes
    if isinstance(result_obj, dict):
        for key in ['content', 'text', 'message', 'response', 'output']:
            if key in result_obj:
                return str(result_obj[key])
    
    # Dernière tentative : convertir en chaîne
    try:
        return str(result_obj)
    except Exception as e:
        print(f"[WARNING] Impossible de convertir l'objet en chaîne: {e}")
        return "Erreur lors de l'extraction du contenu"

def safe_len(obj):
    """Fonction utilitaire pour obtenir la longueur d'un objet de manière sûre."""
    try:
        if obj is None:
            return 0
        if isinstance(obj, (str, list, dict, tuple)):
            return len(obj)
        if hasattr(obj, '__len__'):
            return len(obj)
        # Pour les autres objets, retourner la longueur de leur représentation string
        return len(str(obj))
    except Exception:
        return 0

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}    

@app.post("/invoke")
async def invoke_agent(request: InvokeRequest):
    """Process a message through the agentic flow and return the complete response.

    The agent streams the response but this API endpoint waits for the full output
    before returning so it's a synchronous operation for MCP.
    Another endpoint will be made later to fully stream the response from the API.
    
    Args:
        request: The InvokeRequest containing message and thread info
        
    Returns:
        dict: Contains the complete response from the agent
    """
    try:
        config = request.config or {
            "configurable": {
                "thread_id": request.thread_id
            }
        }

        print(f"[DEBUG] Starting invoke_agent for thread {request.thread_id}")
        
        # Utiliser safe_len pour éviter les erreurs
        message_length = safe_len(request.message)
        if message_length > 100:
            print(f"[DEBUG] Request message: {request.message[:100]}...")
        else:
            print(f"[DEBUG] Request message: {request.message}")
            
        print(f"[DEBUG] Is first message: {request.is_first_message}")
        
        response = ""
        if request.is_first_message:
            print("[DEBUG] Processing first message")
            write_to_log(f"Processing first message for thread {request.thread_id}")
            
            # Initialiser l'état avec tous les champs requis par AgentState
            # Créer un message utilisateur correctement formaté selon le schéma attendu
            user_message = {
                "kind": "request",  # Doit être 'request' ou 'response'
                "parts": [{
                    "part_kind": "user-prompt",
                    "content": request.message,
                    "timestamp": datetime.utcnow().isoformat()
                }]
            }
            
            # Convertir le message en bytes pour le stockage dans l'état
            message_bytes = json.dumps([user_message]).encode('utf-8')
            
            initial_state = {
                "latest_user_message": request.message,
                "next_user_message": "",
                "messages": [message_bytes],  # Stocker le message formaté
                "scope": "",
                "advisor_output": "",
                "file_list": [],
                "refined_prompt": "",
                "refined_tools": "",
                "refined_agent": "",
                "config": config
            }
            
            print("[DEBUG] Initial state created, starting agentic flow...")
            print(f"[DEBUG] Initial state keys: {initial_state.keys()}")
            print("[DEBUG] Message bytes content:", message_bytes.decode('utf-8'))
            
            initial_state_str = json.dumps(initial_state, default=str)
            state_length = safe_len(initial_state_str)
            if state_length > 500:
                print("[DEBUG] Initial state content:", initial_state_str[:500] + "...")
            else:
                print("[DEBUG] Initial state content:", initial_state_str)
            
            # Ajouter un compteur pour suivre les itérations
            iteration = 0
            
            try:
                final_state = None
                async for msg in agentic_flow.astream(
                    initial_state, 
                    config,
                    stream_mode="values"
                ):
                    iteration += 1
                    print(f"[DEBUG] Received state update {iteration} from agentic flow")
                    print(f"[DEBUG] Message type: {type(msg)}")
                    
                    # msg contient l'état complet à chaque étape
                    if isinstance(msg, dict):
                        final_state = msg
                        # Extraire la réponse générée si disponible
                        # if 'generated_code' in msg and msg.generated_code:
                        #     print(f"[DEBUG] Generated code found: {msg.generated_code[:100]}...")
                        # if 'scope' in msg and msg.scope:
                        #     print(f"[DEBUG] Scope found: {msg.scope[:100]}...")
                        # if 'advisor_output' in msg and msg.advisor_output:
                        #     print(f"[DEBUG] Advisor output found: {msg.advisor_output[:100]}...")
                    
                    # Vérifier si nous sommes bloqués dans une boucle
                    if iteration > 10:  # Limite arbitraire pour éviter les boucles infinies
                        print("[WARNING] Possible infinite loop detected, breaking after 10 iterations")
                        break
                
                # Extraire la réponse finale de l'état
                if final_state:
                    print(f"[DEBUG] Final state keys: {list(final_state.keys())}")
                    
                    # Priorité à generated_code, puis advisor_output, puis scope
                    if final_state.get('generated_code'):
                        result_obj = final_state['generated_code']
                        print(f"[DEBUG] Generated code object type: {type(result_obj)}")
                        response = extract_content_from_result(result_obj)
                        response_length = safe_len(response)
                        print(f"[DEBUG] Using generated_code as response: {response_length} characters")
                    elif final_state.get('advisor_output'):
                        result_obj = final_state['advisor_output']
                        response = extract_content_from_result(result_obj)
                        response_length = safe_len(response)
                        print(f"[DEBUG] Using advisor_output as response: {response_length} characters")
                    elif final_state.get('scope'):
                        result_obj = final_state['scope']
                        response = extract_content_from_result(result_obj)
                        response_length = safe_len(response)
                        print(f"[DEBUG] Using scope as response: {response_length} characters")
                    else:
                        print("[WARNING] No generated content found in final state")
                        print(f"[DEBUG] Final state keys: {list(final_state.keys()) if final_state else 'None'}")
                        
            except Exception as e:
                print(f"[ERROR] Exception in agentic flow: {str(e)}")
                print("[ERROR] Traceback:", traceback.format_exc())
                raise
                
        else:
            print("[DEBUG] Processing continuation message")
            write_to_log(f"Processing continuation for thread {request.thread_id}")
            
            try:
                # Créer un état initial avec le message de continuation
                initial_state = {
                    "next_user_message": request.message,
                    "latest_user_message": "",
                    "messages": [],
                    "scope": "",
                    "advisor_output": "",
                    "file_list": [],
                    "refined_prompt": "",
                    "refined_tools": "",
                    "refined_agent": "",
                    "config": config
                }
                
                final_state = None
                async for msg in agentic_flow.astream(
                    initial_state,
                    config,
                    stream_mode="values"
                ):
                    print(f"[DEBUG] Continuation: Received state update from agentic flow")
                    if isinstance(msg, dict):
                        final_state = msg
                
                # Extraire la réponse finale de l'état
                if final_state:
                    # Priorité à generated_code, puis advisor_output, puis scope
                    if final_state.get('generated_code'):
                        response = extract_content_from_result(final_state['generated_code'])
                    elif final_state.get('advisor_output'):
                        response = extract_content_from_result(final_state['advisor_output'])
                    elif final_state.get('scope'):
                        response = extract_content_from_result(final_state['scope'])
            except Exception as e:
                print(f"[ERROR] Exception in continuation flow: {str(e)}")
                print("[ERROR] Traceback:", traceback.format_exc())
                raise

        response_length = safe_len(response)
        print(f"[DEBUG] Final response length: {response_length} characters")
        
        # Créer un aperçu sécurisé de la réponse pour les logs
        response_str = str(response)
        if response_length > 200:
            response_preview = response_str[:200] + "..."
        else:
            response_preview = response_str
            
        write_to_log(f"Final response for thread {request.thread_id}: {response_preview}")
        
        # Vérifier que la réponse n'est pas vide et la formater correctement
        if not response:
            print("[WARNING] Empty response from agentic flow")
            response = "Désolé, je n'ai pas pu générer de réponse. Veuillez réessayer."
        
        # Si la réponse est un dictionnaire (comme une réponse Ollama brute), essayer d'extraire le contenu
        if isinstance(response, dict):
            print(f"[DEBUG] Raw response is a dictionary, extracting content. Keys: {list(response.keys())}")
            # Essayer d'extraire le contenu de différentes manières selon le format de la réponse
            if 'content' in response:
                response = response['content']
            elif 'response' in response:
                response = response['response']
            elif 'message' in response and isinstance(response['message'], dict) and 'content' in response['message']:
                response = response['message']['content']
            else:
                # Si on ne peut pas extraire de contenu, convertir en JSON pour l'affichage
                response = json.dumps(response, indent=2)
        
        # S'assurer que la réponse est une chaîne de caractères
        if not isinstance(response, str):
            try:
                response = str(response)
            except Exception as e:
                print(f"[ERROR] Failed to convert response to string: {e}")
                response = "Désolé, une erreur est survenue lors du formatage de la réponse."
            
        return {"response": response}
        
    except Exception as e:
        error_msg = f"Exception invoking Archon for thread {request.thread_id}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        print("[ERROR] Traceback:", traceback.format_exc())
        write_to_log(f"Error processing message for thread {request.thread_id}: {str(e)}")
        
        # Ajouter plus de détails à l'erreur pour le débogage
        error_detail = {
            "error": str(e),
            "type": type(e).__name__,
            "thread_id": request.thread_id,
            "is_first_message": request.is_first_message if hasattr(request, 'is_first_message') else None,
            "message_length": safe_len(request.message) if hasattr(request, 'message') else 0
        }
        
        # Si c'est une erreur de validation, ajouter les détails de validation
        if hasattr(e, 'errors') and callable(e.errors):
            # Convertir les erreurs en un format sérialisable
            validation_errors = []
            for error in e.errors():
                if isinstance(error, dict):
                    validation_errors.append({
                        'type': str(error.get('type', '')),
                        'loc': [str(loc) for loc in error.get('loc', [])],
                        'msg': str(error.get('msg', '')),
                        'input': str(error.get('input', ''))[:100] + '...' if isinstance(error.get('input'), (bytes, bytearray)) else error.get('input')
                    })
                else:
                    validation_errors.append(str(error))
            error_detail["validation_errors"] = validation_errors
            
        # S'assurer que tous les champs sont sérialisables
        for key, value in error_detail.items():
            if isinstance(value, (bytes, bytearray)):
                error_detail[key] = value.decode('utf-8', errors='replace')
            elif hasattr(value, '__dict__'):
                error_detail[key] = str(value)
                
        raise HTTPException(
            status_code=500, 
            detail=error_detail
        )

@app.post('/test')
async def test_provider(request: InvokeRequest):
    """Endpoint dédié aux tests de fournisseurs"""
    try:
        # Logique de test simplifiée
        test_prompt = f"Test de connexion avec le fournisseur: {request.message}"
        
        # Configuration complète pour le graphe
        inputs = {
            "messages": [{"content": test_prompt, "type": "human"}],
            "thread_id": request.thread_id,
            "configurable": {
                "thread_id": request.thread_id,
                "checkpoint_ns": "test",
                "checkpoint_id": f"test-{datetime.now().isoformat()}"
            }
        }
        
        # Exécution du test
        result = await agentic_flow.ainvoke(inputs)
        
        # Sérialisation correcte
        return {
            "status": "success",
            "provider": "openrouter" if "openrouter" in os.environ.get("LLM_PROVIDER", "").lower() else "ollama",
            "response": str(result)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)

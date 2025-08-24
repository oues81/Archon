from typing import Any, Dict, List, Optional, AsyncIterator, Union, cast, AsyncGenerator
from contextlib import asynccontextmanager
import aiohttp
import asyncio
import json
import os
import httpx  # pour compatibilité avec les tests qui patchent httpx.AsyncClient

# Importations compatibles avec pydantic-ai 0.0.22 (avec garde)
try:
    from pydantic_ai import models  # type: ignore
    from pydantic_ai.models import Model, ModelMessage, ModelSettings, ModelResponse  # type: ignore
    from pydantic_ai.messages import SystemPromptPart, UserPromptPart, ToolReturnPart  # type: ignore
except ImportError:
    # Shims minimales pour exécuter les tests sans pydantic_ai installé
    class Model:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            pass
    class ModelMessage(dict):  # type: ignore
        pass
    class ModelSettings:  # type: ignore
        pass
    class ModelResponse(dict):  # type: ignore
        pass
    class _BaseMsgPart:  # type: ignore
        def __init__(self, text: str = "") -> None:
            self.text = text
    class SystemPromptPart(_BaseMsgPart):  # type: ignore
        pass
    class UserPromptPart(_BaseMsgPart):  # type: ignore
        pass
    class ToolReturnPart(_BaseMsgPart):  # type: ignore
        pass


class OllamaModel(Model):
    """Modèle Ollama pour pydantic-ai."""
    
    def __init__(self, model_name: str, base_url: str = None, **kwargs):
        # Initialiser les attributs avant d'appeler le constructeur parent
        self._model_name = model_name
        self._base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
        self._client = OllamaClient(self._base_url)
        # Exposer base_url pour les tests
        self.base_url = self._base_url
        
        # Appeler le constructeur parent sans arguments
        super().__init__()
        
        # Mettre à jour les attributs après l'initialisation du parent
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def model_name(self) -> str:
        """Retourne le nom du modèle."""
        return self._model_name
    
    @property
    def system(self) -> str:
        """Retourne le nom du système (fournisseur)."""
        return "Ollama"
    
    @property
    def name(self) -> str:
        """Retourne le nom du modèle."""
        return self._model_name

    @property
    def base_url_property(self) -> str:
        """Retourne l'URL de base (compat interne)."""
        return self._base_url
    
    @property
    def agent_model(self):
        """Retourne un objet avec une méthode name() pour la compatibilité avec pydantic-ai."""
        class ModelWrapper:
            def __init__(self, model_name):
                self._model_name = model_name
            
            def name(self):
                return self._model_name
                
        return ModelWrapper(self._model_name)

    async def ainvoke(self, prompt: str, **kwargs) -> str:
        """Méthode pour être compatible avec les appels directs."""
        messages = [{'role': 'user', 'content': prompt}]
        response = await self.request(messages, **kwargs)
        return response.get('content', '')

    async def request(
        self,
        messages: List[Union[ModelMessage, Dict[str, Any]]],
        model_settings: Optional[ModelSettings] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Envoie une requête au modèle Ollama et retourne la réponse.
        
        Args:
            messages: Liste des messages de la conversation
            model_settings: Paramètres du modèle
            **kwargs: Arguments supplémentaires pour la requête
            
        Returns:
            La réponse du modèle
        """
        # Convertir les messages au format attendu par Ollama
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, SystemPromptPart):
                ollama_messages.append({"role": "system", "content": msg.text})
            elif isinstance(msg, UserPromptPart):
                ollama_messages.append({"role": "user", "content": msg.text})
            elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                # Gestion des messages déjà au bon format
                ollama_messages.append({"role": msg['role'], "content": msg['content']})
        
        # Préparer les paramètres de la requête
        params = self._prepare_generation_parameters(model_settings, **kwargs)
        
        # Effectuer la requête
        response = await self._client.chat(
            messages=ollama_messages,
            model=self._model_name,
            **params
        )
        
        # Extraire le contenu de la réponse
        content = response.get("message", {}).get("content", "")
        
        # Créer une réponse standardisée avec les champs requis par Pydantic AI
        response = {
            "content": content,
            "model": self._model_name,
            "usage": {
                "prompt_tokens": len(str(messages)),  # Estimation grossière
                "completion_tokens": len(content.split()) if content else 0,
                "total_tokens": len(str(messages)) + (len(content.split()) if content else 0)
            },
            "object": "chat.completion",
            "choices": [{
                "message": {
                    "content": content,
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }]
        }
        
        return response

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> str:
        """Interface simple de génération pour compatibilité tests.

        Utilise httpx.AsyncClient pour permettre le patch dans les tests.
        Retourne directement le contenu de la réponse.
        """
        url = f"{self._base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }
        headers = {"Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            # Supporte la structure de mock des tests
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    @asynccontextmanager
    async def request_stream(
        self,
        messages: List[Union[ModelMessage, Dict[str, Any]]],
        model_settings: Optional[Union[ModelSettings, Dict[str, Any]]] = None,
        model_request_parameters: Optional[Any] = None,
        **kwargs
    ) -> AsyncIterator[AsyncIterator[Dict[str, Any]]]:
        """
        Envoie une requête en streaming au modèle Ollama.
        
        Args:
            messages: Liste des messages de la conversation
            model_settings: Paramètres du modèle (temperature, max_tokens, etc.)
            model_request_parameters: Paramètres spécifiques à la requête (peut être un objet ou un dict)
            **kwargs: Arguments supplémentaires passés à l'API Ollama
            
        Yields:
            Un itérateur asynchrone de chunks de réponse formatés
        """
        # Convertir les messages au format attendu par Ollama
        ollama_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                ollama_messages.append({
                    'role': msg.get('role', 'user'),
                    'content': msg.get('content', '')
                })
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                ollama_messages.append({
                    'role': msg.role,
                    'content': msg.content
                })
        
        # Préparer les paramètres de la requête
        params = {}
        
        # Ajouter les paramètres du modèle s'ils sont fournis
        if model_settings is not None:
            if isinstance(model_settings, dict):
                params.update(model_settings)
            else:
                # Convertir l'objet ModelSettings en dict
                model_settings_dict = {}
                for field in model_settings.__fields__:
                    value = getattr(model_settings, field, None)
                    if value is not None:
                        model_settings_dict[field] = value
                params.update(model_settings_dict)
        
        # Ajouter les paramètres de requête spécifiques s'ils sont fournis
        if model_request_parameters is not None:
            if hasattr(model_request_parameters, 'dict'):
                # Si c'est un objet Pydantic avec une méthode dict()
                params.update(model_request_parameters.dict())
            elif hasattr(model_request_parameters, '__dict__'):
                # Si c'est un objet avec __dict__
                params.update(vars(model_request_parameters))
            elif isinstance(model_request_parameters, dict):
                # Si c'est déjà un dictionnaire
                params.update(model_request_parameters)
            
        # Ajouter les autres paramètres
        params.update(kwargs)
        
        # Créer un générateur asynchrone pour le streaming
        async def stream_generator():
            full_content = ""
            try:
                async for chunk in self._client.chat_stream(
                    messages=ollama_messages,
                    model=self._model_name,
                    **{k: v for k, v in params.items() if v is not None}
                ):
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        full_content += content
                        # Créer un objet de réponse avec l'attribut usage
                        response = {
                            "id": f"chatcmpl-{str(uuid.uuid4())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": self._model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": content,
                                    "role": "assistant"
                                },
                                "finish_reason": None if not chunk.get("done") else "stop"
                            }],
                            "usage": {
                                "prompt_tokens": len(str(messages)),
                                "completion_tokens": len(full_content.split()),
                                "total_tokens": len(str(messages)) + len(full_content.split())
                            }
                        }
                        yield response
            except Exception as e:
                print(f"Error in stream: {str(e)}")
                raise
        
        # Retourner le générateur dans un gestionnaire de contexte
        try:
            yield stream_generator()
        except Exception as e:
            print(f"Error in request_stream: {str(e)}")
            raise
    
    def _prepare_generation_parameters(
        self,
        model_settings: Optional[ModelSettings] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extrait les paramètres de génération des paramètres du modèle.
        
        Args:
            model_settings: Paramètres du modèle (optionnel)
            **kwargs: Arguments supplémentaires pour la requête
            
        Returns:
            Dictionnaire des paramètres formatés pour l'API Ollama
        """
        # Paramètres de base
        params: Dict[str, Any] = {
            "stream": False,
            "options": {
                "num_ctx": 1024,
                "temperature": 0.7,
                "num_predict": 256
            }
        }
        
        # Mettre à jour avec les paramètres du modèle s'ils sont définis
        if model_settings:
            if hasattr(model_settings, 'temperature') and model_settings.temperature is not None:
                params["options"]["temperature"] = model_settings.temperature
            if hasattr(model_settings, 'max_tokens') and model_settings.max_tokens is not None:
                params["options"]["num_predict"] = model_settings.max_tokens
            if hasattr(model_settings, 'top_p') and model_settings.top_p is not None:
                params["options"]["top_p"] = model_settings.top_p
            if hasattr(model_settings, 'top_k') and model_settings.top_k is not None:
                params["options"]["top_k"] = model_settings.top_k
        
        # Mettre à jour avec les paramètres supplémentaires
        if "options" in kwargs:
            params["options"].update(kwargs.pop("options"))
        
        # Ajouter les autres paramètres passés via **kwargs
        params.update(kwargs)
        
        return params
        
    async def close(self):
        """Ferme proprement le client Ollama."""
        if hasattr(self, '_client') and self._client:
            await self._client.close()


class OllamaClient:
    """Client asynchrone pour interagir avec l'API Ollama."""
    
    def __init__(self, base_url: str = "http://host.docker.internal:11434"):
        self.base_url = base_url.rstrip('/')
        self._session = None
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Retourne une session HTTP, en en créant une nouvelle si nécessaire."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
        
    @property
    def session(self) -> aiohttp.ClientSession:
        """
        Retourne la session HTTP existante ou en crée une nouvelle.
        ATTENTION: Cette méthode doit être utilisée uniquement dans un contexte où
        get_session() a déjà été appelé auparavant dans une tâche asyncio.
        """
        if self._session is None:
            # Créer une session par défaut, mais cela devrait être évité
            # et get_session() devrait être utilisé à la place
            loop = asyncio.get_event_loop()
            self._session = loop.run_until_complete(asyncio.ensure_future(aiohttp.ClientSession()))
        return self._session
    
    async def close(self):
        """Ferme la session HTTP."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _make_chat_request(self, session: aiohttp.ClientSession, url: str, payload: dict) -> Dict[str, Any]:
        """Effectue la requête HTTP vers l'API Ollama."""
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            if payload.get('stream', False):
                return await self._handle_streaming_response(response)
            return await response.json()
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Effectue une conversation avec un modèle Ollama.
        
        Args:
            messages: Liste des messages de la conversation
            model: Nom du modèle Ollama à utiliser
            stream: Si True, active le mode streaming
            **kwargs: Arguments supplémentaires pour la requête
            
        Returns:
            La réponse du modèle au format JSON
            
        Raises:
            TimeoutError: Si la requête dépasse le délai imparti
            aiohttp.ClientError: En cas d'erreur de requête HTTP
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        # Obtenir la session HTTP
        session = await self.get_session()
        
        try:
            # Créer une tâche pour la requête HTTP
            request_task = asyncio.create_task(
                self._make_chat_request(session, url, payload)
            )
            
            # Attendre la fin de la tâche avec un timeout
            return await asyncio.wait_for(request_task, timeout=300)  # 5 minutes de timeout
            
        except asyncio.TimeoutError:
            # Annuler la tâche en cas de timeout
            if 'request_task' in locals() and not request_task.done():
                request_task.cancel()
            raise TimeoutError("La requête a expiré après 5 minutes")
            
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Erreur lors de la requête à l'API Ollama: {str(e)}")
            
        except Exception as e:
            raise RuntimeError(f"Erreur inattendue: {str(e)}")
    
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Effectue une conversation en streaming avec un modèle Ollama.
        
        Args:
            messages: Liste des messages de la conversation
            model: Nom du modèle Ollama à utiliser
            **kwargs: Arguments supplémentaires pour la requête
            
        Yields:
            Des morceaux de la réponse du modèle au fur et à mesure qu'ils sont reçus
            
        Raises:
            TimeoutError: Si la requête dépasse le délai imparti
            aiohttp.ClientError: En cas d'erreur de requête HTTP
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        session = await self.get_session()
        
        # Fonction pour gérer le streaming de la réponse
        async def stream_response():
            try:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    
                    # Lire les données en streaming avec un timeout
                    start_time = asyncio.get_event_loop().time()
                    timeout = 300  # 5 minutes en secondes
                    
                    async for line in response.content:
                        # Vérifier le temps écoulé
                        elapsed = asyncio.get_event_loop().time() - start_time
                        if elapsed > timeout:
                            raise asyncio.TimeoutError("La requête en streaming a expiré après 5 minutes")
                            
                        if line:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])  # Enlève 'data: ' du début
                                    yield data
                                except json.JSONDecodeError:
                                    continue
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"Erreur lors du streaming: {str(e)}")
                raise
        
        # Créer une tâche pour le streaming
        stream_task = asyncio.create_task(stream_response().__anext__())
        
        try:
            # Attendre les résultats avec un timeout global
            while True:
                try:
                    # Utiliser un timeout court pour permettre la vérification périodique
                    item = await asyncio.wait_for(stream_task, timeout=5)
                    yield item
                    # Préparer la prochaine itération
                    stream_task = asyncio.create_task(stream_response().__anext__())
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as e:
                    # Vérifier si la tâche est terminée avec une erreur
                    if stream_task.done():
                        try:
                            await stream_task  # Pour lever l'erreur si elle existe
                        except StopAsyncIteration:
                            break
                        except Exception as e:
                            raise
                    # Sinon, continuer à attendre
                    continue
        except asyncio.CancelledError:
            # Annuler la tâche en cours si elle existe
            if not stream_task.done():
                stream_task.cancel()
                try:
                    await stream_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
            raise
        finally:
            # Nettoyer la tâche si elle existe toujours
            if not stream_task.done():
                stream_task.cancel()
                try:
                    await stream_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
    
    async def _handle_streaming_response(self, response) -> Dict[str, Any]:
        """Traite une réponse en streaming et renvoie un objet de réponse unifié."""
        full_content = ""
        async for line in response.content:
            if line:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])  # Enlève 'data: ' du début
                        if 'message' in data and 'content' in data['message']:
                            full_content += data['message']['content']
                    except json.JSONDecodeError:
                        continue
        
        return {
            "model": self.model_name if hasattr(self, 'model_name') else "",
            "message": {
                "role": "assistant",
                "content": full_content
            },
            "done": True
        }

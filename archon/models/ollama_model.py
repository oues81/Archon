from typing import Any, Dict, List, Optional, AsyncIterator, Union, cast, AsyncGenerator
from contextlib import asynccontextmanager
import aiohttp
import asyncio
import json
import os
from datetime import datetime
import time
import uuid
import logging

# Importations compatibles avec pydantic-ai
from pydantic_ai.models import Model, ModelMessage, ModelSettings, ModelResponse
from pydantic_ai.usage import Usage
from pydantic_ai.messages import TextPart

# Configuration Ollama
from archon.config.ollama_config import get_ollama_config, update_ollama_config

# Configuration du logger
logger = logging.getLogger(__name__)


class OllamaModel(Model):
    """Modèle Ollama pour pydantic-ai."""
    
    def __init__(self, model_name: str, base_url: str = None, **kwargs):
        # Initialiser les attributs avant d'appeler le constructeur parent
        self._model_name = model_name
        
        # Récupérer la configuration Ollama
        ollama_config = get_ollama_config()
        
        # Utiliser l'URL fournie, celle de l'environnement, ou celle détectée automatiquement
        self._base_url = base_url or os.getenv("OLLAMA_BASE_URL") or ollama_config["base_url"]
        self._timeout = float(kwargs.get("timeout", ollama_config["timeout"]))
        self._max_retries = int(kwargs.get("max_retries", ollama_config["max_retries"]))
        self._verify_ssl = kwargs.get("verify_ssl", ollama_config["verify_ssl"])
        
        logger.info(f"Initialisation du modèle Ollama avec l'URL: {self._base_url}")
        logger.debug(f"Configuration Ollama - Timeout: {self._timeout}s, Max retries: {self._max_retries}")
        self._client = OllamaClient(self._base_url)
        
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
        model_request_parameters = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Envoie une requête au modèle Ollama et retourne la réponse.
        
        Args:
            messages: Liste des messages de la conversation
            model_settings: Paramètres du modèle
            model_request_parameters: Paramètres spécifiques à la requête
            **kwargs: Arguments supplémentaires pour la requête
            
        Returns:
            La réponse du modèle
        """
        # Convertir les messages au format attendu par Ollama
        ollama_messages = []
        for msg in messages:
            if hasattr(msg, 'parts'):
                # Handle ModelMessage with parts
                for part in msg.parts:
                    if hasattr(part, 'content'):
                        role = "user"
                        if hasattr(part, 'part_kind'):
                            if 'system' in part.part_kind:
                                role = "system"
                            elif 'assistant' in part.part_kind or 'text' in part.part_kind:
                                role = "assistant"
                        ollama_messages.append({"role": role, "content": part.content})
            elif hasattr(msg, 'content'):
                ollama_messages.append({"role": "user", "content": msg.content})
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
        
        # Créer un objet Usage approprié
        prompt_tokens = len(str(messages))  # Estimation grossière
        completion_tokens = len(content.split()) if content else 0
        usage = Usage(
            requests=1,
            request_tokens=prompt_tokens,
            response_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        
        # Créer une réponse ModelResponse avec les champs requis par Pydantic AI
        model_response = ModelResponse(
            parts=[TextPart(content)],
            model_name=self._model_name,
            usage=usage
        )
        
        return model_response
    
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
                        # Créer un objet Usage approprié
                        prompt_tokens = len(str(messages))
                        completion_tokens = len(full_content.split())
                        usage = Usage(
                            requests=1,
                            request_tokens=prompt_tokens,
                            response_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens
                        )
                        
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
                            "usage": usage
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
    
    def __init__(self, base_url: str = None):
        # Récupérer la configuration Ollama
        ollama_config = get_ollama_config()
        
        # Utiliser l'URL fournie, celle de l'environnement, ou celle détectée automatiquement
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or 
                        ollama_config["base_url"]).rstrip('/')
        self._timeout = float(ollama_config.get("timeout", 30.0))
        self._max_retries = int(ollama_config.get("max_retries", 3))
        self._verify_ssl = bool(ollama_config.get("verify_ssl", False))
        self._session = None
        
        logger.info(f"Initialisation du client Ollama avec l'URL: {self.base_url}")
        logger.debug(f"Configuration - Timeout: {self._timeout}s, Max retries: {self._max_retries}")
    
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
    
    async def _make_chat_request(self, session: aiohttp.ClientSession, url: str, payload: dict):
        """Effectue la requête HTTP vers l'API Ollama avec gestion des erreurs améliorée."""
        last_error = None
        
        for attempt in range(self._max_retries + 1):
            try:
                logger.debug(f"Tentative {attempt + 1}/{self._max_retries + 1} - Envoi à {url}")
                
                timeout = aiohttp.ClientTimeout(total=self._timeout)
                
                async with session.post(
                    url, 
                    json=payload, 
                    timeout=timeout,
                    ssl=None if not self._verify_ssl else True
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Erreur HTTP {response.status} - {error_text}")
                        
                        # Si c'est une erreur 404, vérifier l'URL de base
                        if response.status == 404 and attempt == 0:
                            # Essayer de détecter automatiquement la bonne URL
                            from archon.config.ollama_config import detect_ollama_url
                            new_url = f"{detect_ollama_url()}/api/chat"
                            if new_url != url:
                                logger.warning(f"Tentative avec la nouvelle URL: {new_url}")
                                url = new_url
                                continue
                        
                        raise aiohttp.ClientError(
                            f"Erreur HTTP {response.status}: {error_text}"
                        )
                    
                    return await response.json()
                    
            except asyncio.TimeoutError as e:
                last_error = TimeoutError(f"La requête a expiré après {self._timeout} secondes")
                logger.warning(f"Timeout lors de la tentative {attempt + 1}")
                
            except aiohttp.ClientError as e:
                last_error = e
                logger.error(f"Erreur réseau: {str(e)}")
                
            # Attendre avant de réessayer (avec backoff exponentiel)
            if attempt < self._max_retries:
                wait_time = min(2 ** attempt, 10)  # Maximum 10 secondes
                logger.debug(f"Nouvelle tentative dans {wait_time} secondes...")
                await asyncio.sleep(wait_time)
        
        # Si on arrive ici, toutes les tentatives ont échoué
        raise last_error or Exception("Échec inconnu lors de la requête vers Ollama")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Effectue une conversation avec un modèle Ollama.
        
        Args:
            messages: Liste des messages de la conversation
            model: Nom du modèle Ollama à utiliser
            stream: Si True, active le mode streaming
            **kwargs: Arguments supplémentaires pour la requête
                - temperature: Contrôle le caractère aléatoire (0-1)
                - top_p: Filtre de noyau (nucleus sampling)
                - max_tokens: Nombre maximum de tokens à générer
                - num_ctx: Taille du contexte (en tokens)
                - num_predict: Nombre maximum de tokens à prédire
                
        Returns:
            La réponse du modèle au format JSON ou un itérateur de chunks en mode streaming
            
        Raises:
            TimeoutError: Si la requête dépasse le délai imparti
            aiohttp.ClientError: En cas d'erreur de requête HTTP
            RuntimeError: Pour les autres erreurs inattendues
        """
        # Construire l'URL de l'API
        url = f"{self.base_url}/api/chat"
        
        # Préparer les options du modèle
        options = {
            "temperature": kwargs.pop("temperature", 0.7),
            "top_p": kwargs.pop("top_p", 0.9),
            "num_predict": kwargs.pop("max_tokens", kwargs.pop("num_predict", 2000)),
            "num_ctx": kwargs.pop("num_ctx", 2048),
        }
        
        # Filtrer les options None
        options = {k: v for k, v in options.items() if v is not None}
        
        # Construire le payload de la requête
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options,
            **kwargs
        }
        
        logger.debug(f"Envoi de la requête à {url} avec le modèle {model}")
        logger.debug(f"Options: {options}")
        
        # Obtenir la session HTTP
        session = await self.get_session()
        
        # Gérer le mode streaming
        if stream:
            return self.chat_stream(messages, model, **kwargs)
            
        # Mode normal (non-streaming)
        try:
            # Créer une tâche pour la requête HTTP avec un timeout
            request_task = asyncio.create_task(
                self._make_chat_request(session, url, payload)
            )
            
            # Attendre la fin de la tâche avec un timeout
            response = await asyncio.wait_for(request_task, timeout=self._timeout)
            
            # Vérifier la réponse
            if not response or 'message' not in response:
                logger.error(f"Réponse inattendue de l'API Ollama: {response}")
                raise RuntimeError("Réponse inattendue de l'API Ollama")
                
            return response
            
        except asyncio.TimeoutError:
            # Annuler la tâche en cas de timeout
            if 'request_task' in locals() and not request_task.done():
                request_task.cancel()
            raise TimeoutError(f"La requête a expiré après {self._timeout} secondes")
            
        except aiohttp.ClientError as e:
            logger.error(f"Erreur client HTTP: {str(e)}")
            raise RuntimeError(f"Erreur lors de la communication avec l'API Ollama: {str(e)}")
            
        except Exception as e:
            logger.error(f"Erreur inattendue: {str(e)}", exc_info=True)
            raise RuntimeError(f"Erreur inattendue lors de l'appel à l'API Ollama: {str(e)}")
    
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
                - temperature: Contrôle le caractère aléatoire (0-1)
                - top_p: Filtre de noyau (nucleus sampling)
                - max_tokens: Nombre maximum de tokens à générer
                - num_ctx: Taille du contexte (en tokens)
                - num_predict: Nombre maximum de tokens à prédire
            
        Yields:
            Des morceaux de la réponse du modèle au format:
            {
                "role": "assistant",
                "content": "contenu du message",
                "done": False  # True pour le dernier chunk
            }
            
        Raises:
            TimeoutError: Si la requête dépasse le délai imparti
            RuntimeError: En cas d'erreur de communication ou autre erreur inattendue
        """
        # Construire l'URL de l'API
        url = f"{self.base_url}/api/chat"
        
        # Préparer les options du modèle
        options = {
            "temperature": kwargs.pop("temperature", 0.7),
            "top_p": kwargs.pop("top_p", 0.9),
            "num_predict": kwargs.pop("max_tokens", kwargs.pop("num_predict", 2000)),
            "num_ctx": kwargs.pop("num_ctx", 2048),
        }
        
        # Filtrer les options None
        options = {k: v for k, v in options.items() if v is not None}
        
        # Construire le payload de la requête
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,  # Important pour le streaming
            "options": options,
            **kwargs
        }
        
        logger.debug(f"Début du streaming vers {url} avec le modèle {model}")
        logger.debug(f"Options: {options}")
        
        # Obtenir la session HTTP
        session = await self.get_session()
        
        try:
            # Créer un timeout pour la connexion initiale
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            
            async with session.post(
                url,
                json=payload,
                timeout=timeout,
                ssl=None if not self._verify_ssl else True
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Erreur HTTP {response.status} - {error_text}")
                    raise RuntimeError(f"Erreur HTTP {response.status}: {error_text}")
                
                # Lire le flux de réponse ligne par ligne
                buffer = ""
                async for chunk in response.content.iter_chunked(1024):
                    # Décoder le chunk en texte
                    chunk_text = chunk.decode('utf-8')
                    buffer += chunk_text
                    
                    # Traiter toutes les lignes complètes dans le buffer
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                            
                        try:
                            # Décoder la ligne JSON
                            data = json.loads(line)
                            
                            # Vérifier si c'est un chunk de données valide
                            if 'message' in data and 'content' in data['message']:
                                yield {
                                    'role': 'assistant',
                                    'content': data['message']['content'],
                                    'done': data.get('done', False)
                                }
                            
                            # Si c'est le dernier chunk, on sort de la boucle
                            if data.get('done', False):
                                return
                                
                        except json.JSONDecodeError as e:
                            logger.warning(f"Erreur de décodage JSON: {line}")
                            continue
                        except Exception as e:
                            logger.error(f"Erreur lors du traitement du chunk: {str(e)}", exc_info=True)
                            continue
                            
        except asyncio.TimeoutError:
            logger.error(f"La requête a dépassé le timeout de {self._timeout} secondes")
            raise TimeoutError(f"La requête a expiré après {self._timeout} secondes")
            
        except aiohttp.ClientError as e:
            logger.error(f"Erreur client HTTP: {str(e)}")
            raise RuntimeError(f"Erreur lors de la communication avec l'API Ollama: {str(e)}")
            
        except Exception as e:
            logger.error(f"Erreur inattendue lors du streaming: {str(e)}", exc_info=True)
            raise RuntimeError(f"Erreur inattendue lors du streaming: {str(e)}")
    
    async def _handle_streaming_response(self, response) -> Dict[str, Any]:
        """
        Traite une réponse en streaming et renvoie un objet de réponse unifié.
        
        Args:
            response: La réponse HTTP de l'API Ollama
            
        Returns:
            Un dictionnaire contenant la réponse complète du modèle et les métadonnées
            
        Raises:
            RuntimeError: Si une erreur survient lors du traitement de la réponse
        """
        full_content = ""
        model_name = getattr(self, 'model_name', "")
        
        try:
            # Lire le contenu de la réponse
            buffer = ""
            async for chunk in response.content.iter_chunked(1024):
                # Décoder le chunk en texte
                chunk_text = chunk.decode('utf-8')
                buffer += chunk_text
                
                # Traiter toutes les lignes complètes dans le buffer
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        # Décoder la ligne JSON
                        data = json.loads(line)
                        
                        # Extraire le contenu du message si disponible
                        if 'message' in data and 'content' in data['message']:
                            full_content += data['message']['content']
                            
                        # Mettre à jour le nom du modèle s'il est fourni
                        if 'model' in data and not model_name:
                            model_name = data['model']
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Erreur de décodage JSON: {line}")
                        continue
                        
        except Exception as e:
            logger.error(f"Erreur lors de la lecture de la réponse: {str(e)}", exc_info=True)
            raise RuntimeError(f"Erreur lors du traitement de la réponse: {str(e)}")
        
        # Calculer le nombre de tokens (estimation approximative)
        token_count = len(full_content.split()) if full_content else 0
        
        # Retourner la réponse unifiée
        return {
            "model": model_name,
            "message": {
                "role": "assistant",
                "content": full_content
            },
            "usage": {
                "requests": 1,
                "request_tokens": 0,  # Impossible de connaître le nombre exact sans compter les tokens
                "response_tokens": token_count,
                "total_tokens": token_count
            },
            "done": True
        }

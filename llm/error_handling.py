"""
Module de gestion standardisée des erreurs LLM pour Archon.
Fournit des wrappers unifiés pour tous les appels LLM avec:
- Retry avec backoff exponentiel
- Gestion des timeouts
- Circuit breaker
- Métriques standardisées
"""
import asyncio
import time
import logging
import random
import httpx
import inspect
import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast
from enum import Enum
from pydantic import BaseModel

import logfire

logger = logging.getLogger(__name__)

# Type générique pour la fonction à wrapper
T = TypeVar('T')
AsyncCallable = Callable[..., asyncio.coroutine]

class LLMErrorType(str, Enum):
    """Types d'erreurs LLM standardisés"""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit" 
    INVALID_REQUEST = "invalid_request"
    CONTEXT_WINDOW = "context_window"
    API_ERROR = "api_error"
    CONTENT_FILTER = "content_filter"
    CONNECTION = "connection"
    UNKNOWN = "unknown"


class LLMCallMetrics(BaseModel):
    """Métriques standardisées pour les appels LLM"""
    model: str
    provider: str
    start_time: float
    end_time: float
    duration_ms: float
    attempt_count: int
    success: bool
    error_type: Optional[LLMErrorType] = None
    request_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None


class LLMError(Exception):
    """Exception standardisée pour les erreurs LLM"""
    def __init__(
        self, 
        message: str, 
        error_type: LLMErrorType = LLMErrorType.UNKNOWN,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        original_exception: Optional[Exception] = None,
        retry_after: Optional[float] = None
    ):
        self.error_type = error_type
        self.provider = provider
        self.model = model
        self.original_exception = original_exception
        self.retry_after = retry_after
        super().__init__(message)


def classify_llm_error(error: Exception, provider: str) -> Tuple[LLMErrorType, Optional[float]]:
    """
    Classifie une exception en type d'erreur LLM standardisé.
    
    Args:
        error: Exception originale
        provider: Nom du provider LLM
        
    Returns:
        Tuple de (type_erreur, retry_after_secondes)
    """
    error_msg = str(error).lower()
    retry_after = None
    
    # Timeouts
    if isinstance(error, asyncio.TimeoutError) or isinstance(error, httpx.TimeoutException):
        return LLMErrorType.TIMEOUT, None
        
    # Connection errors
    if isinstance(error, httpx.ConnectError) or "connection" in error_msg:
        return LLMErrorType.CONNECTION, None
        
    # Classification basée sur le provider
    if provider == "openai":
        if "rate limit" in error_msg:
            # Extrait le retry-after s'il existe
            try:
                if hasattr(error, "headers") and error.headers.get("retry-after"):
                    retry_after = float(error.headers.get("retry-after", 60))
            except (AttributeError, ValueError):
                retry_after = 60
            return LLMErrorType.RATE_LIMIT, retry_after
            
        if "invalid request" in error_msg:
            return LLMErrorType.INVALID_REQUEST, None
            
        if "context window" in error_msg or "maximum token" in error_msg:
            return LLMErrorType.CONTEXT_WINDOW, None
            
        if "content filter" in error_msg or "content policy" in error_msg:
            return LLMErrorType.CONTENT_FILTER, None
            
    elif provider == "anthropic":
        if "rate_limit" in error_msg:
            retry_after = 30  # Default for Anthropic
            return LLMErrorType.RATE_LIMIT, retry_after
            
    elif provider == "openrouter":
        if "rate limit" in error_msg or "too many requests" in error_msg:
            return LLMErrorType.RATE_LIMIT, 60  # Default for OpenRouter
    
    # Ollama spécifique
    elif provider == "ollama":
        if "context window" in error_msg:
            return LLMErrorType.CONTEXT_WINDOW, None
    
    # Par défaut, on considère une erreur API générique
    return LLMErrorType.API_ERROR, None


async def with_llm_error_handling(
    func: AsyncCallable,
    *args: Any,
    provider: str,
    model: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout_s: int = 60,
    jitter: bool = True,
    **kwargs: Any
) -> Tuple[Any, LLMCallMetrics]:
    """
    Wrapper pour gérer les erreurs LLM avec retry et métriques standardisées.
    
    Args:
        func: Fonction asynchrone à appeler (généralement un appel LLM)
        *args: Arguments de la fonction
        provider: Nom du fournisseur LLM
        model: Nom du modèle utilisé
        max_retries: Nombre maximum de tentatives
        base_delay: Délai initial pour le backoff exponentiel
        timeout_s: Timeout en secondes
        jitter: Ajouter du bruit aléatoire au délai de retry
        **kwargs: Arguments nommés à passer à la fonction
        
    Returns:
        Tuple de (résultat_fonction, métriques)
        
    Raises:
        LLMError: Si toutes les tentatives échouent
    """
    metrics = LLMCallMetrics(
        model=model,
        provider=provider,
        start_time=time.time(),
        end_time=0.0,
        duration_ms=0.0,
        attempt_count=0,
        success=False
    )
    
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        metrics.attempt_count = attempt
        
        # Ajouter un log structuré pour chaque tentative
        logger.debug(
            f"LLM call attempt {attempt}/{max_retries}",
            extra={"provider": provider, "model": model, "attempt": attempt}
        )
        
        start_time = time.time()
        
        try:
            # Créer une tâche avec timeout
            task = asyncio.create_task(func(*args, **kwargs))
            result = await asyncio.wait_for(task, timeout=timeout_s)
            
            # Succès - mettre à jour les métriques
            end_time = time.time()
            metrics.end_time = end_time
            metrics.duration_ms = (end_time - metrics.start_time) * 1000
            metrics.success = True
            
            # Log structuré de succès
            logfire.info(
                "LLM call successful",
                provider=provider,
                model=model,
                duration_ms=metrics.duration_ms,
                attempts=attempt
            )
            
            return result, metrics
            
        except Exception as e:
            end_time = time.time()
            
            # Classifier l'erreur
            error_type, retry_after = classify_llm_error(e, provider)
            last_error = LLMError(
                message=f"LLM error: {str(e)}",
                error_type=error_type,
                provider=provider,
                model=model,
                original_exception=e,
                retry_after=retry_after
            )
            
            # Si c'est la dernière tentative, on ne fait pas de retry
            if attempt >= max_retries:
                break
                
            # Calcul du délai de retry (backoff exponentiel)
            delay = retry_after or base_delay * (2 ** (attempt - 1))
            
            # Ajouter un jitter pour éviter la synchronisation des retries
            if jitter:
                delay *= 0.8 + 0.4 * random.random()  # ±20% de variation
                
            # Log structuré de l'erreur
            logfire.warn(
                "LLM call error, retrying",
                provider=provider,
                model=model,
                error_type=error_type.value,
                attempt=attempt,
                max_retries=max_retries,
                delay=delay,
                error=str(e)
            )
            
            # Attendre avant de réessayer
            await asyncio.sleep(delay)
    
    # Toutes les tentatives ont échoué - mettre à jour les métriques finales
    metrics.end_time = time.time()
    metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
    metrics.success = False
    metrics.error_type = last_error.error_type if last_error else LLMErrorType.UNKNOWN
    
    # Log structuré d'échec final
    logfire.error(
        "LLM call failed after all retries",
        provider=provider,
        model=model,
        error_type=metrics.error_type.value if metrics.error_type else "unknown",
        attempts=metrics.attempt_count,
        duration_ms=metrics.duration_ms
    )
    
    if last_error:
        raise last_error
        
    # Fallback au cas où nous n'aurions pas d'erreur spécifique
    raise LLMError(
        message="LLM call failed after all retries with unknown error",
        error_type=LLMErrorType.UNKNOWN,
        provider=provider,
        model=model
    )


def llm_error_handler(
    provider: str,
    model: str = "unknown",
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout_s: int = 60,
    jitter: bool = True
):
    """
    Décorateur pour gérer les erreurs LLM avec retry et métriques.
    
    Args:
        provider: Nom du fournisseur LLM
        model: Nom du modèle utilisé
        max_retries: Nombre maximum de tentatives
        base_delay: Délai initial pour le backoff exponentiel
        timeout_s: Timeout en secondes
        jitter: Ajouter du bruit aléatoire au délai de retry
        
    Returns:
        Fonction décorée
        
    Example:
        ```python
        @llm_error_handler(provider="openai", model="gpt-4")
        async def call_completion_api(prompt):
            return await openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
        ```
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # On peut surcharger les paramètres par défaut avec des kwargs
            actual_provider = kwargs.pop("provider", provider)
            actual_model = kwargs.pop("model", model)
            actual_max_retries = kwargs.pop("max_retries", max_retries)
            actual_base_delay = kwargs.pop("base_delay", base_delay)
            actual_timeout_s = kwargs.pop("timeout_s", timeout_s)
            actual_jitter = kwargs.pop("jitter", jitter)
            
            return await with_llm_error_handling(
                func, 
                *args, 
                provider=actual_provider,
                model=actual_model,
                max_retries=actual_max_retries,
                base_delay=actual_base_delay,
                timeout_s=actual_timeout_s,
                jitter=actual_jitter,
                **kwargs
            )
        return wrapper
    return decorator


# Circuit Breaker pour LLM
class LLMCircuitBreaker:
    """
    Implémentation d'un circuit breaker pour les appels LLM.
    Maintient une fenêtre d'erreurs et ouvre le circuit si trop d'erreurs se produisent.
    """
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        half_open_timeout: float = 5.0
    ):
        """
        Initialiser le circuit breaker.
        
        Args:
            failure_threshold: Nombre d'erreurs avant ouverture du circuit
            reset_timeout: Délai avant réinitialisation du compteur d'erreurs (secondes)
            half_open_timeout: Délai avant passage en état semi-ouvert (secondes)
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open
        self.last_state_change = time.time()
    
    def record_failure(self):
        """Enregistrer un échec et potentiellement ouvrir le circuit"""
        now = time.time()
        
        # Réinitialiser le compteur si trop de temps s'est écoulé depuis la dernière erreur
        if now - self.last_failure_time > self.reset_timeout:
            self.failures = 0
        
        self.failures += 1
        self.last_failure_time = now
        
        # Changer l'état si nécessaire
        if self.state == "closed" and self.failures >= self.failure_threshold:
            self.state = "open"
            self.last_state_change = now
            logger.warning(f"Circuit breaker opened after {self.failures} failures")
    
    def record_success(self):
        """Enregistrer un succès et fermer le circuit si en état semi-ouvert"""
        if self.state == "half-open":
            self.state = "closed"
            self.failures = 0
            self.last_state_change = time.time()
            logger.info("Circuit breaker closed after successful test request")
    
    def allow_request(self) -> bool:
        """
        Vérifier si une requête est autorisée selon l'état actuel du circuit.
        
        Returns:
            True si la requête est autorisée, False sinon
        """
        now = time.time()
        
        # En état fermé, toutes les requêtes sont autorisées
        if self.state == "closed":
            return True
            
        # En état ouvert, vérifier si on peut passer en état semi-ouvert
        if self.state == "open":
            if now - self.last_state_change > self.half_open_timeout:
                self.state = "half-open"
                self.last_state_change = now
                logger.info("Circuit breaker half-open, testing with next request")
                return True
            return False
            
        # En état semi-ouvert, autoriser une seule requête de test
        if self.state == "half-open":
            return True
            
        return False

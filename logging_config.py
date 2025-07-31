import logging
import logging.handlers
import os
from pathlib import Path

# Configuration des répertoires
LOG_DIR = Path("/app/logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Formateur de log standard
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Niveaux de log par défaut
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
UVICORN_LOG_LEVEL = os.getenv("UVICORN_LOG_LEVEL", "info").lower()

class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

class WarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.WARNING

def setup_logger(name, log_file, level=logging.INFO):
    """Configure un logger avec rotation des fichiers"""
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    # Créer le gestionnaire de fichiers avec rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    # Configurer le logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Éviter les doublons de handlers
    if not logger.handlers:
        logger.addHandler(file_handler)
    
    return logger

def configure_root_logger():
    """Configure le logger racine avec des handlers pour différents niveaux"""
    # Créer les répertoires de logs si nécessaire
    LOG_DIR.mkdir(exist_ok=True, parents=True)
    
    # Configurer le logger racine
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    
    # Supprimer les handlers existants
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Formatters
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    # Handler pour la console (niveaux WARNING et supérieurs)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(formatter)
    
    # Handler pour le fichier de log principal
    main_log = LOG_DIR / "archon.log"
    file_handler = logging.handlers.RotatingFileHandler(
        main_log,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    
    # Ajouter les handlers
    root_logger.addHandler(console)
    root_logger.addHandler(file_handler)

def get_uvicorn_log_config():
    """Retourne la configuration de logging pour Uvicorn"""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(asctime)s - %(levelprefix)s %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
                "use_colors": False,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(asctime)s - %(client_addr)s - "%(request_line)s" %(status_code)s',
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(LOG_DIR / "uvicorn.log"),
                "maxBytes": 10*1024*1024,  # 10MB
                "backupCount": 5,
                "formatter": "default",
                "encoding": "utf-8"
            },
            "access_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(LOG_DIR / "access.log"),
                "maxBytes": 10*1024*1024,  # 10MB
                "backupCount": 3,
                "formatter": "access",
                "encoding": "utf-8"
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default", "file"], "level": UVICORN_LOG_LEVEL},
            "uvicorn.error": {"level": UVICORN_LOG_LEVEL},
            "uvicorn.access": {"handlers": ["access", "access_file"], "level": "INFO", "propagate": False},
        },
    }

# Configuration initiale
configure_root_logger()

# Exporter la configuration pour Uvicorn
uvicorn_log_config = get_uvicorn_log_config()

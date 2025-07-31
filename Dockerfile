# Étape 1: Builder stage pour les dépendances de compilation
FROM python:3.10-slim-bookworm as builder

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Désactivation CUDA/GPU
    CUDA_VISIBLE_DEVICES=-1 \
    USE_CUDA=0 \
    USE_CUDNN=0 \
    USE_MKLDNN=0 \
    USE_NCCL=0

# Installation des dépendances système minimales
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installation de Poetry
ENV POETRY_VERSION=1.8.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VIRTUALENVS_CREATE=false
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Installer Poetry de manière plus rapide
RUN pip install --no-cache-dir poetry==$POETRY_VERSION

WORKDIR /app

# Copie des fichiers de dépendances
COPY pyproject.toml poetry.lock ./

# Installation des dépendances avec Poetry (sans les dev pour la production)
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry install --no-interaction --no-ansi --without dev && \
    # Installation de PyTorch CPU avec pip pour plus de rapidité
    pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Étape 2: Image finale
FROM python:3.10-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Configuration de la journalisation
    LOG_LEVEL=INFO \
    UVICORN_LOG_LEVEL=info \
    # Désactivation des logs inutiles
    TOKENIZERS_PARALLELISM=false \
    # Configuration du fuseau horaire
    TZ=Europe/Paris \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    # Désactivation CUDA/GPU
    CUDA_VISIBLE_DEVICES=-1 \
    USE_CUDA=0 \
    USE_CUDNN=0 \
    USE_MKLDNN=0 \
    USE_NCCL=0

# Installation des dépendances système minimales
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libxml2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie de l'environnement Python depuis le builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copie des fichiers de configuration et des scripts
COPY logging_config.py .
COPY start_services.sh .
COPY cleanup_logs.sh .

# Rendre les scripts exécutables
RUN chmod +x start_services.sh cleanup_logs.sh

# Créer le répertoire de logs
RUN mkdir -p /app/logs

# Copier le reste des fichiers
COPY . .

# Installation de Playwright et dépendances en une seule couche
RUN --mount=type=cache,target=/root/.cache \
    pip install --no-cache-dir playwright tf-playwright-stealth && \
    python -m playwright install --with-deps chromium && \
    python -m playwright install-deps

# Téléchargement des modèles NLTK nécessaires
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Copie sélective des fichiers nécessaires
COPY . .

# Rendre le script exécutable
RUN chmod +x /app/start_services.sh

# Exposition des ports
EXPOSE 5000
EXPOSE 8501

# Réduction de la taille de l'image finale
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Commande de démarrage
CMD ["/app/start_services.sh"]
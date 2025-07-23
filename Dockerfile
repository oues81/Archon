# Étape 1: Builder stage pour les dépendances de compilation
FROM python:3.10-slim as builder

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
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    procps \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installation de Poetry
RUN pip install --no-cache-dir poetry==1.8.2

# Copie des fichiers de dépendances
COPY pyproject.toml poetry.lock ./

# Installation des dépendances
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root && \
    # Installation de PyTorch CPU uniquement
    pip install --no-cache-dir torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Installation des dépendances supplémentaires
RUN pip install --no-cache-dir uvicorn streamlit

# Étape 2: Image finale
FROM python:3.10-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
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
    libxslt1.1 \
    libjpeg62-turbo \
    zlib1g \
    # Dépêndances pour Playwright
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie de l'environnement Python depuis le builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Installation de Playwright avec uniquement Chromium
RUN pip install --no-cache-dir playwright==1.40.0 && \
    playwright install --with-deps chromium && \
    playwright install chromium && \
    # Nettoyage
    rm -rf /root/.cache/pip

# Installation de NLTK
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Copie du code source
COPY . .

# Rendre le script exécutable
RUN chmod +x /app/start_services.sh

# Installation en mode développement
RUN pip install --no-cache-dir -e .

# Ports exposés
EXPOSE 8110 8501

# Commande de démarrage
CMD ["/app/start_services.sh"]
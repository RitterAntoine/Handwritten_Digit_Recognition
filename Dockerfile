# ==========================================
# ÉTAPE 1 : Construction (Builder)
# ==========================================
# J'utilise une image Python "slim" pour commencer, plus légère.
FROM python:3.10-slim AS builder

# Définir des variables pour éviter des problèmes pendant l'installation
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Installer les dépendances système nécessaires pour compiler certains paquets Python (ex: numpy)
# et les supprimer après pour gagner de la place.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Créer un environnement virtuel pour isoler les dépendances.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# IMPORTANT : Ton dépôt Git DOIT contenir un fichier requirements.txt.
# Il devrait inclure au minimum : flask, gunicorn, tensorflow-cpu (ou pytorch-cpu), numpy, pillow.
# J'utilise les versions CPU des librairies d'IA pour éviter de télécharger des gigaoctets de pilotes GPU inutiles sur ton Dell OptiPlex.
COPY requirements.txt .

# Installer les dépendances dans l'environnement virtuel.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# ÉTAPE 2 : Image Finale (Production)
# ==========================================
FROM python:3.10-slim

WORKDIR /app

# Récupérer l'environnement virtuel construit à l'étape 1.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Définir l'environnement Flask pour la production.
ENV FLASK_ENV=production \
    CUDA_VISIBLE_DEVICES=-1

# Copier le code source de l'application.
# Ton compose.yaml montre que tu montes ./mnist_app:/app,
# mais pour le build, Docker va copier le contenu de ton repo.
# Assure-toi d'avoir un dossier 'app' contenant 'main.py' à la racine de ton repo.
COPY . .

# Créer un utilisateur non-root pour la sécurité (Gunicorn ne devrait pas tourner en root).
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Le port par défaut que ton application écoute.
EXPOSE 5000

# La commande finale est déjà définie dans ton compose.yaml, donc
# ENTRYPOINT/CMD ne sont pas strictement nécessaires ici, mais
# c'est une bonne pratique de les inclure pour la documentation.
# CMD ["gunicorn", "--chdir", "app", "-w", "1", "-b", "0.0.0.0:5000", "main:app"]

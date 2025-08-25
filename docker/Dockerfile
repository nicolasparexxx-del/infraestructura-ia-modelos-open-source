# Dockerfile para la API de Modelos de IA
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de requisitos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY api/ ./api/
COPY config/ ./config/

# Crear directorio para modelos
RUN mkdir -p /app/models

# Configurar variables de entorno
ENV PYTHONPATH=/app
ENV MODEL_CACHE_PATH=/app/models
ENV HF_HOME=/app/models/.cache

# Exponer puerto
EXPOSE 8000

# Comando por defecto
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

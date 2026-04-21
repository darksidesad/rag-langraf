FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema necesarias para pypdf y compilaciones
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Crear directorio para documentos subidos
RUN mkdir -p /app/docs

# Puerto del servidor FastAPI
EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8080/health'); assert r.status_code == 200" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

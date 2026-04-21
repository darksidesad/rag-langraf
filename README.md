# RAG LangGraph - Agente Autónomo MTTR

Agente autónomo de soporte técnico con RAG (Retrieval Augmented Generation) para reducir el MTTR en infraestructuras críticas.

## Arquitectura

| Componente | Tecnología | Puerto |
|-----------|-----------|--------|
| **Frontend** | Astro | 4321 |
| **Backend** | FastAPI + LangGraph | 8080 |
| **Vectorstore** | Qdrant | 6333 |
| **Observabilidad** | MLflow | 5000 |
| **LLM (Chat)** | Ollama Cloud | - |
| **Embeddings** | Google Gemini (gemini-embedding-001) | - |

## Funcionalidades

- 🤖 **Chat RAG** — Consulta manuales técnicos con modelos de Ollama Cloud
- 📄 **Subida de documentos** — PDF, DOCX, TXT, MD indexados automáticamente en Qdrant
- 🧪 **Verificación de alucinaciones** — LLM Juez que valida respuestas contra documentos
- 📊 **Métricas en tiempo real** — Scores de documentos, tiempos del pipeline, resultado del juez
- ⚙️ **API Key configurable** — Cada usuario puede usar su propia API key de Ollama Cloud
- 📏 **Umbral de distancia** — Control de qué tan similar deben ser los documentos
- 🔄 **Selección dinámica de modelos** — Carga automática desde Ollama Cloud

## Despliegue con Docker

```bash
# Clonar el repositorio
git clone https://github.com/darksidesad/rag-langraf.git
cd rag-langraf

# Configurar variables de entorno
cp backend/.env.example backend/.env
# Editar backend/.env con tus API keys

# Levantar todo
docker-compose up -d
```

## Desarrollo Local

```bash
# Backend
cd backend
python -m venv env
.\env\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
cp .env.example .env
# Editar .env con tus keys
python main.py

# Frontend (otra terminal)
cd frontend
npm install
npm run dev
```

## Variables de Entorno

| Variable | Descripción | Requerida |
|----------|------------|-----------|
| `QDRANT_URL` | URL de Qdrant | ✅ |
| `QDRANT_API_KEY` | API Key de Qdrant | ❌ |
| `OLLAMA_BASE_URL` | URL de Ollama Cloud | ✅ |
| `OLLAMA_API_KEY` | API Key de Ollama Cloud | ✅ |
| `GOOGLE_API_KEY` | API Key de Google Gemini (embeddings) | ✅ |
| `MLFLOW_TRACKING_URI` | URI de MLflow (vacío = desactivado) | ❌ |
| `CORS_ORIGINS` | Orígenes permitidos | ❌ |

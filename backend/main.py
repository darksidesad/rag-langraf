import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

# Activar auto-tracking de MLflow ANTES de importar LangGraph/LangChain
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
if mlflow_uri:
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("Agente_MTTR")
        mlflow.langchain.autolog()
        print("✅ MLflow LangChain Autolog activado.")
    except ImportError:
        print("El modulo MLflow no esta instalado en este entorno local. Funcionara en produccion.")
    except Exception as e:
        print(f"⚠️ MLflow no disponible (no crítico): {e}")
else:
    print("ℹ️ MLflow desactivado (MLFLOW_TRACKING_URI vacío).")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from graph import (
    app as graph_workflow,
    get_qdrant_raw_client,
    COLLECTION_NAME,
    DEFAULT_MAX_DISTANCE,
    OLLAMA_API_KEY as SERVER_OLLAMA_API_KEY,
)
from ingest import load_documents, chunk_documents, index_in_qdrant, DOCS_DIR
import uvicorn

app = FastAPI(title="Soporte Técnico API", version="2.0.0")

# CORS
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración Ollama Cloud
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")

# --- MODELOS DE REQUEST ---

class QueryRequest(BaseModel):
    question: str
    model_name: str = "gemma3:27b"
    api_key: Optional[str] = None           # API key del usuario (override)
    max_distance: Optional[float] = None    # Umbral de distancia personalizado

class SettingsRequest(BaseModel):
    api_key: str

# Tipos de archivo permitidos y tamaño máximo (10 MB)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_SIZE_MB = 10

# --- ENDPOINTS ---

@app.get("/models")
async def list_models(api_key: Optional[str] = None):
    """Lista modelos desde Ollama Cloud. Acepta API key por query param."""
    effective_key = api_key or OLLAMA_API_KEY
    api_url = OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
    headers = {}
    if effective_key:
        headers["Authorization"] = f"Bearer {effective_key}"
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(api_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model in data.get("models", []):
                name = model.get("name", "")
                display_name = name.replace(":latest", "")
                size_gb = round(model.get("size", 0) / (1024**3), 1)
                models.append({
                    "id": name,
                    "name": display_name,
                    "size_gb": size_gb,
                    "family": model.get("details", {}).get("family", "unknown"),
                    "parameter_size": model.get("details", {}).get("parameter_size", ""),
                })
            
            return {"models": models, "count": len(models)}
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Ollama API error: {e.response.text}")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"No se pudo conectar a Ollama en {OLLAMA_BASE_URL}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo modelos: {str(e)}")

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """Endpoint principal de consulta con métricas detalladas."""
    try:
        effective_key = request.api_key or OLLAMA_API_KEY
        effective_distance = request.max_distance if request.max_distance is not None else DEFAULT_MAX_DISTANCE

        inputs = {
            "question": request.question,
            "model_name": request.model_name,
            "api_key": effective_key,
            "max_distance": effective_distance,
        }
        output = graph_workflow.invoke(inputs)
        
        return {
            "question": output.get("question"),
            "response": output.get("generation"),
            "escalated": output.get("escalate", False),
            # Métricas del pipeline RAG
            "metrics": {
                "doc_scores": output.get("doc_scores", []),
                "total_docs_found": output.get("total_docs_found", 0),
                "docs_accepted": output.get("docs_accepted", 0),
                "hallucination_result": output.get("hallucination_result", "skipped"),
                "hallucination_detail": output.get("hallucination_detail", ""),
                "retrieve_time_ms": output.get("retrieve_time_ms", 0),
                "generate_time_ms": output.get("generate_time_ms", 0),
                "hallucination_time_ms": output.get("hallucination_time_ms", 0),
                "model_used": request.model_name,
                "distance_threshold": effective_distance,
            }
        }
    except Exception as e:
        print(f"❌ Error en /ask: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando la consulta: {str(e)}")

@app.get("/documents")
async def list_documents():
    """Lista todos los documentos/puntos indexados en la colección de Qdrant."""
    try:
        client = get_qdrant_raw_client()
        
        # Verificar si la colección existe
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if COLLECTION_NAME not in collection_names:
            return {"documents": [], "count": 0, "collection_exists": False}
        
        # Obtener info de la colección
        collection_info = client.get_collection(COLLECTION_NAME)
        
        # Obtener los primeros 100 puntos con sus metadatos
        points = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        
        documents = []
        seen_sources = {}
        for point in points[0]:
            payload = point.payload or {}
            metadata = payload.get("metadata", {})
            source = metadata.get("source", "desconocido")
            # Limpiar path
            if os.sep in source or "/" in source:
                source = os.path.basename(source)
            
            content = payload.get("page_content", "")
            snippet = content[:200] + "..." if len(content) > 200 else content
            
            if source not in seen_sources:
                seen_sources[source] = {"source": source, "chunks": 0, "snippets": []}
            seen_sources[source]["chunks"] += 1
            if len(seen_sources[source]["snippets"]) < 2:
                seen_sources[source]["snippets"].append(snippet)
        
        return {
            "documents": list(seen_sources.values()),
            "total_chunks": collection_info.points_count,
            "collection_exists": True,
            "vector_size": collection_info.config.params.vectors.size if hasattr(collection_info.config.params.vectors, 'size') else "N/A",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando documentos: {str(e)}")

@app.delete("/documents")
async def delete_all_documents():
    """Elimina toda la colección de documentos."""
    try:
        client = get_qdrant_raw_client()
        client.delete_collection(COLLECTION_NAME)
        return {"status": "success", "message": f"Colección '{COLLECTION_NAME}' eliminada."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando colección: {str(e)}")

@app.post("/validate-key")
async def validate_api_key(settings: SettingsRequest):
    """Valida una API key de Ollama Cloud haciendo una petición de prueba."""
    api_url = OLLAMA_BASE_URL.rstrip("/") + "/api/tags"
    headers = {"Authorization": f"Bearer {settings.api_key}"}
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(api_url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                model_count = len(data.get("models", []))
                return {"valid": True, "models_available": model_count}
            else:
                return {"valid": False, "error": f"Status {response.status_code}: {response.text[:200]}"}
    except Exception as e:
        return {"valid": False, "error": str(e)}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Tipo no permitido: {file_ext}. Permitidos: {', '.join(ALLOWED_EXTENSIONS)}")
    
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"El archivo excede {MAX_FILE_SIZE_MB} MB.")
    
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        
    file_path = os.path.join(DOCS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(contents)
        
    try:
        docs = load_documents(DOCS_DIR)
        chunks = chunk_documents(docs)
        index_in_qdrant(chunks)
        return {"status": "success", "message": f"Archivo {file.filename} procesado e indexado.", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexando: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "ollama_url": OLLAMA_BASE_URL,
        "ollama_cloud": bool(OLLAMA_API_KEY),
        "qdrant_url": os.environ.get("QDRANT_URL", "not configured"),
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

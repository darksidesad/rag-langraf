import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

load_dotenv()

# Configuración básica
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
COLLECTION_NAME = "manuales_tecnicos"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")

def parse_qdrant_url(url: str):
    """Parsea la URL de Qdrant en host, port, https para compatibilidad con EasyPanel."""
    parsed = urlparse(url)
    use_https = parsed.scheme == "https"
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if use_https else 6333)
    return host, port, use_https

def get_base_url():
    """Construye la URL compatible con la API de OpenAI para Ollama."""
    url = OLLAMA_BASE_URL
    if not url.endswith("/v1"):
        url = url.rstrip("/") + "/v1"
    return url

def get_embeddings():
    """
    Usa Google Gemini embeddings (gemini-embedding-001).
    Más estable y rápido que embeddings locales.
    IMPORTANTE: Esta función DEBE ser idéntica en graph.py e ingest.py.
    """
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
    )

def load_documents(docs_dir: str):
    documents = []
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"Directorio creado en: {docs_dir}. Por favor añade manuales aquí.")
        return documents

    for file in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
        elif file.endswith(".md") or file.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
    
    return documents

def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(docs)

def index_in_qdrant(chunks):
    if not chunks:
        print("No hay documentos para indexar.")
        return
    
    embeddings = get_embeddings()
    host, port, use_https = parse_qdrant_url(QDRANT_URL)
    
    client = QdrantClient(
        host=host,
        port=port,
        https=use_https,
        api_key=QDRANT_API_KEY,
    )
    
    print(f"Indexando {len(chunks)} fragmentos en Qdrant ({host}:{port}, https={use_https})...")
    
    qdrant = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        location=f"{'https' if use_https else 'http'}://{host}:{port}",
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=True,
    )
    print("✅ Indexación completada.")

if __name__ == "__main__":
    print("Iniciando proceso de ingesta...")
    loaded_docs = load_documents(DOCS_DIR)
    if loaded_docs:
        print(f"Se cargaron {len(loaded_docs)} páginas/documentos en total.")
        chunks = chunk_documents(loaded_docs)
        index_in_qdrant(chunks)
    else:
        print("Finalizado sin cambios. Añade archivos en la carpeta docs/")

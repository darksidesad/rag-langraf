from typing import TypedDict, List, Optional
from urllib.parse import urlparse
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
import time
from dotenv import load_dotenv
load_dotenv()

# Configuraciones
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

# Umbral de DISTANCIA máxima configurable desde el frontend
# Gemini embeddings producen distancias coseno más altas (~0.5-0.8 para docs relevantes)
DEFAULT_MAX_DISTANCE = 0.85

class GraphState(TypedDict):
    question: str
    model_name: str
    generation: str
    documents: List[Document]
    escalate: bool
    # Campos de diagnóstico/métricas
    api_key: str
    max_distance: float
    doc_scores: list          # [{source, score, accepted, snippet}]
    hallucination_result: str # "grounded", "hallucinated", "skipped"
    hallucination_detail: str
    retrieve_time_ms: float
    generate_time_ms: float
    hallucination_time_ms: float
    total_docs_found: int
    docs_accepted: int

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

def get_qdrant_raw_client():
    """Retorna el QdrantClient nativo (para operaciones directas como listar puntos)."""
    host, port, use_https = parse_qdrant_url(QDRANT_URL)
    return QdrantClient(
        host=host,
        port=port,
        https=use_https,
        api_key=QDRANT_API_KEY,
    )

def get_qdrant_vectorstore():
    """Conecta a la colección existente en Qdrant usando langchain-qdrant."""
    embeddings = get_embeddings()
    client = get_qdrant_raw_client()
    return QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
    )

def retrieve(state: GraphState):
    print("---RETRIEVE---")
    start = time.time()
    question = state["question"]
    max_distance = state.get("max_distance", DEFAULT_MAX_DISTANCE)
    
    qdrant = get_qdrant_vectorstore()
    docs_and_scores = qdrant.similarity_search_with_score(question, k=4)
    
    filtered_docs = [doc for doc, score in docs_and_scores if score <= max_distance]
    
    # Recopilar métricas de cada documento
    doc_scores = []
    for doc, score in docs_and_scores:
        accepted = score <= max_distance
        source = doc.metadata.get("source", "desconocido")
        # Limpiar el path para mostrar solo el nombre del archivo
        if os.sep in source:
            source = os.path.basename(source)
        doc_scores.append({
            "source": source,
            "score": round(score, 4),
            "accepted": accepted,
            "snippet": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
        })
    
    elapsed = (time.time() - start) * 1000
    print(f"  Documentos encontrados: {len(docs_and_scores)}, después del filtro: {len(filtered_docs)} ({elapsed:.0f}ms)")
    for ds in doc_scores:
        print(f"  - Score: {ds['score']:.4f} | {'✅' if ds['accepted'] else '❌'} | {ds['source']}")
    
    return {
        "documents": filtered_docs,
        "question": question,
        "model_name": state.get("model_name", "gemma3:27b"),
        "doc_scores": doc_scores,
        "retrieve_time_ms": round(elapsed, 1),
        "total_docs_found": len(docs_and_scores),
        "docs_accepted": len(filtered_docs),
    }

def grade_documents(state: GraphState):
    print("---GRADE DOCUMENTS---")
    if not state["documents"]:
        print("  No hay documentos relevantes. Escalando a soporte humano.")
        return {"escalate": True}
    print(f"  {len(state['documents'])} documentos relevantes encontrados.")
    return {"escalate": False, "documents": state["documents"]}

def generate(state: GraphState):
    print("---GENERATE---")
    start = time.time()
    question = state["question"]
    documents = state["documents"]
    model_name = state.get("model_name", "gemma3:27b")
    api_key = state.get("api_key", OLLAMA_API_KEY)
    
    context = "\n\n".join([doc.page_content for doc in documents])
    
    llm = ChatOpenAI(
        api_key=api_key if api_key else "ollama",
        base_url=get_base_url(),
        model=model_name,
        temperature=0
    )
    
    prompt = PromptTemplate(
        template="""Eres un asistente experto en soporte técnico. Contesta basado en los manuales de contexto.
        Si la información en los manuales no es suficiente para dar una respuesta segura, indícalo claramente.
        
        Pregunta: {question}
        Manuales: {context}
        
        Respuesta:""",
        input_variables=["question", "context"],
    )
    
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"question": question, "context": context})
    
    elapsed = (time.time() - start) * 1000
    print(f"  Generación completada ({elapsed:.0f}ms)")
    
    return {"generation": generation, "generate_time_ms": round(elapsed, 1)}

def check_hallucination(state: GraphState):
    """El 'LLM Juez' evalúa si la respuesta está fundamentada en los documentos."""
    print("---CHECK HALLUCINATION---")
    start = time.time()
    generation = state.get("generation", "")
    documents = state.get("documents", [])
    
    if not documents or not generation:
        return {
            "escalate": True,
            "hallucination_result": "skipped",
            "hallucination_detail": "Sin documentos o generación para evaluar.",
            "hallucination_time_ms": 0,
        }
    
    context = "\n\n".join([doc.page_content for doc in documents])
    model_name = state.get("model_name", "gemma3:27b")
    api_key = state.get("api_key", OLLAMA_API_KEY)
    
    llm = ChatOpenAI(
        api_key=api_key if api_key else "ollama",
        base_url=get_base_url(),
        model=model_name,
        temperature=0
    )
    
    grader_prompt = PromptTemplate(
        template="""Eres un evaluador estricto. Determina si la siguiente respuesta está fundamentada
        ÚNICAMENTE en la información contenida en los documentos proporcionados.
        
        Documentos: {context}
        Respuesta generada: {generation}
        
        Responde SOLO con "si" si la respuesta está completamente fundamentada en los documentos,
        o "no" si contiene información inventada o no respaldada.
        
        Evaluación:""",
        input_variables=["context", "generation"],
    )
    
    chain = grader_prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "generation": generation})
    
    is_grounded = "si" in result.strip().lower()
    elapsed = (time.time() - start) * 1000
    
    print(f"  Resultado: {'✅ Fundamentada' if is_grounded else '❌ Posible alucinación'} ({elapsed:.0f}ms)")
    
    if not is_grounded:
        return {
            "escalate": True,
            "generation": "⚠️ La respuesta generada no pudo ser verificada contra los documentos. Escalando a soporte humano...",
            "hallucination_result": "hallucinated",
            "hallucination_detail": f"El juez LLM determinó que la respuesta NO está fundamentada. Evaluación: {result.strip()}",
            "hallucination_time_ms": round(elapsed, 1),
        }
    
    return {
        "escalate": False,
        "hallucination_result": "grounded",
        "hallucination_detail": f"La respuesta está fundamentada en los documentos. Evaluación: {result.strip()}",
        "hallucination_time_ms": round(elapsed, 1),
    }

# --- LÓGICA DE RUTEO ---
def decide_to_generate(state: GraphState):
    if state["escalate"]:
        return "escalate_to_human"
    else:
        return "generate"

def decide_after_hallucination(state: GraphState):
    if state.get("escalate", False):
        return "escalate_to_human"
    else:
        return END

# --- CREACIÓN DEL GRAFO ---
workflow = StateGraph(GraphState)

# Nodos
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("check_hallucination", check_hallucination)
workflow.add_node("escalate_to_human", lambda state: {
    "generation": "No tengo información fiable para responder esto de manera segura. Escalando a soporte humano...",
    "hallucination_result": state.get("hallucination_result", "skipped"),
    "hallucination_detail": state.get("hallucination_detail", "Escalado antes de la verificación de alucinaciones."),
})

# Aristas
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "escalate_to_human": "escalate_to_human"
    }
)
workflow.add_edge("generate", "check_hallucination")
workflow.add_conditional_edges(
    "check_hallucination",
    decide_after_hallucination,
    {
        "escalate_to_human": "escalate_to_human",
        END: END
    }
)
workflow.add_edge("escalate_to_human", END)

# Compilar grafo
app = workflow.compile()

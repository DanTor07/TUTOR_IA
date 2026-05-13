"""
agent/tutor_agent.py
====================
Agente tutor RAG — busqueda semantica + Ollama.

Cambios respecto a version anterior:
  - Retrieval: SemanticStore (ChromaDB + similitud coseno) en lugar de BM25
  - LLM: Ollama (sin cambio)
  - ask() devuelve (str, list[dict]) en lugar de (dict, str)
    para compatibilidad con Streamlit (muestra texto + fuentes)
  - Prompt: texto plano, sin formato JSON obligatorio
"""

import requests
import json
import os
from dotenv import load_dotenv

from rag.semantic_store import SemanticStore
from rag.loader import load_and_chunk_documents
from prompts.system_prompt import get_system_prompt

load_dotenv()

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
TOP_K        = int(os.getenv("TOP_K", 3))


class TutorAgent:
    """
    Agente Tutor con pipeline RAG semantico y generacion via Ollama.

    Flujo:
      1. Busqueda semantica (ChromaDB, similitud coseno, k chunks)
      2. Construccion del prompt aumentado
      3. Generacion con Ollama (texto plano, sin JSON)
      4. Retorno de (respuesta_texto, lista_de_fuentes)
    """

    def __init__(self, ollama_url: str = OLLAMA_URL,
                 model: str = OLLAMA_MODEL,
                 base_kb_dir: str = "knowledge_base"):
        self.ollama_url  = ollama_url
        self.model       = model
        self.base_kb_dir = base_kb_dir
        self.current_collection = None
        self.vector_store       = None

    def load_collection(self, collection_name: str):
        """Carga o inicializa el vector store semantico para la coleccion."""
        self.current_collection = collection_name
        self.vector_store = SemanticStore(collection_name=collection_name)
        if not self.vector_store.chunks:
            print(f"[TutorAgent] '{collection_name}' vacia — indexando...")
            self.refresh_knowledge_base()

    def refresh_knowledge_base(self):
        """Re-indexa todos los documentos de la coleccion activa."""
        if not self.current_collection:
            return
        collection_dir = os.path.join(self.base_kb_dir, self.current_collection)
        os.makedirs(collection_dir, exist_ok=True)
        chunks = load_and_chunk_documents(collection_dir)
        if chunks:
            self.vector_store.fit(chunks)
        else:
            print(f"[TutorAgent] Sin documentos en '{collection_dir}'.")

    def _build_prompt(self, query: str) -> tuple:
        """Construye el prompt aumentado con contexto semantico."""
        sources = self.vector_store.search(query, k=TOP_K) if self.vector_store else []
        if sources:
            context_text = "\n\n".join(
                f"[Fragmento {i+1} — Fuente: {s['source']}]\n{s['text']}"
                for i, s in enumerate(sources)
            )
        else:
            context_text = "No se encontro informacion relevante en la base de conocimientos."

        # Prompt con la coleccion activa en lugar de texto fijo
        system = get_system_prompt(self.current_collection or "base de conocimiento")
        full_prompt = (
            f"{system}\n\n"
            f"--- CONTEXTO RECUPERADO ---\n{context_text}\n\n"
            f"--- PREGUNTA ---\n{query}\n\n"
            f"--- RESPUESTA ---"
        )
        return full_prompt, sources

    def stream_response(self, query: str):
        """
        Generador: emite tokens de Ollama de forma progresiva (streaming).
        Evita el timeout porque cada token reinicia el temporizador de lectura.
        Guarda las fuentes en self._last_sources al terminar.

        Uso en Streamlit: answer = st.write_stream(tutor.stream_response(query))
        """
        self._last_sources = []
        full_prompt, sources = self._build_prompt(query)
        self._last_sources = sources

        payload = {
            "model":  self.model,
            "prompt": full_prompt,
            "stream": True           # tokens progresivos
        }

        try:
            with requests.post(
                self.ollama_url, json=payload,
                stream=True,
                timeout=(30, 600)    # (conexion, entre-tokens)
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break

            print(f"[RAG] {self.current_collection} | {len(sources)} chunks | {query[:60]}")

        except Exception as e:
            err = str(e)
            if any(k in err for k in ("timed out", "Connection refused", "HTTPConnection")):
                yield (
                    "\u26a0\ufe0f **Ollama no responde.**\n\n"
                    "Verifica que est\u00e9 corriendo:\n"
                    "```\nollama serve\n```\n"
                    f"Y que el modelo est\u00e9 disponible:\n"
                    f"```\nollama pull {self.model}\n```"
                )
            else:
                yield f"Error: {err}"
            self._last_sources = []

    def ask(self, query: str) -> tuple:
        """Interfaz sincrona: llama a stream_response y acumula la respuesta."""
        self._last_sources = []
        answer = "".join(self.stream_response(query))
        return answer, self._last_sources

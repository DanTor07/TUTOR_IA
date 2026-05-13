"""
rag/semantic_store.py
=====================
Almacen vectorial semantico basado en ChromaDB + sentence-transformers.
Reemplaza BM25Store con busqueda semantica (similitud coseno).

Fixes:
  - Sanitiza nombres de coleccion para cumplir restriccion ChromaDB
    ([a-zA-Z0-9._-], sin espacios, 3-512 chars)
  - Deshabilita telemetria para evitar conflicto opentelemetry/Streamlit
"""

import os
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from chromadb.config import Settings

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
CHROMA_BASE_DIR = "chroma_db"

# Configuracion ChromaDB: telemetria deshabilitada para evitar
# conflicto entre opentelemetry y el executor de Streamlit
CHROMA_SETTINGS = Settings(anonymized_telemetry=False)

# Singleton del modelo de embeddings
_embeddings_instance = None


def get_embeddings():
    global _embeddings_instance
    if _embeddings_instance is None:
        print(f"[SemanticStore] Cargando embeddings: {EMBEDDING_MODEL}")
        _embeddings_instance = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings_instance


def _sanitize_name(name: str) -> str:
    """
    ChromaDB requiere nombres en [a-zA-Z0-9._-], 3-512 chars,
    iniciando y terminando con [a-zA-Z0-9].
    Ej: 'Machine Learning' -> 'Machine_Learning'
    """
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", name)       # reemplaza invalidos
    safe = re.sub(r"^[^a-zA-Z0-9]+", "", safe)           # limpia inicio
    safe = re.sub(r"[^a-zA-Z0-9]+$", "", safe)           # limpia fin
    if len(safe) < 3:
        safe = safe + "_col"
    return safe[:512]


class SemanticStore:
    """
    Vector store semantico por coleccion.
    Interfaz compatible con BM25Store: fit(chunks) y search(query, k).
    """

    def __init__(self, collection_name: str = "default", db_base_path: str = CHROMA_BASE_DIR):
        self.collection_name  = collection_name
        self.safe_name        = _sanitize_name(collection_name)
        self.persist_dir      = os.path.join(db_base_path, self.safe_name)
        self.embeddings       = get_embeddings()
        self.vectorstore      = None
        self._load_store()

    def _load_store(self):
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=self.safe_name,
                    client_settings=CHROMA_SETTINGS
                )
                count = self.vectorstore._collection.count()
                print(f"[SemanticStore:{self.safe_name}] Cargado ({count} chunks)")
            except Exception as e:
                print(f"[SemanticStore:{self.safe_name}] Error cargando: {e}")
                self.vectorstore = None

    @property
    def chunks(self):
        if self.vectorstore is None:
            return []
        try:
            return list(range(self.vectorstore._collection.count()))
        except Exception:
            return []

    def fit(self, chunks: list):
        if not chunks:
            return
        os.makedirs(self.persist_dir, exist_ok=True)
        documents = [
            Document(
                page_content=chunk["text"],
                metadata={"source": chunk.get("source", ""), "id": chunk.get("id", "")}
            )
            for chunk in chunks
        ]
        print(f"[SemanticStore:{self.safe_name}] Indexando {len(documents)} chunks...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.safe_name,
            client_settings=CHROMA_SETTINGS
        )
        print(f"[SemanticStore:{self.safe_name}] Persistido en {self.persist_dir}")

    def search(self, query: str, k: int = 3) -> list:
        if self.vectorstore is None:
            return []
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        try:
            docs = retriever.invoke(query)
            return [
                {"text": doc.page_content, "source": doc.metadata.get("source", "")}
                for doc in docs
            ]
        except Exception as e:
            print(f"[SemanticStore:{self.safe_name}] Error en busqueda: {e}")
            return []

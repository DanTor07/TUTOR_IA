import requests
import json
import os
from rag.embeddings import get_embeddings
from rag.vector_store import load_vector_store, create_vector_store
from rag.loader import load_documents
from rag.retriever import get_relevant_documents, format_context
from prompts.system_prompt import SYSTEM_PROMPT
from prompts.fewshot_examples import FEW_SHOT_EXAMPLES

class TutorAgent:
    def __init__(self, ollama_url="http://localhost:11434/api/generate", model="phi3"):
        self.ollama_url = ollama_url
        self.model = model
        self.embeddings = get_embeddings()
        self.vector_store = load_vector_store(self.embeddings)
        
        if self.vector_store is None:
            self.refresh_knowledge_base()

    def refresh_knowledge_base(self):
        docs = load_documents("knowledge_base")
        self.vector_store = create_vector_store(docs, self.embeddings)

    def ask(self, query):
        # 1. Recuperar contexto
        relevant_docs = get_relevant_documents(self.vector_store, query)
        context = format_context(relevant_docs)
        
        if not relevant_docs:
            context = "No se encontró información relevante."

        # 2. Construir el prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\n{FEW_SHOT_EXAMPLES}\n\n"
        full_prompt += f'""" CONTEXTO """\n{context}\n\n'
        full_prompt += f'""" PREGUNTA DEL USUARIO """\n{query}\n'
        
        # 3. Llamar a Ollama (local)
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "format": "json"
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()
            answer_text = result.get("response", "{}")
            
            # Intentar parsear como JSON para validar el formato
            answer_json = json.loads(answer_text)
            return answer_json, context
        except Exception as e:
            return {
                "concepto": "Error",
                "explicacion": f"Hubo un error al procesar la solicitud: {str(e)}",
                "ejemplo": "N/A",
                "fuente": "Error de Sistema"
            }, context

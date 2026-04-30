import requests
import json
import os
from rag.bm25_store import BM25Store
from rag.loader import load_and_chunk_documents
from prompts.system_prompt import SYSTEM_PROMPT
from prompts.fewshot_examples import FEW_SHOT_EXAMPLES

class TutorAgent:
    def __init__(self, ollama_url="http://localhost:11434/api/generate", model="phi3", kb_dir="knowledge_base"):
        self.ollama_url = ollama_url
        self.model = model
        self.kb_dir = kb_dir
        self.vector_store = BM25Store()
        
        # Si la base de datos está vacía, refrescar
        if not self.vector_store.chunks:
            self.refresh_knowledge_base()

    def refresh_knowledge_base(self):
        chunks = load_and_chunk_documents(self.kb_dir)
        self.vector_store.fit(chunks)

    def ask(self, query):
        # 1. Recuperar contexto con BM25
        top_chunks = self.vector_store.search(query, k=3)
        
        context_text = ""
        if top_chunks:
            for i, chunk in enumerate(top_chunks):
                context_text += f"--- Documento {i+1} (Fuente: {chunk['source']}) ---\n{chunk['text']}\n\n"
        else:
            context_text = "No se encontró información relevante en la base de conocimientos."
            
        # 2. Construir el prompt
        full_prompt = f"{SYSTEM_PROMPT}\n\n{FEW_SHOT_EXAMPLES}\n\n"
        full_prompt += f'""" CONTEXTO """\n{context_text}\n\n'
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
            return answer_json, context_text
        except Exception as e:
            return {
                "concepto": "Error",
                "explicacion": f"Hubo un error al procesar la solicitud: {str(e)}",
                "ejemplo": "N/A",
                "fuente": "Error de Sistema"
            }, context_text



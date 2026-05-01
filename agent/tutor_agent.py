import requests
import json
import os
from rag.bm25_store import BM25Store
from rag.loader import load_and_chunk_documents
from prompts.system_prompt import SYSTEM_PROMPT
from prompts.fewshot_examples import FEW_SHOT_EXAMPLES

class TutorAgent:
    def __init__(self, ollama_url="http://localhost:11434/api/generate", model="phi3", base_kb_dir="knowledge_base"):
        self.ollama_url = ollama_url
        self.model = model
        self.base_kb_dir = base_kb_dir
        self.current_collection = None
        self.vector_store = None

    def load_collection(self, collection_name):
        self.current_collection = collection_name
        self.vector_store = BM25Store(collection_name=collection_name)
        if not self.vector_store.chunks:
            self.refresh_knowledge_base()

    def refresh_knowledge_base(self):
        if not self.current_collection:
            return
        collection_dir = os.path.join(self.base_kb_dir, self.current_collection)
        if not os.path.exists(collection_dir):
            os.makedirs(collection_dir)
            
        chunks = load_and_chunk_documents(collection_dir)
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
            
            # === LOGS PARA LA TERMINAL (EVIDENCIA DE FUNCIONAMIENTO) ===
            print("\n" + "="*60)
            print("🚀 [RAG SYSTEM LOG] - CONSULTA PROCESADA")
            print("="*60)
            print(f"📁 Colección Activa: {self.current_collection}")
            print("⚙️  Configuración del Sistema:")
            print("   -> System Prompt: Forzando rol de Tutor Experto.")
            print("   -> Restricción: Responder ÚNICAMENTE con la información del contexto.")
            print("   -> Formato de Salida: Objeto JSON estricto.")
            print("\n📄 Contexto recuperado por BM25 (Fragmento):")
            print(context_text[:300] + "..." if len(context_text) > 300 else context_text)
            print("\n🧠 Salida Cruda del LLM (Formato JSON evidenciado):")
            print(answer_text)
            print("="*60 + "\n")
            
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



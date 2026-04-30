import os
import json
import numpy as np

def tokenize(text: str) -> list:
    import re
    # Convertir a minúsculas y extraer solo palabras alfanuméricas
    words = re.findall(r'\b\w+\b', text.lower())
    return words

class BM25Store:
    def __init__(self, db_path="vector_store_data"):
        self.db_path = db_path
        self.db_file = os.path.join(db_path, "bm25_db.json")
        self.chunks = []
        self.vocab_idf = {}
        self.avg_doc_length = 0.0
        self.k1 = 1.5
        self.b = 0.75
        self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.chunks = data.get("chunks", [])
                    self.vocab_idf = data.get("vocab_idf", {})
                    self.avg_doc_length = data.get("avg_doc_length", 0.0)
            except Exception as e:
                print(f"Error cargando la base de datos BM25: {e}")

    def _save_db(self):
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump({
                "chunks": self.chunks,
                "vocab_idf": self.vocab_idf,
                "avg_doc_length": self.avg_doc_length
            }, f, ensure_ascii=False, indent=2)

    def _calculo_idf(self, termino_buscado: str, tokenized_docs: list) -> float:
        N = len(tokenized_docs)
        df = sum(1 for doc_tokens in tokenized_docs if termino_buscado in doc_tokens)
        return np.log(N / df) if df > 0 else 0

    def fit(self, chunks: list):
        self.chunks = chunks
        tokenized_docs = [tokenize(chunk['text']) for chunk in self.chunks]
        
        N = len(tokenized_docs)
        if N == 0:
            return

        total_palabras = sum(len(doc_tokens) for doc_tokens in tokenized_docs)
        self.avg_doc_length = total_palabras / N
        
        # Construir vocabulario y calcular IDF para cada término
        vocabulario = set(word for doc_tokens in tokenized_docs for word in doc_tokens)
        self.vocab_idf = {}
        for term in vocabulario:
            self.vocab_idf[term] = self._calculo_idf(term, tokenized_docs)
            
        self._save_db()

    def _calculo_tf(self, termino_buscado: str, tokenized_doc: list) -> float:
        if len(tokenized_doc) == 0:
            return 0
        return tokenized_doc.count(termino_buscado) / len(tokenized_doc)

    def search(self, query: str, k=3):
        if not self.chunks:
            return []

        query_tokens = tokenize(query)
        scores = []

        for i, chunk in enumerate(self.chunks):
            doc_tokens = tokenize(chunk['text'])
            doc_length = len(doc_tokens)
            bm25_score = 0.0
            
            for q_term in query_tokens:
                if q_term not in self.vocab_idf:
                    continue
                
                idf = self.vocab_idf[q_term]
                tf = self._calculo_tf(q_term, doc_tokens)
                
                # Fórmula BM25
                if doc_length > 0:
                    score = idf * ((tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))))
                    bm25_score += score
            
            if bm25_score > 0:
                scores.append({'chunk': chunk, 'score': bm25_score})

        scores.sort(key=lambda x: x['score'], reverse=True)
        return [item['chunk'] for item in scores[:k]]

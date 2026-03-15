from langchain_community.vectorstores import FAISS
import os

def create_vector_store(documents, embeddings, store_path="vector_store_data"):
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(store_path)
    return vector_store

def load_vector_store(embeddings, store_path="vector_store_data"):
    if os.path.exists(store_path):
        return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    return None

import os

def get_relevant_documents(vector_store, query, k=3):
    if vector_store:
        return vector_store.similarity_search(query, k=k)
    return []

def format_context(documents):
    context_text = ""
    for i, doc in enumerate(documents):
        source = doc.metadata.get('source', 'Unknown')
        context_text += f"--- Documento {i+1} (Fuente: {os.path.basename(source)}) ---\n{doc.page_content}\n\n"
    return context_text

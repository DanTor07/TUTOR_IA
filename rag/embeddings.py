from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    # Using a local model for embeddings via SentenceTransformers
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

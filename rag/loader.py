import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(directory_path):
    documents = []
    for file in os.listdir(directory_path):
        full_path = os.path.join(directory_path, file)
        if file.endswith('.pdf'):
            loader = PyPDFLoader(full_path)
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            loader = TextLoader(full_path)
            documents.extend(loader.load())
        elif file.endswith('.md'):
            loader = UnstructuredMarkdownLoader(full_path)
            documents.extend(loader.load())
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(documents)

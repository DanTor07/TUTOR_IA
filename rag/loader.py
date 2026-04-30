import os
import PyPDF2

def split_text(text: str, chunk_size=500, overlap=50) -> list:
    """Divide el texto en fragmentos (chunks) de longitud aproximada (en palabras)."""
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        i += chunk_size - overlap
        
    return chunks

def load_and_chunk_documents(kb_path="knowledge_base", chunk_size=500, overlap=50):
    """
    Lee todos los documentos (.txt, .md, .pdf) de la carpeta indicada
    y los divide en chunks, retornando una lista de diccionarios.
    """
    if not os.path.exists(kb_path):
        return []

    all_chunks = []
    
    for filename in os.listdir(kb_path):
        file_path = os.path.join(kb_path, filename)
        
        if not os.path.isfile(file_path):
            continue
            
        ext = filename.lower().split('.')[-1]
        content = ""
        
        try:
            if ext in ['txt', 'md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif ext == 'pdf':
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            content += page_text + "\n"
                            
            if content.strip():
                doc_chunks = split_text(content.strip(), chunk_size, overlap)
                for i, chunk_text in enumerate(doc_chunks):
                    all_chunks.append({
                        "id": f"{filename}_chunk_{i}",
                        "source": filename,
                        "text": chunk_text
                    })
        except Exception as e:
            print(f"Error procesando {filename}: {e}")
            
    return all_chunks

import os
import PyPDF2

def get_knowledge_base_context(kb_path="knowledge_base"):
    """
    Lee todos los documentos en la carpeta knowledge_base y
    retorna su contenido como un solo string formateado para el contexto.
    """
    if not os.path.exists(kb_path):
        return "No hay una base de conocimiento configurada."

    context_parts = []
    
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
                context_parts.append(f"--- Documento: {filename} ---\n{content.strip()}\n")
        except Exception as e:
            print(f"Error leyendo {filename}: {e}")
            
    if not context_parts:
        return "No se encontraron documentos legibles en la base de conocimiento."
        
    return "\n".join(context_parts)

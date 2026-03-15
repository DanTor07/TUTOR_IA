# 📚 AI Tutor ML - Asistente Académico RAG Local

**AI Tutor ML** es un ecosistema avanzado de aprendizaje diseñado para asistir a estudiantes en la comprensión de Inteligencia Artificial y Machine Learning. Utiliza una arquitectura **RAG (Retrieval-Augmented Generation)** descentralizada que permite procesar documentos privados de manera local, eliminando la dependencia de APIs en la nube y garantizando la privacidad.

## 🚀 Propósito y Funcionalidades
Este asistente actúa como un experto en la materia que:
-   **Analiza Documentos:** Extrae conocimiento de archivos PDF, Markdown y TXT cargados en su base de conocimiento.
-   **Enseñanza Socrática y Estructurada:** Proporciona definiciones académicas profundas acompañadas de analogías y ejemplos prácticos.
-   **Transparencia de Datos:** Permite visualizar los fragmentos exactos de los textos originales que fundamentan cada respuesta.
-   **Privacidad por Diseño:** Todo el procesamiento (embeddings, búsqueda vectorial e inferencia de lenguaje) ocurre en el hardware local.

## 🏗️ Arquitectura del Sistema
-   **Motor de Recuperación:** Basado en `LangChain` y `FAISS` para búsquedas semánticas ultrarrápidas.
-   **Modelos de Lenguaje:** Conexión nativa con `Ollama` para ejecutar modelos como `Phi-3`, `Mistral` o `Llama`.
-   **Interfaz de Usuario:** Aplicación web moderna con diseño *Glassmorphism* optimizada para la interacción educativa.

---

## 🛠️ Guía de Instalación y Configuración

Siga estos pasos para desplegar el asistente en un entorno local:

### 1. Clonar el Repositorio
```bash
git clone https://github.com/DanTor07/TUTOR_IA.git
cd TUTOR_IA
```

### 2. Preparar el Entorno Virtual (Recomendado)
Es altamente recomendable utilizar un entorno virtual para evitar conflictos de dependencias:

**En Windows (PowerShell):**
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

**En Linux / macOS:**
```bash
python3 -m venv env
source env/bin/activate
```

### 3. Instalar Dependencias
Una vez activado el entorno, instale todos los componentes necesarios:
```bash
pip install -r requirements.txt
```

### 4. Configurar Ollama
1.  Instalar Ollama desde [ollama.com](https://ollama.com).
2.  Asegurar que el servicio esté corriendo en segundo plano.
3.  Descargar el modelo preferido (por defecto configurado para `phi3`):
    ```bash
    ollama run phi3
    ```

---

## 🚀 Ejecución del Asistente
1.  **Cargar Documentos:** Colocar los archivos PDF o TXT de estudio en la carpeta `knowledge_base/`.
2.  **Lanzar el Servidor:**
    ```bash
    python app.py
    ```
3.  **Acceder al Chat:** Abrir el navegador en [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 📂 Estructura de Módulos
-   `rag/`: Implementación del pipeline de datos y búsqueda semántica.
-   `agent/`: Lógica del asistente y comunicación con el LLM local.
-   `prompts/`: Definición de la personalidad experta y ejemplos de entrenamiento.
-   `knowledge_base/`: Repositorio local de documentos de estudio.
-   `templates/`: Interfaz de usuario de alta fidelidad.

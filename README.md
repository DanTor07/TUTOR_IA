# 📚 AI Tutor ML - Asistente Académico RAG Local

Este proyecto es un **Asistente Experto en Machine Learning** que utiliza una arquitectura **RAG (Retrieval-Augmented Generation)** y agentes para funcionar de manera local, garantizando la privacidad de los datos académicos.

## 🎯 Propósito del Asistente
El **Tutor Académico de ML** está diseñado para ayudar a estudiantes universitarios a navegar conceptos complejos como redes neuronales, overfitting y métricas de evaluación. A diferencia de un chat convencional, este sistema:
1.  **Consulta una Base de Conocimiento:** Solo responde basándose en los documentos (PDF, TXT, MD) que se proporcionen en la carpeta correspondiente.
2.  **Mantiene el Rigor Académico:** Cita las fuentes de donde extrae la información.
3.  **Entrega Respuestas Estructuradas:** Devuelve explicaciones y ejemplos prácticos en un formato claro.

## 🏗️ Arquitectura Técnica (RAG Local)
El flujo de información sigue estos pasos:
-   **Loader:** Procesa archivos locales en `knowledge_base/` usando `Unstructured` y `PyPDF`.
-   **Embeddings:** Genera representaciones vectoriales usando el modelo local `all-MiniLM-L6-v2`.
-   **Vector Store:** Almacena los vectores en **FAISS**, permitiendo búsquedas semánticas instantáneas.
-   **Retriever:** Encuentra los fragmentos más relevantes para cada pregunta.
-   **Ollama Agent:** Envía el contexto recuperado a un LLM local (ej. Mistral) para generar la respuesta final.

## 🧠 Diseño de Prompts

El sistema utiliza técnicas avanzadas de ingeniería de prompts:

### System Prompt Estructurado
Se define una personalidad de tutor académico con instrucciones estrictas:
-   Responder **únicamente** con el contexto proporcionado.
-   Si no hay información suficiente, admitirlo honestamente.
-   Mantener una salida en formato JSON para la integración con la interfaz.

### Few-Shot Prompting
Para asegurar que el modelo entienda el formato y el tono, el prompt incluye ejemplos:
> **Pregunta:** ¿Qué es overfitting?
> **Respuesta:** { "concepto": "Overfitting", "explicacion": "Es cuando un modelo memoriza los datos...", "ejemplo": "Un modelo que falla con datos nuevos", "fuente": "ml_basics.md" }

### Delimitadores de Contexto
Se utilizan triples comillas (`"""`) para separar claramente el conocimiento recuperado de la pregunta del usuario, evitando alucinaciones o inyecciones de prompt.

## 🛠️ Instalación y Uso

### 1. Requisitos
-   **Python 3.9+**
-   **Ollama** (ejecutando `mistral` o `llama3`): [ollama.com](https://ollama.com)

### 2. Configuración
```bash
# Instalar dependencias
pip install -r requirements.txt
```

### 3. Ejecución
1.  Asegurar que Ollama esté en ejecución.
2.  Ejecutar: `flask run` o `python app.py`.
3.  Acceder a `http://127.0.0.1:5000`.

## 📂 Estructura del Proyecto
-   `rag/`: Motor de ingesta y búsqueda.
-   `agent/`: Cerebro del asistente (conexión local).
-   `prompts/`: Definición de lógica y ejemplos.
-   `knowledge_base/`: Documentos de estudio.
-   `templates/`: Interfaz de usuario mejorada.

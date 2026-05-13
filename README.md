# 🤖 TUTOR\_IA — Sistema RAG con Búsqueda Semántica

Asistente conversacional basado en **RAG (Retrieval-Augmented Generation)** que responde consultas sobre cualquier colección de documentos usando búsqueda semántica vectorial y un LLM local. Diseñado para funcionar completamente **offline**, sin dependencias de APIs externas pagas.

---

## 📋 Tabla de Contenidos

1. [Arquitectura del sistema](#arquitectura-del-sistema)
2. [Proceso de ingesta de documentos](#proceso-de-ingesta-de-documentos)
3. [Vectorización y búsqueda semántica](#vectorización-y-búsqueda-semántica)
4. [Construcción del prompt aumentado](#construcción-del-prompt-aumentado)
5. [Informe de resultados](#informe-de-resultados)
6. [Instalación y configuración](#instalación-y-configuración)
7. [Estructura del proyecto](#estructura-del-proyecto)

---

## 🏗️ Arquitectura del Sistema

El sistema implementa el patrón RAG en dos fases bien diferenciadas:

```
╔══════════════════════════════════════════════════════════════════════╗
║              FASE 1 — INDEXACIÓN (se ejecuta una vez)               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  [Documentos PDF/TXT/MD]                                             ║
║          │                                                           ║
║          ▼                                                           ║
║  [Carga: PyPDFLoader / TextLoader]                                   ║
║          │                                                           ║
║          ▼                                                           ║
║  [Fragmentación: RecursiveCharacterTextSplitter]                     ║
║   chunk_size=500 palabras · chunk_overlap=50                         ║
║          │                                                           ║
║          ▼                                                           ║
║  [Embeddings: paraphrase-multilingual-MiniLM-L12-v2]                ║
║   Cada chunk → vector de 384 dimensiones                             ║
║          │                                                           ║
║          ▼                                                           ║
║  [Vector Store: ChromaDB (persistente en chroma_db/)]               ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║              FASE 2 — CONSULTA (por cada pregunta)                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  [Pregunta del usuario]                                              ║
║          │                                                           ║
║          ▼                                                           ║
║  [Embedding de la pregunta — mismo modelo]                           ║
║          │                                                           ║
║          ▼                                                           ║
║  [Búsqueda Semántica: similitud coseno en ChromaDB]                  ║
║   Recupera los k=3 fragmentos más similares                          ║
║          │                                                           ║
║          ▼                                                           ║
║  [Prompt Aumentado: sistema + contexto + pregunta]                   ║
║          │                                                           ║
║          ▼                                                           ║
║  [LLM: Ollama (phi3) — genera respuesta vía streaming]              ║
║          │                                                           ║
║          ▼                                                           ║
║  [Interfaz Streamlit — muestra respuesta + fuentes]                 ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Stack Tecnológico

| Capa | Componente | Detalle |
|------|-----------|---------|
| **Interfaz** | Streamlit | UI futurista con chat progresivo (streaming) |
| **LLM** | Ollama + phi3 | Inferencia 100% local, sin API externa |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` | Local, soporte español, 384 dimensiones |
| **Vector Store** | ChromaDB | Persistente en disco, por colección |
| **Retrieval** | LangChain Chroma Retriever | `search_type="similarity"` (coseno) |
| **Carga docs** | PyPDFLoader / TextLoader | PDF, TXT, MD |
| **Fragmentación** | RecursiveCharacterTextSplitter | 500 palabras, 50 de solapamiento |

---

## 📥 Proceso de Ingesta de Documentos

### 1. Carga

Los documentos se cargan desde `knowledge_base/{colección}/` usando LangChain:

```python
# PDF
loader = PyPDFLoader(filepath)      # Extrae texto página por página

# TXT / MD
loader = TextLoader(filepath, encoding="utf-8")

docs = loader.load()
```

Cada documento cargado conserva su `metadata` (nombre del archivo, página) para que pueda citarse como fuente en las respuestas.

### 2. Fragmentación (Chunking)

Los documentos completos se dividen en fragmentos manejables:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Máximo 500 palabras por fragmento
    chunk_overlap=50     # 50 palabras de solapamiento entre fragmentos
)
chunks = splitter.split_documents(docs)
```

**¿Por qué solapamiento?** Evita que información relevante quede cortada en el borde de dos fragmentos. El contexto compartido asegura coherencia semántica.

### 3. Almacenamiento

Cada colección tiene su propio directorio en ChromaDB:

```
chroma_db/
├── Machine_Learning/    ← vectores de la colección "Machine Learning"
├── Reglamento/          ← vectores de la colección "Reglamento"
└── ...
```

---

## 🔢 Vectorización y Búsqueda Semántica

### Modelo de Embeddings

**Modelo elegido:** `paraphrase-multilingual-MiniLM-L12-v2`

| Característica | Valor |
|----------------|-------|
| Dimensiones | 384 |
| Idiomas soportados | 50+ (incluyendo español) |
| Tamaño | ~117 MB |
| Ejecución | Local (sin API, sin costo) |
| Arquitectura | BERT multilingual destilado |

**¿Por qué este modelo?** A diferencia de `all-MiniLM-L6-v2` (solo inglés), el modelo multilingual comprende el español de forma nativa — esencial para documentos académicos y reglamentos en español.

### Proceso de Vectorización

```
"El aprendizaje supervisado usa ejemplos etiquetados"
                    │
                    ▼
    paraphrase-multilingual-MiniLM-L12-v2
                    │
                    ▼
    [0.23, -0.81, 0.44, 0.12, ..., 0.67]  ← vector 384 dims
```

Tanto los chunks del documento como la consulta del usuario pasan por el mismo modelo, garantizando que estén en el mismo espacio vectorial.

### Búsqueda Semántica (Similitud Coseno)

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",       # similitud coseno
    search_kwargs={"k": 3}          # recuperar 3 fragmentos más similares
)
```

**Ventaja clave sobre BM25/TF-IDF:**

| Criterio | BM25 (anterior) | Semántica (actual) |
|----------|----------------|-------------------|
| Busca por | Palabras exactas | Significado / concepto |
| "perder la materia" → "reprobación" | ❌ No encuentra | ✅ Encuentra |
| "aprender con ejemplos" → "supervisado" | ❌ No encuentra | ✅ Encuentra |
| Soporte sinónimos | ❌ | ✅ |
| Soporte lenguaje coloquial | ❌ | ✅ |

**Demostración de similitud coseno:**

```
Consulta:    "¿Cómo aprende una máquina de datos previos?"
                    │  (embedding)
                    ▼
         [0.12, -0.45, 0.78, ...]

Chunk A: "aprendizaje supervisado con datos etiquetados"
                    │  (embedding)
                    ▼                coseno = 0.89 ✅ TOP 1
         [0.11, -0.43, 0.80, ...]

Chunk B: "redes neuronales profundas"
                    │  (embedding)
                    ▼                coseno = 0.51
         [0.55, 0.12, -0.20, ...]

Chunk C: "historia del café en Colombia"
                    │  (embedding)
                    ▼                coseno = 0.08 ❌ Descartado
         [-0.30, 0.88, 0.01, ...]
```

---

## 🧠 Construcción del Prompt Aumentado

El prompt tiene tres secciones ensambladas dinámicamente en `tutor_agent.py`:

```
┌─────────────────────────────────────────────────────┐
│  SYSTEM PROMPT (instrucciones del asistente)        │
│  · Rol: experto en la colección "{collection}"      │
│  · Regla 1: solo usar el CONTEXTO provisto          │
│  · Regla 2: si no está → "No encuentro esa info"    │
│  · Regla 3: responder siempre en español            │
├─────────────────────────────────────────────────────┤
│  CONTEXTO RECUPERADO (top-k fragmentos)             │
│  [Fragmento 1 — Fuente: documento_a.pdf]            │
│  "...texto del chunk más relevante..."              │
│                                                     │
│  [Fragmento 2 — Fuente: documento_b.pdf]            │
│  "...segundo chunk más relevante..."                │
│                                                     │
│  [Fragmento 3 — Fuente: documento_a.pdf]            │
│  "...tercer chunk más relevante..."                 │
├─────────────────────────────────────────────────────┤
│  PREGUNTA                                           │
│  "{consulta del usuario}"                           │
├─────────────────────────────────────────────────────┤
│  → RESPUESTA: (genera Ollama)                       │
└─────────────────────────────────────────────────────┘
```

### Anti-alucinaciones

El system prompt instruye estrictamente:

```python
SYSTEM_PROMPT_TEMPLATE = """
Eres un asistente experto en la base de conocimiento: "{collection}".

REGLAS ESTRICTAS:
1. Responde EXCLUSIVAMENTE con la información del CONTEXTO.
2. Si no está en el contexto, responde EXACTAMENTE:
   "No encuentro esa información en '{collection}'."
3. No parafrasees ni alteres el contenido.
4. Responde siempre en español.
"""
```

---

## 📊 Informe de Resultados

### Configuración de Evaluación

| Parámetro | Valor |
|-----------|-------|
| Modelo de embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| chunk_size / overlap | 500 / 50 |
| k (chunks recuperados) | 3 |
| LLM generador | Ollama · phi3 |
| Modo de respuesta | Streaming (token a token) |
| Colección de prueba | Machine Learning |

### Casos de Prueba

| Tipo | Ejemplo | Resultado esperado |
|------|---------|-------------------|
| **Textual directo** | "¿Qué es el aprendizaje supervisado?" | Recupera definición exacta del doc |
| **Vocabulario diferente** | "¿Cómo aprende una IA de ejemplos clasificados?" | Encuentra chunks de supervisado |
| **Multi-chunk** | "Diferencias entre supervisado, no supervisado y refuerzo" | Combina fragmentos de varias partes |
| **Fuera de contexto** | "¿Quién ganó el Nobel de Física 2024?" | "No encuentro esa información..." |

### Comportamiento Observado

| Escenario | Comportamiento | Correcto |
|-----------|---------------|---------|
| Pregunta con vocabulario del documento | Recupera fragmento exacto | ✅ |
| Pregunta con sinónimos/coloquial | Búsqueda semántica encuentra conceptos | ✅ |
| Pregunta fuera de la base de conocimiento | Respuesta de fallback exacta | ✅ |
| Primera consulta (modelo frío) | Demora ~30-60s en cargar phi3 | ⚠️ Normal |
| Consultas siguientes | Respuesta en streaming ~5-15s | ✅ |

### Limitaciones Conocidas

- **Primera carga**: phi3 tarda en cargar a RAM (~30-60s en hardware modesto)
- **Idioma del modelo**: phi3 responde mejor en inglés; para español se recomienda `llama3.1:8b` o `mistral`
- **Calidad de PDFs**: PDFs escaneados (imágenes) no se indexan — solo PDFs con texto seleccionable

---

## ⚙️ Instalación y Configuración

### Requisitos Previos

- Python 3.10+
- [Ollama](https://ollama.com) instalado y corriendo

### Pasos

```powershell
# 1. Clonar
git clone https://github.com/DanTor07/TUTOR_IA.git
cd TUTOR_IA

# 2. Entorno virtual
python -m venv env
.\env\Scripts\Activate.ps1

# 3. Dependencias
pip install -r requirements.txt

# 4. Variables de entorno
cp .env.example .env
# Editar .env si es necesario

# 5. Descargar modelo LLM
ollama pull phi3

# 6. Iniciar Ollama (terminal separada)
ollama serve

# 7. Lanzar la aplicación
streamlit run streamlit_app.py
```

Abrir en el navegador: [http://localhost:8501](http://localhost:8501)

### Variables de Entorno (`.env`)

| Variable | Descripción | Default |
|----------|-------------|---------|
| `OLLAMA_URL` | Endpoint de Ollama | `http://localhost:11434/api/generate` |
| `OLLAMA_MODEL` | Modelo a usar | `phi3` |
| `EMBEDDING_MODEL` | Modelo de embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| `CHUNK_SIZE` | Palabras por fragmento | `500` |
| `CHUNK_OVERLAP` | Solapamiento entre fragmentos | `50` |
| `TOP_K` | Fragmentos recuperados por consulta | `3` |

---

## 📂 Estructura del Proyecto

```
TUTOR_IA/
├── streamlit_app.py          ← Interfaz de usuario (Streamlit)
├── agent/
│   └── tutor_agent.py        ← Pipeline RAG: retrieval + generación
├── rag/
│   ├── semantic_store.py     ← ChromaDB + embeddings (reemplaza BM25)
│   └── loader.py             ← Carga y chunking de documentos
├── prompts/
│   ├── system_prompt.py      ← Prompt anti-alucinaciones (dinámico por colección)
│   └── fewshot_examples.py   ← Ejemplos few-shot
├── knowledge_base/
│   └── {colección}/          ← Documentos fuente (PDF, TXT, MD)
├── chroma_db/                ← Vectores persistidos (generado automáticamente)
├── requirements.txt
├── .env.example              ← Plantilla de configuración
└── README.md
```

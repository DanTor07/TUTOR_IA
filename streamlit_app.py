"""
streamlit_app.py — Interfaz RAG futurista v2
Ejecutar: streamlit run streamlit_app.py
"""
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Assistant", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")
TOP_K        = os.getenv("TOP_K", "3")

# ── CSS completo ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset y base ── */
* { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #0b0b25 !important;
}

/* ── Luces ambientales de fondo ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background:
        radial-gradient(ellipse 60% 50% at 15% 90%, rgba(0,180,255,.10) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 85% 10%, rgba(168,85,247,.10) 0%, transparent 70%),
        radial-gradient(ellipse 40% 30% at 50% 50%, rgba(0,100,180,.04) 0%, transparent 70%);
}

/* ── Grid ── */
[data-testid="stMain"] {
    position: relative; z-index: 1;
    background:
        linear-gradient(rgba(0,180,255,.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,180,255,.03) 1px, transparent 1px);
    background-size: 44px 44px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(6,6,24,.98) !important;
    border-right: 1px solid rgba(0,180,255,.15) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span:not(.tech-tag),
[data-testid="stSidebar"] div:not(.tech-tag) { color: #a8c4f0 !important; }

/* ── Inputs sidebar ── */
[data-testid="stTextInput"] input {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid rgba(0,180,255,.22) !important;
    border-radius: 8px !important;
    color: #c0d8ff !important;
}
[data-testid="stTextInput"] input::placeholder { color: rgba(140,170,255,.4) !important; }

[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,.05) !important;
    border: 1px solid rgba(0,180,255,.22) !important;
    border-radius: 8px !important; color: #c0d8ff !important;
}

/* File uploader — quitar caja negra */
[data-testid="stFileUploader"] > div {
    background: rgba(0,180,255,.04) !important;
    border: 1px dashed rgba(0,180,255,.28) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] small { color: rgba(140,170,255,.6) !important; }
[data-testid="stFileUploader"] button {
    background: rgba(0,180,255,.1) !important;
    border: 1px solid rgba(0,180,255,.3) !important;
    color: #4dcfff !important; border-radius: 8px !important;
}

/* ── Contenedor principal centrado ── */
.main-wrap {
    max-width: 820px;
    margin: 0 auto;
    padding: 0 16px;
}

/* ── Chat input — sin caja oscura flotante ── */
[data-testid="stBottom"],
[data-testid="stBottom"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] {
    background: transparent !important;
    border: none !important;
    padding: 6px 20px 18px !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(255,255,255,.06) !important;
    border: 1px solid rgba(0,180,255,.30) !important;
    border-radius: 14px !important;
    color: #e0eaff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .95rem !important;
    box-shadow: 0 0 24px rgba(0,180,255,.06) inset !important;
    padding: 14px 18px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: rgba(0,180,255,.6) !important;
    box-shadow: 0 0 0 2px rgba(0,180,255,.12), 0 0 24px rgba(0,180,255,.06) inset !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: rgba(140,170,255,.45) !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #00b4ff, #7b2ff7) !important;
    border: none !important; border-radius: 10px !important;
    box-shadow: 0 0 14px rgba(0,180,255,.25) !important;
}

/* ── Chat messages — mas claros ── */
[data-testid="stChatMessage"] {
    background: rgba(20,20,60,.65) !important;
    border: 1px solid rgba(0,180,255,.13) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(16px) !important;
    margin: 8px 0 !important;
    color: #dde8ff !important;
}
[data-testid="stChatMessage"] p { color: #dde8ff !important; }

/* ── Botones ── */
.stButton > button {
    background: rgba(0,180,255,.07) !important;
    border: 1px solid rgba(0,180,255,.3) !important;
    color: #4dcfff !important;
    border-radius: 9px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .82rem !important; letter-spacing: .04em !important;
    transition: all .22s ease !important;
}
.stButton > button:hover {
    background: rgba(0,180,255,.16) !important;
    box-shadow: 0 0 18px rgba(0,180,255,.25) !important;
    transform: translateY(-1px) !important;
}

/* ── Select / inputs ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,.05) !important;
    border-color: rgba(0,180,255,.25) !important;
    border-radius: 8px !important;
    color: #c0d8ff !important;
}

/* ── Expander / Alertas ── */
details {
    background: rgba(0,180,255,.04) !important;
    border: 1px solid rgba(0,180,255,.14) !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"] {
    background: rgba(0,100,200,.12) !important;
    border: 1px solid rgba(0,180,255,.22) !important;
    border-radius: 12px !important; color: #c0d8ff !important;
}

/* ── Divider / Scrollbar ── */
hr { border-color: rgba(0,180,255,.1) !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(0,180,255,.22); border-radius: 10px; }

/* ════════════════════════════════════
   CLASES PERSONALIZADAS
   ════════════════════════════════════ */

/* Título principal */
.rag-title {
    font-family: 'Orbitron', monospace;
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(90deg, #00b4ff 0%, #a855f7 50%, #00b4ff 100%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    margin: 0; line-height: 1.2;
}
.rag-sub {
    color: rgba(148,172,255,.62);
    font-family: 'Inter', sans-serif;
    font-size: .83rem; letter-spacing: .07em; margin-top: 6px;
}

/* Tech tags sidebar */
.tech-tag {
    display: inline-block;
    background: rgba(0,180,255,.08);
    border: 1px solid rgba(0,180,255,.25);
    border-radius: 20px; padding: 3px 11px;
    font-size: .74rem; color: #4dcfff !important;
    margin: 2px; font-family: 'Inter', sans-serif;
    letter-spacing: .04em;
}

/* Hero (estado vacío) */
.hero {
    text-align: center;
    padding: 60px 20px 40px;
}
.hero-icon {
    font-size: 3.5rem; margin-bottom: 20px;
    filter: drop-shadow(0 0 24px rgba(0,180,255,.5));
    animation: float 3s ease-in-out infinite;
}
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem; font-weight: 700; color: #e0eaff;
    letter-spacing: .08em; margin-bottom: 10px;
}
.hero-desc {
    color: rgba(160,185,255,.65);
    font-family: 'Inter', sans-serif;
    font-size: .92rem; line-height: 1.7;
    max-width: 500px; margin: 0 auto 30px;
}
.chips-row {
    display: flex; flex-wrap: wrap;
    gap: 10px; justify-content: center;
    max-width: 600px; margin: 0 auto;
}
.chip {
    background: rgba(0,180,255,.08);
    border: 1px solid rgba(0,180,255,.28);
    border-radius: 24px; padding: 7px 16px;
    font-size: .83rem; color: #7dd4ff;
    font-family: 'Inter', sans-serif;
    cursor: default; transition: all .2s;
}
.chip:hover {
    background: rgba(0,180,255,.15);
    box-shadow: 0 0 12px rgba(0,180,255,.2);
}

/* Decoración ambiental */
.ambient {
    position: fixed; border-radius: 50%;
    pointer-events: none; z-index: 0;
    filter: blur(80px);
}
.amb1 {
    width: 320px; height: 320px;
    background: rgba(0,140,255,.06);
    bottom: -60px; left: 30%;
    animation: drift 8s ease-in-out infinite;
}
.amb2 {
    width: 220px; height: 220px;
    background: rgba(130,60,220,.06);
    top: 10%; right: 5%;
    animation: drift 11s ease-in-out infinite reverse;
}

/* Source card */
.source-card {
    background: rgba(0,180,255,.05);
    border-left: 3px solid rgba(0,180,255,.45);
    border-radius: 0 10px 10px 0;
    padding: 10px 15px; margin: 8px 0;
    font-size: .83rem; color: #b8d0ff;
    font-family: 'Inter', sans-serif; line-height: 1.6;
}
.source-label {
    font-size: .70rem; text-transform: uppercase;
    letter-spacing: .09em; color: #4dcfff;
    font-weight: 600; margin-bottom: 5px;
}

/* Loading */
.loading-wrap {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 65vh; gap: 22px;
}
.loading-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.2rem; color: #00b4ff;
    letter-spacing: .12em; text-align: center;
}
.loading-sub {
    color: rgba(150,175,255,.6);
    font-family: 'Inter', sans-serif;
    font-size: .87rem; text-align: center;
}
.dot {
    width: 11px; height: 11px; border-radius: 50%;
    display: inline-block; margin: 0 4px;
    animation: pulse 1.3s ease-in-out infinite;
    box-shadow: 0 0 10px currentColor;
}
.dot1 { background:#00b4ff; color:#00b4ff; }
.dot2 { background:#a855f7; color:#a855f7; animation-delay:.22s; }
.dot3 { background:#00b4ff; color:#00b4ff; animation-delay:.44s; }

@keyframes shimmer { 0%{background-position:0% center} 100%{background-position:200% center} }
@keyframes pulse { 0%,100%{opacity:.25;transform:scale(.75)} 50%{opacity:1;transform:scale(1.25)} }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
@keyframes drift { 0%,100%{transform:translate(0,0)} 50%{transform:translate(20px,15px)} }
</style>

<!-- Decoración ambiental -->
<div class='ambient amb1'></div>
<div class='ambient amb2'></div>
""", unsafe_allow_html=True)


LOADING_HTML = lambda col: f"""
<div class='loading-wrap'>
    <div class='rag-title' style='font-size:1.6rem;letter-spacing:.14em;'>🤖 SISTEMA RAG</div>
    <div><span class='dot dot1'></span><span class='dot dot2'></span><span class='dot dot3'></span></div>
    <div class='loading-title'>INICIALIZANDO · {col.upper()}</div>
    <div class='loading-sub'>Cargando modelo de embeddings y base vectorial…<br>
    La primera vez puede tomar unos segundos.</div>
</div>"""


def hero_html(collection: str, examples: list) -> str:
    chips = "".join(f"<span class='chip'>💬 {e}</span>" for e in examples)
    return f"""
<div class='hero'>
    <div class='hero-icon'>🤖</div>
    <div class='hero-title'>ASISTENTE RAG · {collection.upper()}</div>
    <div class='hero-desc'>
        Consulta la base de conocimiento usando lenguaje natural.<br>
        La búsqueda semántica encuentra respuestas incluso si usas
        sinónimos o vocabulario diferente al del documento.
    </div>
    <div style='color:rgba(120,160,255,.5);font-size:.78rem;
                font-family:Inter,sans-serif;letter-spacing:.06em;
                margin-bottom:14px;'>PREGUNTAS DE EJEMPLO</div>
    <div class='chips-row'>{chips}</div>
</div>"""


EXAMPLE_QUESTIONS = {
    "default": [
        "¿Cuáles son los principales conceptos?",
        "Explícame con un ejemplo práctico",
        "¿Qué diferencias hay entre...?",
        "Resume los puntos más importantes",
    ]
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_collections():
    kb = "knowledge_base"
    os.makedirs(kb, exist_ok=True)
    cols = [d for d in os.listdir(kb) if os.path.isdir(os.path.join(kb, d))]
    if not cols:
        os.makedirs(os.path.join(kb, "Reglamento"), exist_ok=True)
        cols = ["Reglamento"]
    return sorted(cols)


@st.cache_resource(show_spinner=False)
def load_tutor(col):
    from agent.tutor_agent import TutorAgent
    t = TutorAgent()
    t.load_collection(col)
    return t


def render_sources(sources):
    with st.expander(f"📑 Fragmentos consultados ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"""
            <div class='source-card'>
                <div class='source-label'>Fragmento {i} · {s['source']}</div>
                {s['text'][:440]}{'…' if len(s['text'])>440 else ''}
            </div>""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [("messages", []), ("active_col", None)]:
    if key not in st.session_state:
        st.session_state[key] = val


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ CONTROL")
    st.divider()

    collections = get_collections()
    selected = st.selectbox("📁 Colección activa", collections)

    if selected != st.session_state.active_col:
        st.session_state.active_col = selected
        st.session_state.messages = []
        load_tutor.clear()

    st.divider()
    st.markdown("**📄 Subir documentos**")
    uploads = st.file_uploader("PDF / TXT / MD", type=["pdf","txt","md"],
                               accept_multiple_files=True)
    if uploads:
        col_dir = os.path.join("knowledge_base", selected)
        for f in uploads:
            with open(os.path.join(col_dir, f.name), "wb") as out:
                out.write(f.getbuffer())
        if st.button("🔄 Reconstruir índice", use_container_width=True):
            load_tutor.clear()
            ph = st.empty()
            ph.markdown(LOADING_HTML(selected), unsafe_allow_html=True)
            load_tutor(selected)
            ph.empty()
            st.success("✅ Índice actualizado")

    st.divider()
    st.markdown("**➕ Nueva colección**")
    new_col = st.text_input("Nombre de la colección", placeholder="Ej: Reglamento 2024",
                            label_visibility="collapsed")
    if st.button("Crear", use_container_width=True) and new_col.strip():
        os.makedirs(os.path.join("knowledge_base", new_col.strip()), exist_ok=True)
        st.rerun()

    st.divider()
    st.markdown("**🔬 Stack Técnico**")
    st.markdown(f"""
    <span class='tech-tag'>🧠 {OLLAMA_MODEL} · Ollama</span>
    <span class='tech-tag'>🔢 multilingual-MiniLM-L12</span>
    <span class='tech-tag'>🗄️ ChromaDB</span>
    <span class='tech-tag'>🔍 coseno · k={TOP_K}</span>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️ Limpiar historial", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Loading inicial ───────────────────────────────────────────────────────────
main_slot = st.empty()

if f"ready_{selected}" not in st.session_state:
    main_slot.markdown(LOADING_HTML(selected), unsafe_allow_html=True)
    load_tutor(selected)
    st.session_state[f"ready_{selected}"] = True
    main_slot.empty()
    st.rerun()

tutor = load_tutor(selected)
main_slot.empty()


# ── Layout principal ──────────────────────────────────────────────────────────
# Header dinámico
col_l, col_r = st.columns([6, 1])
with col_l:
    st.markdown(f"""
    <div class='rag-title'>⬡ {selected.upper()}</div>
    <div class='rag-sub'>▸ ASISTENTE RAG &nbsp;·&nbsp; búsqueda semántica
    &nbsp;·&nbsp; {OLLAMA_MODEL} local &nbsp;·&nbsp; sin alucinaciones</div>
    """, unsafe_allow_html=True)

st.divider()

# ── Hero (estado vacío) ───────────────────────────────────────────────────────
if not st.session_state.messages:
    examples = EXAMPLE_QUESTIONS.get(selected, EXAMPLE_QUESTIONS["default"])
    st.markdown(f"<div class='main-wrap'>{hero_html(selected, examples)}</div>",
                unsafe_allow_html=True)

# ── Historial centrado ────────────────────────────────────────────────────────
else:
    _, chat_col, _ = st.columns([1, 10, 1])
    with chat_col:
        for msg in st.session_state.messages:
            av = "🧑‍💻" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=av):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    render_sources(msg["sources"])

# ── Input ─────────────────────────────────────────────────────────────────────
THINKING_HTML = """
<div style='display:flex;align-items:center;gap:10px;padding:6px 2px;'>
    <span style='color:rgba(150,175,255,.55);font-family:Inter,sans-serif;
                 font-size:.84rem;letter-spacing:.06em;'>Procesando</span>
    <span class='dot dot1'></span>
    <span class='dot dot2'></span>
    <span class='dot dot3'></span>
</div>"""

if prompt := st.chat_input(f"Consulta sobre {selected}…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    _, chat_col2, _ = st.columns([1, 10, 1])
    with chat_col2:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            # 1. Dots animados mientras llega el primer token
            thinking_slot = st.empty()
            thinking_slot.markdown(THINKING_HTML, unsafe_allow_html=True)

            # 2. Streaming manual: primer token → quitar dots → texto progresivo
            text_slot   = st.empty()
            chunks      = []
            got_first   = False

            for token in tutor.stream_response(prompt):
                if not got_first:
                    thinking_slot.empty()   # quitar "Procesando..."
                    got_first = True
                chunks.append(token)
                # Cursor parpadeante al final mientras llegan tokens
                text_slot.markdown("".join(chunks) + " ▋")

            # 3. Render final limpio sin cursor
            answer = "".join(chunks)
            if not got_first:
                thinking_slot.empty()       # si no llegó ningún token
            text_slot.markdown(answer)

            sources = tutor._last_sources
            if sources:
                render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources
    })
    st.rerun()

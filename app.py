from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import os
import shutil
import secrets
from agent.tutor_agent import TutorAgent
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'knowledge_base'

# Configuración inicial y migración de colecciones
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

default_coll_path = os.path.join(app.config['UPLOAD_FOLDER'], "Machine Learning")
if not os.path.exists(default_coll_path):
    os.makedirs(default_coll_path)

for item in os.listdir(app.config['UPLOAD_FOLDER']):
    item_path = os.path.join(app.config['UPLOAD_FOLDER'], item)
    if os.path.isfile(item_path):
        shutil.move(item_path, os.path.join(default_coll_path, item))

# Inicializar Agente Tutor
tutor = TutorAgent(base_kb_dir=app.config['UPLOAD_FOLDER'])

def get_collections():
    collections = []
    for item in os.listdir(app.config['UPLOAD_FOLDER']):
        if os.path.isdir(os.path.join(app.config['UPLOAD_FOLDER'], item)):
            collections.append(item)
    return collections

@app.route("/", methods=["GET", "POST"])
def index():
    if "history" not in session:
        session["history"] = []

    collections = get_collections()
    if not collections:
        collections = ["Machine Learning"]
        
    if "current_collection" not in session or session["current_collection"] not in collections:
        session["current_collection"] = collections[0]

    # Cargar la colección actual en el agente
    tutor.load_collection(session["current_collection"])

    if request.method == "POST":
        user_input = request.form.get("message", "").strip()
        
        if user_input:
            # Obtener respuesta del tutor RAG
            response_json, retrieved_context = tutor.ask(user_input)
            
            # Guardar en historial
            session["history"].append({
                "role": "Estudiante",
                "message": user_input
            })
            session["history"].append({
                "role": "Asistente",
                "structured_response": response_json,
                "context": retrieved_context
            })
            session.modified = True

    return render_template("index.html", 
                           history=session["history"], 
                           collections=collections, 
                           current_collection=session["current_collection"])

@app.route("/change_collection", methods=["POST"])
def change_collection():
    new_col = request.form.get("collection")
    if new_col and new_col in get_collections():
        session["current_collection"] = new_col
        session["history"] = []
    return redirect(url_for("index"))

@app.route("/create_collection", methods=["POST"])
def create_collection():
    new_col = request.form.get("new_collection", "").strip()
    # Sanitizar nombre
    new_col = secure_filename(new_col)
    if new_col:
        new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_col)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        session["current_collection"] = new_col
        session["history"] = []
    return redirect(url_for("index"))

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and "current_collection" in session:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], session["current_collection"], filename)
        file.save(save_path)
        # Refrescar la base de conocimientos
        tutor.refresh_knowledge_base()
        return redirect(url_for('index'))

@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)

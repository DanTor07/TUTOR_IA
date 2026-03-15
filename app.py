from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import os
import secrets
from agent.tutor_agent import TutorAgent
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['UPLOAD_FOLDER'] = 'knowledge_base'

# Inicializar Agente Tutor
# Nota: Se necesita tener Ollama corriendo localmente.
tutor = TutorAgent()

@app.route("/", methods=["GET", "POST"])
def index():
    if "history" not in session:
        session["history"] = []

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

    return render_template("index.html", history=session["history"])

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Refrescar la base de conocimientos
        tutor.refresh_knowledge_base()
        return redirect(url_for('index'))

@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)

from flask import Flask, render_template, jsonify, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
import general as gen
# import audio_to_text as audio_convtr
import model as bert
import subprocess
import sys
import os
import re

app = Flask(__name__)
CORS(app)

# Static Audio Folder
AUDIO_FOLDER = os.path.join("static", "Audio")
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Allowed Audio Extensions
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "ogg", "flac", "mpeg"}

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/internal")
def internal():
    return render_template("index.html")

@app.route('/media')
def media():
    return render_template('media.html')

@app.route('/general')
def general():
    return render_template('general.html')

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    print(data)
    messages = data.get("messages")

    res, table_data = bert.get_bert_response(messages)

    return jsonify({
        "reply": res,
        "table": table_data
    })

@app.route("/general-chat", methods=["POST"])
def general_chat():
    data = request.get_json()
    print(data)
    messages = data.get("messages")

    res = gen.general_questions(messages)
    print(f'[General Message] : {str(res)}')

    return jsonify({
        "reply": res,
    })

@app.route("/media_chat", methods=["POST"])
def media_chat():
    data = request.get_json()
    print(data)

    messages = data.get("messages")
    context = data.get("context")

    try:
        script_path = os.path.join(os.path.dirname(__file__), "audio_rag.py")

        process = subprocess.Popen(
            [
                sys.executable,
                script_path,
                "--input", str(context),
                "--question", str(messages)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            stdout, stderr = process.communicate(timeout=1000)  # optional timeout
            print("Output:", stdout)

        except subprocess.TimeoutExpired:
            print("Process timed out. Killing it...")
            process.kill()
            stdout, stderr = process.communicate()

        finally:
            if process.poll() is None:
                process.kill()   # ensure cleanup

        full_output = stdout
        print(f"[Full Script Output]:\n{full_output}")

        match = re.search(r"Answer:\s*(.*)", full_output)

        if match:
            clean_answer = match.group(1).strip()
        else:
            clean_answer = "No answer found."

        print(f"[Extracted Answer]: {clean_answer}")

        return jsonify({
            "reply": clean_answer
        })

    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return jsonify({
            "error": e.stderr
        }), 500
    
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(AUDIO_FOLDER, filename)

        file.save(save_path)

        # Load Whisper only when audio conversion is actually requested
        import audio_to_text as audio_convtr

        cvt_text = audio_convtr.audio_text_cvtr(
            f"static/Audio/{filename}"
        )
    else:
        return jsonify({"error": "Invalid audio format"}), 400
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

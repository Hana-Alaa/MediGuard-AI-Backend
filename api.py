import os
import json
import base64
import tempfile
import numpy as np
from gtts import gTTS
from openai import OpenAI
from flask import Flask, request, jsonify
from integrated_system import IntegratedMediGuardSystem
from chatbot_module import process_message, transcribe_audio_bytes, speech_to_text, model

app = Flask(__name__)
system = IntegratedMediGuardSystem(language="en")

PATIENTS_DIR = os.getenv("PATIENTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "patients")
os.makedirs(PATIENTS_DIR, exist_ok=True)

UPLOADS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# ---------------- Utility Functions ----------------
def load_patient(patient_id):
    file_path = os.path.join(PATIENTS_DIR, f"patient_{patient_id}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None

def save_patient(patient_id, data):
    file_path = os.path.join(PATIENTS_DIR, f"patient_{patient_id}.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def convert(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

# ---------------- Audio Handling ----------------
def save_audio_file(audio_bytes, patient_id, ext="wav"):
    file_path = os.path.join(UPLOADS_DIR, f"{patient_id}_voice.{ext}")
    
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
    return file_path

# ---------------- API Endpoints ----------------
@app.route("/integrated-analysis", methods=["POST"])
def integrated_analysis():
    try:
        data = request.json
        patient_id = data.get("patient_id", "unknown")

        ecg_signal = np.array(data.get("ecg_signal", []))
        vitals = data.get("vitals", {})

        flattened_vitals = {}
        for k, v in vitals.items():
            if isinstance(v, dict) and "value" in v:
                flattened_vitals[k] = v["value"]
            elif isinstance(v, dict) and "systolic" in v and "diastolic" in v:
                flattened_vitals["systolic_bp"] = v["systolic"]
                flattened_vitals["diastolic_bp"] = v["diastolic"]
            else:
                flattened_vitals[k] = v

        patient_data = {
            "patient_id": patient_id,
            "ecg_signal": ecg_signal,
            "vital_signs": flattened_vitals
        }
        results = system.comprehensive_patient_analysis(patient_data)
        results_clean = json.loads(json.dumps(results, default=convert))

        patient = load_patient(patient_id) or {
            "patient_id": patient_id,
            "name": data.get("name"),
            "age": data.get("age"),
            "gender": data.get("gender") if data.get("gender") else "unknown",
            "chronic_conditions": data.get("chronic_conditions", []),
            "notes": data.get("notes", ""),
            "analysis_history": []
        }

        existing = load_patient(patient_id)
        if existing and existing.get("gender"):
            patient["gender"] = existing.get("gender")

        patient["ecg_signal"] = ecg_signal.tolist()
        patient["vitals"] = vitals
        patient["last_analysis"] = results_clean
        patient["analysis_history"].append({
            "timestamp": results_clean.get("analysis_timestamp"),
            "risk_level": results_clean.get("combined_assessment", {}).get("combined_risk_level"),
            "alert_color": results_clean.get("combined_assessment", {}).get("alert_color")
        })

        if not patient.get("gender"):
            patient["gender"] = "unknown"

        save_patient(patient_id, patient)

        return jsonify({patient_id: patient})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/patient/<patient_id>", methods=["GET"])
def get_patient(patient_id):
    patient = load_patient(patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    return jsonify({patient_id: patient})

@app.route("/patient/<patient_id>/history", methods=["GET"])
def get_history(patient_id):
    patient = load_patient(patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    return jsonify(patient.get("analysis_history", []))

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    patient_id = data.get("patient_id", "unknown")

    patient = load_patient(patient_id)
    if not patient:
        return jsonify({"reply": f"🚫 Patient '{patient_id}' not found. Please run analysis first."}), 404

    try:
        bot_reply = process_message(user_message, patient_id=patient_id)
    except Exception as e:
        bot_reply = f"⚠️ حصل خطأ: {str(e)}"

    return jsonify({"reply": bot_reply})

@app.route("/chat-audio", methods=["POST"])
def chat_audio():
    try:
        data = request.get_json()
        audio_base64 = data.get("audio_base64")
        patient_id = data.get("patient_id", "unknown")
        file_ext = data.get("file_ext", "wav")

        if not audio_base64:
            return jsonify({"error": "No audio data provided"}), 400

        # Decode & transcribe
        audio_bytes = base64.b64decode(audio_base64)
        text, detected_lang = transcribe_audio_bytes(audio_bytes, ext=file_ext)
        print(f"🎤 Transcribed ({detected_lang}): {text}")

        # Process chatbot reply
        reply = process_message(text, patient_id=patient_id)
        print(f"🤖 Reply: {reply}")

        # ---------- TTS ----------
        try:
            tts_lang = "ar" if detected_lang == "ar" else "en"
            tts = gTTS(text=reply, lang=tts_lang)

            # safe temp file for Windows
            fd, temp_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)

            tts.save(temp_path)

            with open(temp_path, "rb") as f:
                reply_audio_base64 = base64.b64encode(f.read()).decode("utf-8")

            os.remove(temp_path)  
        except Exception as tts_err:
            print(f"❌ TTS error: {tts_err}")
            reply_audio_base64 = None

        return jsonify({
            "transcript": text,
            "lang": detected_lang,
            "reply": reply,
            "reply_audio": reply_audio_base64
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500
    
# ---------------- Run Flask ----------------
if __name__ == "__main__":
    print(app.url_map)
    app.run(host="0.0.0.0", port=5000, debug=True)

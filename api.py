import numpy as np, json, os
from flask import Flask, request, jsonify
from chatbot_module import process_message
from integrated_system import IntegratedMediGuardSystem

app = Flask(__name__)
system = IntegratedMediGuardSystem(language="en")

PATIENTS_DIR = os.getenv("PATIENTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "patients")
os.makedirs(PATIENTS_DIR, exist_ok=True)

# Load patient data
def load_patient(patient_id):
    file_path = os.path.join(PATIENTS_DIR, f"patient_{patient_id}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None

# Save patient data
def save_patient(patient_id, data):
    file_path = os.path.join(PATIENTS_DIR, f"patient_{patient_id}.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def convert(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

# API for integrated patient analysis
@app.route("/integrated-analysis", methods=["POST"])
def integrated_analysis():
    try:
        data = request.json
        patient_id = data.get("patient_id", "unknown")

        ecg_signal = np.array(data.get("ecg_signal", []))
        vitals = data.get("vitals", {})

        # Flatten vitals to extract only the values
        flattened_vitals = {}
        for k, v in vitals.items():
            if isinstance(v, dict) and "value" in v:
                flattened_vitals[k] = v["value"]
            elif isinstance(v, dict) and "systolic" in v and "diastolic" in v:
                flattened_vitals["systolic_bp"] = v["systolic"]
                flattened_vitals["diastolic_bp"] = v["diastolic"]
            else:
                flattened_vitals[k] = v

        # Run analysis using the integrated system
        patient_data = {
            "patient_id": patient_id,
            "ecg_signal": ecg_signal,
            "vital_signs": flattened_vitals
        }
        results = system.comprehensive_patient_analysis(patient_data)
        results_clean = json.loads(json.dumps(results, default=convert))

        # Load existing patient data if available
        patient = load_patient(patient_id) or {
            "patient_id": patient_id,
            "name": data.get("name"),
            "age": data.get("age"),
            # Ensure gender falls back to existing or a safe default
            "gender": data.get("gender") if data.get("gender") not in (None, "") else "unknown",
            "chronic_conditions": data.get("chronic_conditions", []),
            "notes": data.get("notes", ""),
            "analysis_history": []
        }

        # If an existing patient file has a gender, prefer it when incoming data omits it
        existing = load_patient(patient_id)
        if existing:
            existing_gender = existing.get("gender")
            if existing_gender not in (None, ""):
                patient["gender"] = existing_gender

        # Update patient data
        patient["ecg_signal"] = ecg_signal.tolist()
        patient["vitals"] = vitals
        patient["last_analysis"] = results_clean
        patient["analysis_history"].append({
            "timestamp": results_clean.get("analysis_timestamp"),
            "risk_level": results_clean.get("combined_assessment", {}).get("combined_risk_level"),
            "alert_color": results_clean.get("combined_assessment", {}).get("alert_color")
        })

        # Make sure gender is always set to a non-empty value before saving
        if not patient.get("gender"):
            patient["gender"] = "unknown"

        # Save updated patient file
        save_patient(patient_id, patient)

        return jsonify({patient_id: patient})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API to fetch patient data
@app.route("/patient/<patient_id>", methods=["GET"])
def get_patient(patient_id):
    patient = load_patient(patient_id)
    if not patient:
        return jsonify({"error": "Patient not found"}), 404
    return jsonify({patient_id: patient})

# API to fetch patient analysis history
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

    # Check if the patient exists
    patient = load_patient(patient_id)
    if not patient:
        return jsonify({"reply": f"🚫 Patient '{patient_id}' not found. Please run analysis first."}), 404

    try:
        # Pass patient_id to chatbot processing
        bot_reply = process_message(user_message, patient_id=patient_id)
    except Exception as e:
        bot_reply = f"⚠️ حصل خطأ: {str(e)}"

    return jsonify({"reply": bot_reply})

if __name__ == "__main__":
    print(app.url_map)
    app.run(host="0.0.0.0", port=5000, debug=True)

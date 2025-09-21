# MediGuard Backend

**Project Description:**  
MediGuard Backend is part of the MediGuard AI system, designed to automatically assess and monitor patients' health using artificial intelligence. The system analyzes vital signs such as blood pressure, oxygen saturation, heart rate, and temperature, and provides intelligent medical recommendations.

**Key Features:**  
- Patient vital sign analysis.  
- General risk assessment based on NEWS (National Early Warning Score).  
- Smart recommendations based on clinical rules and AI.  
- Integration-ready for frontend or full system.

**Core Libraries:**  
- Python 3.11+  
- Flask  
- Pandas, NumPy  
- TensorFlow, scikit-learn  
- SpeechRecognition  
- OpenAI API

---

## Setup Instructions

### 1. Create a virtual environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
$env:OPENROUTER_API_KEY="your_api_key_here"
python app.py





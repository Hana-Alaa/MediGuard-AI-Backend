import os
import re
import io
import json
import wave
import tempfile
import playsound
import numpy as np
from gtts import gTTS
from openai import OpenAI
from langdetect import detect
import speech_recognition as sr
from faster_whisper import WhisperModel
from vitals_analyzer import generate_report, get_vitals, generate_summary_from_report
# =====================
# Setup OpenRouter 
# =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ No OpenRouter API Key found. Please set OPENROUTER_API_KEY environment variable.")

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
LLM_MODEL = "openai/gpt-4o-mini"

PATIENTS_DIR = os.getenv("PATIENTS_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "patients")
os.makedirs(PATIENTS_DIR, exist_ok=True)

# Function to call LLM
def call_llm(prompt, system="You are a helpful AI medical assistant.", role="user"):
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": role, "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error calling LLM: {str(e)}")
        return ""

# Detect language
def detect_language(text):
    if not text or not text.strip():
        return "en"
    # Any Arabic character -> set as Arabic
    if any("\u0600" <= c <= "\u06FF" for c in text):
        return "ar"
    try:
        lang = detect(text)
        return "ar" if lang == "ar" else "en"
    except:
        return "en"

# Speech recognition
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak your question...")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source, timeout=5, phrase_time_limit=10)

    # Try Arabic then English
    for language in ["ar-EG", "en-US"]:
        try:
            text = r.recognize_google(audio, language=language)
            if text.strip():
                return text
        except:
            continue
    return None

# Get patient context
def get_patient_context(patient_id="unknown"):
    try:
        file_path = os.path.join(PATIENTS_DIR, f"patient_{patient_id}.json")
        if not os.path.exists(file_path):
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "name": data.get("name", "الـمريض"),
            "age": data.get("age", "?"),
            "gender": data.get("gender", "?"),
            "chronic": data.get("chronic_conditions", []),
            "notes": data.get("notes", ""),
            "vitals": data.get("vitals", {}),
            "ecg": data.get("last_analysis", {}).get("ecg_analysis", {}),
            "risk": data.get("last_analysis", {}).get("combined_assessment", {}).get("combined_risk_level", "N/A")
        }
    except Exception as e:
        print(f"Error in patient data: {e}")
        return None

# Question classification using LLM
def classify_question(user_input, lang="ar"):
    # Retain vital keywords check as is
    vital_keywords = ["blood pressure", "bp", "heart rate", "hr", "temperature", "temp", "spo2", "oxygen", "respiratory", "ecg", "ekg", "electrocardiogram","pulse rate"
                    "ضغط الدم", "النبض", "معدل ضربات القلب", "درجة الحرارة", "الأكسجين", "معدل التنفس", "رسم القلب", "إيكو", "إي كيه جي", "مخطط كهربية القلب", "مخطط القلب", "حرارة"]
    if any(k.lower() in user_input.lower() for k in vital_keywords):
        return "medical_question"

    # Use LLM for all other classifications
    if lang == "ar":
        prompt = f"""
صنّف النص التالي إلى واحد فقط من التصنيفات:
- medical_question
- status_report
- lifestyle_question
- non_medical
- greeting
- farewell

النص: "{user_input}"
جاوب بكلمة واحدة فقط (بالعربية أو بالإنجليزية حسب النص).
"""
    else:
        prompt = f"""
Classify the following text into one of: 
- medical_question
- status_report
- lifestyle_question
- non_medical
- greeting
- farewell

Text: "{user_input}"
Answer with ONLY one category.
"""
    result = call_llm(prompt, system="You are an assistant that classifies user questions.")
    return result.lower() if result else "non_medical"

# Main logic
def process_message(user_message, patient_id="unknown"):
    context = get_patient_context(patient_id)
    if not user_message or not user_message.strip():
        return "من فضلك اسأل سؤال طبي محدد."
    lang = detect_language(user_message)

    # Classify the question (now includes greeting/farewell)
    category = classify_question(user_message, lang)

    # -------- Handle greetings --------
    if category == "greeting":
        return "مرحباً! كيف يمكنني مساعدتك اليوم؟" if lang=="ar" else "Hello! How can I help you today?"

    # -------- Handle farewells --------
    if category == "farewell":
        return "وداعاً! نتمنى لك صحة جيدة." if lang=="ar" else "Goodbye! Wishing you good health."

    # -------- If report requested --------
    if category == "status_report":
        if not context:
            return "❌ لم يتم العثور على بيانات المريض"
        if any(word in user_message.lower() for word in ["باختصار", "ملخص", "summary", "brief", "summarize", "overall"]):
            return generate_summary_from_report(context, lang=lang)  
        else:
            return generate_report(context, lang=lang)               

    # -------- If medical or lifestyle question --------
    if category in ["medical_question", "lifestyle_question"]:
        context = get_patient_context(patient_id)

        if lang == "ar":
            base_prompt = """أنت مساعد صحي ذكي وشخصي متخصص. ردّ بالعربية فقط.أنت مساعد صحي ذكي متخصص.
مهتك إعطاء إجابات طبية أو أسلوب حياة بصيغة الغائب فقط عن حالة المريض (المريض/هو
راعِ بيانات المريض الحالية في كل إجابة. قدّم نصيحة مناسبة لحالته فقط."""
        else:
            base_prompt = """You are a smart medical assistant, personalized health assistant.
            All responses must be in THIRD PERSON about the patient.
            Never use second-person phrases like "you" or "your". Always say: "The patient..." / "His condition..." Reply in English ONLY. Always consider the patient's current condition before giving advice."""

        # Build patient context string
        patient_info = ""
        if context:
            if lang == "ar":
                patient_info = f"""
بيانات المريض:
- الاسم: {context['name']}، العمر: {context['age']}، الجنس: {context['gender']}
- الأمراض المزمنة: {', '.join(context['chronic']) if context['chronic'] else 'لا يوجد'}
- الملاحظات: {context['notes']}
- النبض: {context['vitals']['hr']} نبضة/دقيقة
- ضغط الدم: {context['vitals']['bp']} ملم زئبقي
- الأكسجين: {context['vitals']['spo2']}%
- الحرارة: {context['vitals']['temp']}°C
- ECG: {context['ecg'].get('ar', 'غير متوفر')}
- مستوى الخطر: {context['risk']}"""
            else:
                patient_info = f"""
Patient data:
- Name: {context['name']}, Age: {context['age']}, Gender: {context['gender']}
- Chronic: {', '.join(context['chronic']) if context['chronic'] else 'None'}
- Notes: {context['notes']}
- HR: {context['vitals']['hr']} bpm, BP: {context['vitals']['bp']} mmHg
- SpO2: {context['vitals']['spo2']}%, Temp: {context['vitals']['temp']}°C
- ECG: {context['ecg'].get('en','Not available')}
- Risk: {context['risk']}"""

        final_prompt = f"""{base_prompt}

{patient_info}

{"المريض يسأل" if lang=="ar" else "Patient asks"}: {user_message}

{"أجب بشكل مختصر" if lang=="ar" else "Answer concisely"}."""

        return call_llm(final_prompt)

    # -------- If non-medical question --------
    return "آسف، يمكنني فقط الإجابة على الأسئلة المتعلقة بالصحة." if lang == "ar" else \
            "I'm sorry, I can only answer health-related questions."

# Interfaces
def record_audio(duration=5, samplerate=16000):
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
    sd.wait()
    
    # Save to WAV in memory
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    return buffer.getvalue()

def speak_text(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)  
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
            tts.save(fp.name)
            playsound.playsound(fp.name)
    except Exception as e:
        print(f"❌ Error in TTS: {e}")

# ---------------- Whisper Model ----------------
model = WhisperModel("small", device="cpu", compute_type="int8")

def speech_to_text(audio_file_path):
    segments, info = model.transcribe(audio_file_path, beam_size=5)
    text = " ".join([seg.text for seg in segments])
    return text.strip()

def transcribe_audio_bytes(audio_bytes, ext="wav"):
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    segments, info = model.transcribe(tmp_path, beam_size=5, language=None)  
    text = " ".join([seg.text for seg in segments]).strip()

    detected_lang = info.language  
    return text, detected_lang

def chatbot_voice():
    print("🟢 Voice conversation...")
    while True:
        try:
            audio_bytes = record_audio(5)
            text, detected_lang = transcribe_audio_bytes(audio_bytes, ext="wav")
            print(f"🎤 Voice ({detected_lang}): {text}")

            if text.lower() in ["quit", "خروج", "إنهاء"]:
                print("👋 Goodbye!")
                break

            response = process_message(text)  # هيديك الرد بناءً على اللغة المكتشفة
            print(f"🤖 Bot: {response}")

            # نطق الرد بنفس اللغة
            reply_lang = detect_language(response)  # يستخدم detect_language اللي عندك
            speak_text(response, lang="ar" if reply_lang=="ar" else "en")

        except KeyboardInterrupt:
            print("👋 Conversation ended")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def chatbot_text():
    print("🟢 Text conversation...")
    while True:
        try:
            user_message = input("📝 Write your question: ").strip()
            if user_message.lower() in ["quit", "خروج", "إنهاء"]:
                print("👋 Goodbye!")
                break
            if not user_message:
                continue
            response = process_message(user_message)
            print(f"🤖 Bot: {response}")
        except KeyboardInterrupt:
            print("👋 Conversation ended")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

# Main entry point
if __name__ == "__main__":
    print("💻 Welcome to the Smart Personal Health Assistant!")
    print("🔹 You can ask any medical question or request your health report")

    mode = input("🟢 Choose mode - (1) Voice or (2) Text? ").strip()
    if mode == "1":
        chatbot_voice()
    else:
        chatbot_text()
import os
import re
import json
from openai import OpenAI
from rule_based_system import MediGuardRuleBasedSystem

# =====================
# Setup OpenRouter Client
# =====================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ No OpenRouter API Key found. Please set OPENROUTER_API_KEY.")

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
LLM_MODEL = "openai/gpt-4o-mini"

def get_vitals():
    """Read patient data file"""
    try:
        with open("vitals4.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Patient data not found."}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format in patient_data.json."}

def call_llm(prompt):
    """Call LLM via OpenRouter"""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI medical assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Error calling LLM: {e}")
        return ""

def normalize_name(name, lang="ar"):
    """Translate patient name to match report language"""
    if not name or name == "Unknown":
        return name 

    # Translate to English
    if lang == "en" and re.search(r'[\u0600-\u06FF]', name):
        prompt = f"Translate the following name into English only, no extra text: {name}"
        return call_llm(prompt)

        # Translate to Arabic
    if lang == "ar" and re.search(r'[A-Za-z]', name):
        prompt = f"ترجم الاسم التالي إلى العربية فقط بدون أي إضافات: {name}"
        return call_llm(prompt)

    return name
        
def generate_smart_recommendations(data, lang="ar"):
    """Generate smart recommendations with translation support"""
    try:
        # 1. Patient Data 
        name = data.get("name", "Patient")
        age = data.get("age", "Unknown")
        gender = data.get("gender", "Unknown")
        chronic = ", ".join(data.get("chronic_conditions", []))

        vitals = data.get("vitals", {})
        hr = vitals.get("hr", {}).get("value", "N/A")
        bp_sys = vitals.get("bp", {}).get("systolic", "N/A")
        bp_dia = vitals.get("bp", {}).get("diastolic", "N/A")
        spo2 = vitals.get("spo2", {}).get("value", "N/A")
        temp = vitals.get("temp", {}).get("value", "N/A")
        resp = vitals.get("respiratory_rate", {}).get("value", "N/A")

        last_analysis = data.get("last_analysis", {})
        ecg_info = last_analysis.get("ecg_analysis", {})
        ecg_class = ecg_info.get("class_name", {}).get("en", "Not available")
        risk_level = last_analysis.get("combined_assessment", {}).get("risk_level", "medium")

        # 2. Generate recommendations in English 
        prompt_en = f"""
As a specialist doctor, provide 3 additional specific and precise medical recommendations 
for this patient's case based on the following data:

Patient: {name}, Age: {age}, Gender: {gender}
Chronic conditions: {chronic}

Vital signs:
- Heart rate: {hr} bpm
- Blood pressure: {bp_sys}/{bp_dia} mmHg
- SpO2: {spo2}%
- Temperature: {temp}°C
- Respiratory rate: {resp} breaths/min

ECG analysis: {ecg_class}
Risk level: {risk_level}

⚠ IMPORTANT RULES:
- ALWAYS write in third person about the patient.
- Example: "The patient should..." NOT "You should..."
- ONLY base recommendations on the data given above.
- If some information (such as weight, BMI, HbA1c, cholesterol) is not provided, DO NOT assume or mention it.
- Present 3 bullet points. No intro or conclusion.
- Recommendations must directly reference the provided values and conditions only.
"""
        smart_recs_en = call_llm(prompt_en).split("\n")

        # Clean bullets
        clean_recs = []
        for rec in smart_recs_en:
            rec = rec.strip()
            rec = re.sub(r'^[\-\*\•\d\.\s]+', '', rec)
            rec = rec.replace("**", "")  # Remove markdown styling
            if rec:
                clean_recs.append(rec)


        # 3. Translate to Arabic if requested 
        if lang == "ar" and clean_recs:
            translation_prompt = "ترجم النقاط التالية إلى العربية باحتفاظ بالمعنى الطبي (لا تضيف نجوم أو أي رموز للتنسيق):\n" + "\n".join(clean_recs)
            translated = call_llm(translation_prompt).split("\n")

            translated_clean = []
            for rec in translated:
                rec = rec.strip()
                rec = re.sub(r'^[\-\*\•\d\.\s]+', '', rec)
                rec = rec.replace("**", "")  # Remove markdown styling
                rec = re.sub(r'[^\u0600-\u06FFa-zA-Z0-9\s\.,;:!?()%$@#&*\-+=\[\]{}]', '', rec)
                if rec:
                    translated_clean.append(rec)
            return translated_clean

        return clean_recs

    except Exception as e:
        print(f"Error generating smart recommendations: {e}")
        return []

def generate_report(data, lang="ar"):
    """
    Generate a detailed report based on patient data
    using RuleBasedSystem for personalized analysis, with a structured format.
    """

    if "error" in data:
        return "❌ لم يتم العثور على بيانات المريض" if lang == "ar" else "❌ Patient data not found"
    
    # Translation dictionary
    ar_to_en = {
        "ذكر": "male",
        "أنثى": "female",
        "غير محدد": "unspecified",
        "ضغط الدم الانبساطي": "Diastolic BP",
        "ضغط الدم الانقباضي": "Systolic BP",
        "درجة الحرارة": "Temperature",
        "النبض": "Pulse",
        "الأكسجين": "SpO2",
        "معدل التنفس": "Respiratory rate",
        "طبيعي": "normal",
        "غير طبيعي": "abnormal",
        "يحتاج مراقبة": "needs monitoring",
        "يحتاج تدخل": "needs intervention",
        "منخفض": "low",
        "متوسط": "medium",
        "مرتفع": "high"
    }

    en_to_ar = {
        "Respiration rate (per minute)": "معدل التنفس (في الدقيقة)",
        "Respiratory rate": "معدل التنفس",
        "SpO2 (%)": "الأكسجين (%)",
        "SpO2": "الأكسجين",
        "Systolic BP (mmHg)": "ضغط الدم الانقباضي (ملم زئبق)",
        "Diastolic BP": "ضغط الدم الانبساطي (ملم زئبق)",
        "Blood Pressure": "ضغط الدم",
        "Pulse (per minute)": "النبض (في الدقيقة)",
        "Pulse": "النبض",
        "Temperature (°C)": "درجة الحرارة (°م)",
        "Temperature": "درجة الحرارة",
        "abnormal": "غير طبيعي",
        "normal": "طبيعي",
        "needs intervention": "يحتاج تدخل",
        "needs monitoring": "يحتاج مراقبة",
        "low": "منخفض",
        "medium": "متوسط",
        "high": "مرتفع"
    }

    def translate(text):
        if not isinstance(text, str):
            return text

        if lang == "ar":
            for en_word, ar_word in en_to_ar.items():
                text = text.replace(en_word, ar_word)
        elif lang == "en":
            for ar_word, en_word in ar_to_en.items():
                text = text.replace(ar_word, en_word)

        return text
    
    def normalize_gender_raw(gender_raw, lang):
        """Normalize incoming gender values and return a language-appropriate label.
        Returns Arabic labels when lang=='ar', English when lang=='en'.
        """
        if not isinstance(gender_raw, str) or not gender_raw.strip():
            return 'غير محدد' if lang == 'ar' else 'unspecified'

        g = gender_raw.strip().lower()
        # common variants
        male_values = {'male', 'm', 'ذكر', 'man', 'masculine'}
        female_values = {'female', 'f', 'أنثى', 'woman', 'feminine'}

        if g in male_values:
            return 'ذكر' if lang == 'ar' else 'male'
        if g in female_values:
            return 'أنثى' if lang == 'ar' else 'female'

        # fallback
        return 'غير محدد' if lang == 'ar' else 'unspecified'
    
    # --- Basic patient info ---
    name = data.get("name", "Unknown")
    age = data.get("age", "N/A")
    gender_raw = data.get("gender", "")
    vitals = data.get("vitals", {})

    # --- Prepare data for RuleBasedSystem ---
    vital_signs = {
        'respiratory_rate': vitals.get("respiratory_rate", {}).get("value", None),
        'spo2': vitals.get("spo2", {}).get("value", None),
        'systolic_bp': vitals.get("bp", {}).get("systolic", None),
        'diastolic_bp': vitals.get("bp", {}).get("diastolic", None),
        'pulse': vitals.get("hr", {}).get("value", None),
        'temperature': vitals.get("temp", {}).get("value", None)
    }

    vital_signs = {k: v for k, v in vital_signs.items() if v is not None}
    rule_system = MediGuardRuleBasedSystem(language=lang)

    patient_data = {'patient_id': name, 'vital_signs': vital_signs}
    analysis = rule_system.comprehensive_vital_analysis(patient_data)

    sensor_errors = analysis.get("sensor_errors", [])
    error_msgs = []
    if sensor_errors:
        if lang == "ar":
            for err in sensor_errors:
                sensor_name = translate(err['sensor'])  
                if "disconnected" in err['error']:
                    error_msgs.append(f"جهاز {sensor_name} غير متصل. رجاءً تأكد من تشغيله وتوصيله بشكل صحيح.")
                else:  # implausible
                    error_msgs.append(f"السنسور الخاص بـ {sensor_name} غير مركب بشكل صحيح . تأكد من تركيبه فى المكان الصحيح.")
        else:
            for err in sensor_errors:
                if "disconnected" in err['error']:
                    error_msgs.append(f"{err['sensor']} device is not connected. Please check the device connection.")
                else:  # implausible
                    error_msgs.append(f"The sensor for {err['sensor']} seems improperly attached or giving implausible readings. Please reattach correctly.")

    # Normalize gender once for the chosen language so `gender` is always defined
    gender = normalize_gender_raw(str(gender_raw), lang)


    last_analysis = data.get("last_analysis", {})

    news_score = analysis['news_analysis']['total_news_score']
    risk_level = translate(analysis['news_analysis']['risk_category']['level'])
    recommendations = [translate(r) for r in analysis['recommendations']]
    additional_assessments = analysis['additional_assessments']

    positives, warnings, notes = [], [], []

    for vital, details in analysis['news_analysis']['individual_scores'].items():
        param = translate(details['parameter'])
        val = details['value']
        if details['score'] == 0:
            positives.append(f"{param} {translate('normal')} ({val})")        
        elif details['score'] >= 2:
            warnings.append(f"{param} {translate('abnormal')} ({val}) - {translate('needs intervention')}")
        else:
            notes.append(f"{param} {translate('needs monitoring')} ({val})")

    if 'diastolic_bp' in additional_assessments:
        dbp = additional_assessments['diastolic_bp']
        param = translate("ضغط الدم الانبساطي")
        if dbp['status'] == 'normal':
            positives.append(f"{param} {translate('normal')} ({dbp['value']})")
        else:
            warnings.append(f"{param} {translate(dbp['status'])} ({dbp['value']})")

    if 'critical_combinations' in additional_assessments:
        for combo in additional_assessments['critical_combinations']:
            warnings.append(f"⚠️ {translate(combo['description'])}")

    name = data.get("name", "Unknown")
    name = normalize_name(name, lang)  

    warnings_notes = warnings + error_msgs

    # Arabic report
    if lang == "ar":
        report = f"""
🩺 تحليل شامل لحالة المريض

المريض: {name}  
العمر: {age}  
الجنس: {gender}  

⚠️ مستوى الخطر العام: {risk_level} 

📋 تقييم القراءات الحيوية  
{('✅ مؤشرات إيجابية:\n' + "\n".join([f"- {p}" for p in positives]) ) if positives else ''}
{('⚠️ تحذيرات:\n' + "\n".join([f"- {dn}" for dn in warnings_notes])+ "\n") if warnings_notes else ''}
{('🔎 مؤشرات تحتاج متابعة\n' + "\n".join([f"- {n}" for n in notes])) + "\n"   if notes else ''}
{('📋 توصيات للحفاظ على المستوى:\n' + "\n" .join([f"- {r}" for r in recommendations]) + "") if recommendations else ''}

📝 ملاحظة طبية
- يُنصح بأن يستمر المريض في القياسات الدورية  
- يجب متابعة أي تغييرات مفاجئة في حالة المريض  
- يُفضّل الاحتفاظ بسجل القراءات الخاصة بالمريض
"""

    # English report 
    else:
        report = f"""
🩺 Comprehensive Patient Analysis

Patient: {name}  
Age: {age}  
Gender: {gender}  

⚠️ Overall Risk Level: {risk_level}

📋 Vital Signs Evaluation 
{('✅ Positive Health Indicators:\n' + "\n".join([f"- {p}" for p in positives])) if positives else ''}
{('⚠️ Urgent Warnings:\n' + "\n".join([f"- {dn}" for dn in warnings_notes])+ "\n") if warnings_notes else ''}
{('🔎 Observations for Monitoring:\n' + "\n".join([f"- {n}" for n in notes])) + "\n" if notes else ''}
{('📋 Recommendations to Maintain Level:\n' + "\n".join([f"- {r}" for r in recommendations]) + "") if recommendations else ''}

📝 Medical Note 
- Continue regular monitoring  
- Watch for sudden changes  
- Keep a record of readings
"""

    # Add smart recommendations 
    smart_recs = generate_smart_recommendations(data, lang=lang)
    if smart_recs:
        smart_recs_section = "\n" + ("💡 توصيات :\n" if lang == "ar" else "💡 Recommendations:\n")
        smart_recs_section += "\n".join([f"- {translate(rec)}" for rec in smart_recs])
        report += smart_recs_section

    return report.strip()

def generate_summary_from_report(data, lang="ar"):
    """
    Generate a short, natural summary of the patient's condition
    directly from the detailed report using the LLM.
    """

    # Fix patient name normalization based on language
    if "name" in data:
        data["name"] = normalize_name(data["name"], lang)
        
    full_report = generate_report(data, lang=lang)

    if lang == "ar":
        prompt = f"""
هذه بيانات حالة المريض بالتفصيل:

{full_report}

❗ المطلوب: 
- أعطِ ملخص قصير (2-3 جمل) عن حالة المريض بصيغة الغائب.
- وضّح الحالة العامة (جيدة/مستقرة/بحاجة متابعة...) مع أهم الملاحظات فقط.
- لا تُكرر النص الحرفي، اكتبه بصياغتك.
"""
    else:
        prompt = f"""
Here is the full detailed patient report:

{full_report}

❗ Task:
- Provide a short summary (2-3 sentences) of the patient's condition in third-person.
- Describe the overall health status (e.g. stable / good / needs close monitoring) with only the key points.
- Do not copy-paste; rephrase naturally.
"""

    return call_llm(prompt)

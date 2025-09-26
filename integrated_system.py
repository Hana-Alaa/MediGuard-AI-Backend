import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from rule_based_system import MediGuardRuleBasedSystem
from datetime import datetime
import numpy as np

class IntegratedMediGuardSystem:
    def __init__(self, language="en"):
        """
        Integrated MediGuard System:
        Combines:
        1. ECG Deep Learning model
        2. Rule-based system for vital signs
        to provide a full patient analysis
        """
        # Initialize system language (English or Arabic)
        self.language = language
        
        # Rule-based system (for vitals like blood pressure, oxygen, etc.)
        self.rule_based_system = MediGuardRuleBasedSystem(language=language)
        
        # Load trained ECG classification model
        self.ecg_model = tf.keras.models.load_model("model.h5")

        self.ecg_scaler = MinMaxScaler()
        
        # ECG class mappings (simplified explanations for non-doctors)
        self.ecg_classes = {
            0: {"en": "Normal heartbeat", "ar": "ضربات قلب طبيعية"},
            1: {"en": "Supraventricular arrhythmia (irregular beats above the ventricles)",
                "ar": "عدم انتظام ضربات القلب فوق البطين (اضطراب في الجزء العلوي للقلب)"}, 
            2: {"en": "Ventricular arrhythmia (dangerous irregular beats from ventricles)",
                "ar": "عدم انتظام ضربات القلب البطيني (خطير - من الجزء السفلي للقلب)"},
            3: {"en": "Fusion beats (mixed signals between normal and abnormal beats)",
                "ar": "ضربات قلب مدمجة (خليط بين الطبيعي وغير الطبيعي)"},
            4: {"en": "Unknown / Unclear signal", "ar": "غير معروف / إشارة غير واضحة"}
        }
        
        # Risk levels for ECG classifications
        self.ecg_risk_levels = {
            0: 'low',      # Normal = Low risk
            1: 'medium',   # Supraventricular = Moderate risk
            2: 'high',     # Ventricular = High risk (dangerous)
            3: 'medium',   # Fusion = Moderate risk
            4: 'high'      # Unknown = treat as high risk
        }

        # Translated recommendations (simplified for patients)
        self.ecg_translations = {
            "ventricular": {
                "en": "🚨 Dangerous heartbeat pattern detected - see a heart doctor immediately",
                "ar": "🚨 تم رصد نمط خطير في ضربات القلب - يجب مراجعة طبيب القلب فوراً"
            },
            "supraventricular": {
                "en": "⚠️ Irregular heartbeat detected (upper heart area) - follow up with a doctor",
                "ar": "⚠️ تم رصد عدم انتظام في ضربات القلب (الجزء العلوي) - يلزم متابعة مع الطبيب"
            },
            "fusion": {
                "en": "📋 Mixed heartbeat signals detected - medical check advised",
                "ar": "📋 تم رصد خليط في إشارات ضربات القلب - ينصح بمتابعة طبية"
            },
            "low_conf": {
                "en": "🔍 Low confidence in result - repeat ECG recording recommended",
                "ar": "🔍 النتيجة غير مؤكدة - يفضل إعادة تسجيل تخطيط القلب"
            },
            "emergency": {
                "en": "🆘 Emergency detected - go to hospital immediately",
                "ar": "🆘 حالة طوارئ - التوجه للمستشفى فوراً"
            }
        }

    def analyze_ecg_signal(self, ecg_data):
        """Analyze ECG signal using CNN model"""
        try:
            # Preprocess ECG data (normalize + reshape to fit model input)
            ecg_scaled = self.ecg_scaler.fit_transform(ecg_data.reshape(1, -1))
            ecg_reshaped = ecg_scaled.reshape(1, 187, 1)
            
            # Predict class probabilities
            prediction_probs = self.ecg_model.predict(ecg_reshaped, verbose=0)
            predicted_class = np.argmax(prediction_probs)
            confidence = float(np.max(prediction_probs))
            
            return {
                'class': predicted_class,
                'class_name': self.ecg_classes[predicted_class][self.language],
                'confidence': confidence,
                'risk_level': self.ecg_risk_levels[predicted_class],
                'all_probabilities': prediction_probs.tolist()
            }
        except Exception as e:
            # If error, treat as high risk
            return {
                'class': 4,
                'class_name': {"en": "Error in ECG Analysis", "ar": "خطأ في تحليل تخطيط القلب"}[self.language],
                'confidence': 0.0,
                'risk_level': 'high',
                'error': str(e)
            }

    def comprehensive_patient_analysis(self, patient_data):
        """
        Complete patient analysis combining:
        - ECG Deep Learning model
        - Vital Signs Rule-based system
        """
        results = {
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # 1. ECG Analysis
        if 'ecg_signal' in patient_data:
            ecg_analysis = self.analyze_ecg_signal(patient_data['ecg_signal'])
            results['ecg_analysis'] = ecg_analysis
        else:
            results['ecg_analysis'] = None
        
        # 2. Vital Signs Analysis
        if 'vital_signs' in patient_data:
            vital_analysis = self.rule_based_system.comprehensive_vital_analysis(patient_data)
            results['vital_signs_analysis'] = vital_analysis
        else:
            results['vital_signs_analysis'] = None
        
        # 3. Combined Risk
        combined_assessment = self._calculate_combined_risk(results)
        results['combined_assessment'] = combined_assessment
        
        # 4. Unified Recommendations
        unified_recommendations = self._generate_unified_recommendations(results)
        results['unified_recommendations'] = unified_recommendations
        
        return results

    def _calculate_combined_risk(self, analysis_results):
        """Combine ECG + Vital signs into a single risk score"""
        risk_scores = []
        risk_factors = []
        
        # ECG Risk
        if analysis_results.get('ecg_analysis'):
            ecg_risk = analysis_results['ecg_analysis']['risk_level']
            risk_map = {'low': 1, 'medium': 2, 'high': 3}
            ecg_score = risk_map.get(ecg_risk, 2)
            risk_scores.append(ecg_score * 0.4)  # ECG = 40% weight
            
            if ecg_risk in ['medium', 'high']:
                risk_factors.append(f"ECG issue: {analysis_results['ecg_analysis']['class_name']}")

        # Vital Signs Risk
        if analysis_results.get('vital_signs_analysis'):
            vital_analysis = analysis_results['vital_signs_analysis']
            news_score = vital_analysis['news_analysis']['total_news_score']
            
            # Convert NEWS score into risk
            if news_score >= 7:
                vital_risk_score = 3
            elif news_score >= 5:
                vital_risk_score = 2
            else:
                vital_risk_score = 1
            
            risk_scores.append(vital_risk_score * 0.6)  # Vitals = 60% weight
            
            # Add critical combinations if found
            if vital_analysis.get('additional_assessments', {}).get('critical_combinations'):
                risk_factors.extend([
                    combo['description'] 
                    for combo in vital_analysis['additional_assessments']['critical_combinations']
                ])
        
        # Final combined risk level
        if risk_scores:
            combined_score = sum(risk_scores)
            if combined_score >= 2.5:
                final_risk = 'high'
                alert_color = 'red'
            elif combined_score >= 1.5:
                final_risk = 'medium'
                alert_color = 'yellow'
            else:
                final_risk = 'low'
                alert_color = 'green'
        else:
            final_risk = 'unknown'
            alert_color = 'gray'
        
        return {
            'combined_risk_level': final_risk,
            'alert_color': alert_color,
            'risk_score': sum(risk_scores) if risk_scores else 0,
            'contributing_factors': risk_factors,
            'requires_immediate_attention': final_risk == 'high'
        }

    def _generate_unified_recommendations(self, analysis_results):
        """Generate clear recommendations for patient/doctors"""
        recommendations = []
        
        # ECG-specific advice
        if analysis_results.get('ecg_analysis'):
            ecg_class = analysis_results['ecg_analysis']['class']
            if ecg_class in [2, 4]:  # Ventricular or Unknown
                recommendations.append(self.ecg_translations["ventricular"][self.language])
            elif ecg_class == 1:  # Supraventricular
                recommendations.append(self.ecg_translations["supraventricular"][self.language])
            elif ecg_class == 3:  # Fusion
                recommendations.append(self.ecg_translations["fusion"][self.language])
        
        # Vital signs advice
        if analysis_results.get('vital_signs_analysis'):
            vital_recs = analysis_results['vital_signs_analysis'].get('recommendations', [])
            recommendations.extend(vital_recs)
        
        # Emergency advice if needed
        combined_risk = analysis_results.get('combined_assessment', {})
        if combined_risk.get('requires_immediate_attention'):
            recommendations.insert(0, self.ecg_translations["emergency"][self.language])
        
        return recommendations


if __name__ == "__main__":
    # Example run
    system = IntegratedMediGuardSystem(language="ar")

    patient_data = {
        'patient_id': 'TEST_INTEGRATED',
        'ecg_signal': np.random.normal(0, 1, 187),
        'vital_signs': {
            'respiratory_rate': 24,  # breathing rate
            'spo2': 91,              # blood oxygen %
            'systolic_bp': 95,       # upper blood pressure
            'diastolic_bp': 65,      # lower blood pressure
            'pulse': 105,            # heart rate
            'temperature': 37.8      # body temperature (Celsius)
        }
    }
    results = system.comprehensive_patient_analysis(patient_data)
    print(results)

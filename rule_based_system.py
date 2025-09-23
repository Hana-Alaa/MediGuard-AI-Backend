import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any

class MediGuardRuleBasedSystem:
    def __init__(self, language="en"):
        """
        Initialize the Rule-Based Medical Assessment System
        Based on NEWS (National Early Warning Score) and clinical guidelines
        """
        self.language = language
        # NEWS Score Thresholds based on clinical research [1]
        self.news_thresholds = {
            'respiratory_rate': {
                'ranges': [(float('-inf'), 8, 3), (9, 11, 1), (12, 20, 0), (21, 24, 2), (25, float('inf'), 3)],
                'parameter': 'Respiration rate (per minute)'
            },
            'spo2': {
                'ranges': [(float('-inf'), 91, 3), (92, 93, 2), (94, 95, 1), (96, float('inf'), 0)],
                'parameter': 'SpO2 (%)'
            },
            'systolic_bp': {
                'ranges': [(float('-inf'), 90, 3), (91, 100, 2), (101, 110, 1), (111, 219, 0), (220, float('inf'), 3)],
                'parameter': 'Systolic BP (mmHg)'
            },
            'pulse': {
                'ranges': [(float('-inf'), 40, 3), (41, 50, 1), (51, 90, 0), (91, 110, 1), (111, 130, 2), (131, float('inf'), 3)],
                'parameter': 'Pulse (per minute)'
            },
            'temperature': {
                'ranges': [(float('-inf'), 35.0, 3), (35.1, 36.0, 1), (36.1, 38.0, 0), (38.1, 39.0, 1), (39.1, float('inf'), 2)],
                'parameter': 'Temperature (°C)'
            }
        }
        
        # Risk categories based on NEWS score [1]
        self.risk_categories = {
            'low': {'range': (0, 4), 'response': 'Ward-based response', 'alert_level': 'green'},
            'medium': {'range': (5, 6), 'response': 'Key threshold for urgent response', 'alert_level': 'yellow'},
            'high': {'range': (7, float('inf')), 'response': 'Urgent or emergency response', 'alert_level': 'red'}
        }
        
        # Additional clinical thresholds
        self.clinical_thresholds = {
            'diastolic_bp': {
                'hypotension': 60,
                'normal_max': 89,
                'stage1_hypertension': 99,
                'stage2_hypertension': 109
            }
        }
        # Translation for recommendations
        self.translations = {
            "urgent": {"en": "Urgent medical intervention required - call the doctor immediately",
                        "ar": "مطلوب تدخل طبي عاجل - اتصل بالطبيب فوراً"},
            "monitor_15": {"en": "Monitor vital signs every 15 minutes",
                        "ar": "مراقبة العلامات الحيوية كل 15 دقيقة"},
            "medium": {"en": " Medical evaluation required - see a doctor soon",
                        "ar": " مطلوب تقييم طبي - راجع الطبيب في أقرب وقت"},
            "monitor_30": {"en": " Monitor vital signs every 30 minutes",
                        "ar": " مراقبة العلامات الحيوية كل 30 دقيقة"},
            "normal": {"en": " Vital signs are within acceptable range",
                        "ar": " العلامات الحيوية في المعدل المقبول"},
            "routine": {"en": " Routine monitoring every 4-6 hours",
                        "ar": " متابعة روتينية كل 4-6 ساعات"},
            "critical_combo": {"en": " {desc} - Immediate intervention required",
                        "ar": " {desc} - تدخل فوري مطلوب"}
        }

    def _translate(self, key, **kwargs):
        return self.translations[key][self.language].format(**kwargs)
    
    def calculate_news_score(self, vital_signs: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate NEWS score based on vital signs
        Returns detailed breakdown and overall assessment
        """
        scores = {}
        total_score = 0
        
        for vital, value in vital_signs.items():
            if vital in self.news_thresholds:
                score = self._get_news_parameter_score(vital, value)
                scores[vital] = {
                    'value': value,
                    'score': score,
                    'parameter': self.news_thresholds[vital]['parameter']
                }
                total_score += score
        
        # Determine risk category
        risk_category = self._determine_risk_category(total_score)
        
        return {
            'individual_scores': scores,
            'total_news_score': total_score,
            'risk_category': risk_category,
            'timestamp': datetime.now().isoformat()
        }

    def _check_sensor_integrity(self, vital_signs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if devices are connected and readings are plausible.
        Adds errors if device missing or values are out of range.
        """
        errors = []

        # Heart Rate
        hr = vital_signs.get('pulse') or vital_signs.get('hr')

        if hr is None:
            errors.append({"sensor": "Heart Rate", "error": "device_disconnected"})
        elif not (30 <= hr <= 220):
            errors.append({"sensor": "Heart Rate", "error": f"implausible_value: {hr}"})

        # SpO2
        spo2 = vital_signs.get('spo2')
        if spo2 is None:
            errors.append({"sensor": "SpO2", "error": "device_disconnected"})
        elif not (50 <= spo2 <= 100):
            errors.append({"sensor": "SpO2", "error": f"implausible_value: {spo2}"})

        # Temperature
        temp = vital_signs.get('temperature') or vital_signs.get('temp')
        if temp is None:
            errors.append({"sensor": "Temperature", "error": "device_disconnected"})
        elif not (30 <= temp <= 43):
            errors.append({"sensor": "Temperature", "error": f"implausible_value: {temp}"})

        # Blood Pressure
        sbp = vital_signs.get('systolic_bp')
        dbp = vital_signs.get('diastolic_bp')
        if sbp is None or dbp is None:
            errors.append({"sensor": "Blood Pressure", "error": "device_disconnected"})
        elif not (60 <= sbp <= 250 and 30 <= dbp <= 150):
            errors.append({"sensor": "Blood Pressure", "error": f"implausible_value: {sbp}/{dbp}"})

        # Respiratory rate
        rr = vital_signs.get('respiratory_rate')
        if rr is None:
            errors.append({"sensor": "Respiratory Rate", "error": "device_disconnected"})
        elif not (5 <= rr <= 60):
            errors.append({"sensor": "Respiratory Rate", "error": f"implausible_value: {rr}"})

        return errors
    
    def _get_news_parameter_score(self, vital: str, value: float) -> int:
        """Get NEWS score for a specific vital sign parameter"""
        ranges = self.news_thresholds[vital]['ranges']
        
        for min_val, max_val, score in ranges:
            if min_val <= value <= max_val:
                return score
        
        return 0  

    def _determine_risk_category(self, total_score: int) -> Dict[str, Any]:
        """Determine risk category based on total NEWS score"""
        for category, details in self.risk_categories.items():
            min_score, max_score = details['range']
            if min_score <= total_score <= max_score:
                return {
                    'level': category,
                    'response': details['response'],
                    'alert_level': details['alert_level'],
                    'total_score': total_score
                }
        
        return {'level': 'unknown', 'response': 'Review required', 'alert_level': 'gray', 'total_score': total_score}

    def comprehensive_vital_analysis(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive analysis combining NEWS score with additional clinical rules,
        while reporting sensor errors and ignoring disconnected devices in scoring.
        """
        vital_signs = patient_data.get('vital_signs', {})

        # 1) Check sensor errors
        errors = self._check_sensor_integrity(vital_signs)

        # 2) Clean vital_signs 
        cleaned_vital_signs = vital_signs.copy()
        for err in errors:
            if "disconnected" in err["error"] or "implausible_value" in err["error"]:
                if err["sensor"] == "Blood Pressure":
                    cleaned_vital_signs["systolic_bp"] = None
                    cleaned_vital_signs["diastolic_bp"] = None
                if err["sensor"] == "Blood Pressure":
                    cleaned_vital_signs["systolic_bp"] = None
                    cleaned_vital_signs["diastolic_bp"] = None
                elif err["sensor"] == "SpO2":
                    cleaned_vital_signs["spo2"] = None
                elif err["sensor"] == "Heart Rate":
                    cleaned_vital_signs["pulse"] = None
                elif err["sensor"] == "Temperature":
                    cleaned_vital_signs["temperature"] = None
                elif err["sensor"] == "Respiratory Rate":
                    cleaned_vital_signs["respiratory_rate"] = None

        # 3) Filter out None before NEWS score
        filtered_vitals = {k: v for k, v in cleaned_vital_signs.items() if v is not None}

        # 4) Calculate NEWS score
        news_analysis = self.calculate_news_score(filtered_vitals)

        # 5) Additional rules (using cleaned_vital_signs)
        additional_assessments = self._additional_clinical_rules(filtered_vitals)

        # 6) Generate recommendations
        recommendations = self._generate_recommendations(news_analysis, additional_assessments)

        # 7) Final alert
        final_alert = self._determine_final_alert(news_analysis, additional_assessments)

        # 8) Return full analysis
        return {
            "patient_id": patient_data.get("patient_id", "unknown"),
            "assessment_time": datetime.now().isoformat(),
            "news_analysis": news_analysis,
            "additional_assessments": additional_assessments,
            "recommendations": recommendations,
            "final_alert": final_alert,
            "requires_immediate_attention": final_alert["level"] in ["red", "critical"],
            "sensor_errors": errors if errors else [],
            "cleaned_vital_signs": cleaned_vital_signs  # علشان يظهر في JSON إن القيم null
        }

    def _additional_clinical_rules(self, vital_signs: Dict[str, float]) -> Dict[str, Any]:
        """Additional clinical rule-based assessments beyond NEWS score"""
        assessments = {}
        
        # Blood pressure assessment (diastolic)
        if 'diastolic_bp' in vital_signs:
            diastolic = vital_signs['diastolic_bp']
            if diastolic < self.clinical_thresholds['diastolic_bp']['hypotension']:
                assessments['diastolic_bp'] = {'status': 'hypotension', 'severity': 'medium', 'value': diastolic}
            elif diastolic > self.clinical_thresholds['diastolic_bp']['stage2_hypertension']:
                assessments['diastolic_bp'] = {'status': 'severe_hypertension', 'severity': 'high', 'value': diastolic}
            elif diastolic > self.clinical_thresholds['diastolic_bp']['stage1_hypertension']:
                assessments['diastolic_bp'] = {'status': 'moderate_hypertension', 'severity': 'medium', 'value': diastolic}
            else:
                assessments['diastolic_bp'] = {'status': 'normal', 'severity': 'low', 'value': diastolic}
        
        # Critical combinations check
        critical_combinations = self._check_critical_combinations(vital_signs)
        if critical_combinations:
            assessments['critical_combinations'] = critical_combinations
        
        return assessments

    def _check_critical_combinations(self, vital_signs: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for critical combinations of vital signs"""
        critical_alerts = []
        
        # Low SpO2 + High respiratory rate (potential respiratory distress)
        if vital_signs.get('spo2', 100) < 92 and vital_signs.get('respiratory_rate', 15) > 22:
            critical_alerts.append({
                'type': 'respiratory_distress',
                'description': {"en": "Low oxygen saturation combined with high respiratory rate",
                                "ar": "انخفاض تشبع الأكسجين مع زيادة معدل التنفس"}[self.language],
                'severity': 'critical'
            })
        
        # Low blood pressure + High heart rate (potential shock)
        if vital_signs.get('systolic_bp', 120) < 90 and vital_signs.get('pulse', 70) > 100:
            critical_alerts.append({
                'type': 'potential_shock',
                'description': {"en": "Low blood pressure with compensatory tachycardia",
                                "ar": "ضغط دم منخفض مع سرعة ضربات القلب"}[self.language],
                'severity': 'critical'
            })
        
        # High temperature + High heart rate (possible severe infection)
        if vital_signs.get('temperature', 37) > 38.3 and vital_signs.get('pulse', 70) > 110:
            critical_alerts.append({
                'type': 'potential_sepsis',
                'description': {"en": "High fever with tachycardia - sepsis consideration",
                                "ar": "ارتفاع الحرارة مع سرعة ضربات القلب - احتمال عدوى خطيرة"}[self.language],
                'severity': 'high'
            })
        
        return critical_alerts

    def _generate_recommendations(self, news_analysis: Dict, additional_assessments: Dict) -> List[str]:
        """Generate clinical recommendations based on analysis"""
        recommendations = []
        
        risk_level = news_analysis['risk_category']['level']
        
        if risk_level == 'high':
            recommendations.append(self._translate("urgent"))
            recommendations.append(self._translate("monitor_15"))
        elif risk_level == 'medium':
            recommendations.append(self._translate("medium"))
            recommendations.append(self._translate("monitor_30"))
        else:
            recommendations.append(self._translate("normal"))
            recommendations.append(self._translate("routine"))
        
        # Additional specific recommendations
        for assessment_type, details in additional_assessments.items():
            if assessment_type == 'critical_combinations':
                for combo in details:
                    if combo['severity'] == 'critical':
                        recommendations.append(self._translate("critical_combo", desc=combo['description']))
        
        return recommendations

    def _determine_final_alert(self, news_analysis: Dict, additional_assessments: Dict) -> Dict[str, Any]:
            """Determine the final alert level for the patient"""
            news_alert = news_analysis['risk_category']['alert_level']
            
            # Check for critical combinations
            has_critical = any(
                combo.get('severity') == 'critical' 
                for combo in additional_assessments.get('critical_combinations', [])
            )
            
            if has_critical:
                return {'level': 'critical', 'color': 'purple', 'action': 'immediate_intervention'}
            elif news_alert == 'red':
                return {'level': 'red', 'color': 'red', 'action': 'urgent_response'}
            elif news_alert == 'yellow':
                return {'level': 'yellow', 'color': 'yellow', 'action': 'prompt_assessment'}
            else:
                return {'level': 'green', 'color': 'green', 'action': 'routine_monitoring'}

if __name__ == "__main__":
    # Run the demo
    # run_system_demo()
    
    # Example of how to use with real data
    print("\n" + "="*60)
    print("Example with Custom Patient Data")
    print("="*60)
    
    mediguard = MediGuardRuleBasedSystem(language="ar")
    
    # patient data
    patient_data = {
        'patient_id': 'REAL_PATIENT_001',
        'vital_signs': {
            'respiratory_rate': 18,    # breaths per minute
            'spo2': 95,               # oxygen saturation %
            'systolic_bp': 140,       # systolic blood pressure mmHg
            'diastolic_bp': 90,       # diastolic blood pressure mmHg
            'pulse': 88,              # heart rate bpm
            'temperature': 37.8       # temperature °C
        }
    }
    
    # Run analysis
    analysis = mediguard.comprehensive_vital_analysis(patient_data)
    
    # print results
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
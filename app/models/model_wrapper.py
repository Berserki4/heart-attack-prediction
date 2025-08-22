import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Any

class SafeXGBClassifier:
    """Безопасная обертка вокруг XGBClassifier для обхода проблем версионности"""
    
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.feature_names = [
            'Age', 'Cholesterol', 'Heart rate', 'Exercise Hours Per Week',
            'Systolic blood pressure', 'Diastolic blood pressure', 'Troponin',
            'Troponin_log', 'Sedentary Hours Per Day', 'Triglycerides',
            'Sleep Hours Per Day', 'Diabetes', 'Smoking', 'Obesity',
            'Family History', 'Previous Heart Problems', 'Gender', 'Diet'
        ]
    
    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([data])
        df['Troponin_log'] = np.log(df['Troponin'] + 1e-6)
        return df[self.feature_names]
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, float]:
        try:
            # Обход всех проблем версионности
            input_df = self.preprocess_input(data)
            
            # Достаем только классификатор из пайплайна
            classifier = self.model.named_steps['classifier']
            
            # Используем низкоуровневый predict_proba для обхода sklearn проблем
            preprocessed_data = self.model.named_steps['preprocessor'].transform(input_df)
            prediction = classifier.predict_proba(preprocessed_data)[0]
            
            return {
                'risk_score': float(prediction[1]),
                'risk_class': int(prediction[1] > 0.5),
                'confidence': float(np.max(prediction))
            }
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")

# Создаем глобальный экземпляр модели
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_fixed_model.pkl')
model = SafeXGBClassifier(MODEL_PATH)
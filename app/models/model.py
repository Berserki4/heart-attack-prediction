import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Any

class HeartAttackModel:
    def __init__(self, model_path: str):
        # Абсолютный путь к модели - поднимаемся на ДВА уровня выше
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_model_path = os.path.join(BASE_DIR, 'models', 'final_fixed_model.pkl')
        
        # ОТЛАДОЧНАЯ ИНФОРМАЦИЯ
        print("=" * 50)
        print("ОТЛАДКА ПУТЕЙ:")
        print(f"Файл model.py находится: {os.path.abspath(__file__)}")
        print(f"BASE_DIR: {BASE_DIR}")
        print(f"Ожидаемый путь к модели: {full_model_path}")
        print(f"Существует ли файл: {os.path.exists(full_model_path)}")
        print("=" * 50)
        
        # Проверяем существует ли файл модели
        if not os.path.exists(full_model_path):
            raise FileNotFoundError(f"Модель не найдена по пути: {full_model_path}")
        
        # Загружаем обученную модель
        self.model = joblib.load(full_model_path)
        self.feature_names = [
            'Age', 'Cholesterol', 'Heart rate', 'Exercise Hours Per Week',
            'Systolic blood pressure', 'Diastolic blood pressure', 'Troponin',
            'Troponin_log', 'Sedentary Hours Per Day', 'Triglycerides',
            'Sleep Hours Per Day', 'Diabetes', 'Smoking', 'Obesity',
            'Family History', 'Previous Heart Problems', 'Gender', 'Diet'
        ]
        print("✅ Модель успешно загружена!")
    
    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame([data])
        df['Troponin_log'] = np.log(df['Troponin'] + 1e-6)
        return df[self.feature_names]
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, float]:
        try:
            input_df = self.preprocess_input(data)
            prediction = self.model.predict_proba(input_df)[0]
            
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
model = HeartAttackModel(MODEL_PATH)
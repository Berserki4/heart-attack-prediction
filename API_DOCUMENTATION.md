# Heart Attack Prediction API - Документация

## Основные URL:

- **Главная страница**: http://localhost:8000
- **Документация Swagger**: http://localhost:8000/docs  
- **Health check**: http://localhost:8000/health
- **Предсказание**: http://localhost:8000/predict

## Пример запроса:

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Age": 45,
  "Cholesterol": 200,
  "Heart_rate": 75,
  "Exercise_Hours_Per_Week": 3,
  "Systolic_blood_pressure": 120,
  "Diastolic_blood_pressure": 80,
  "Troponin": 0.01,
  "Sedentary_Hours_Per_Day": 8,
  "Triglycerides": 150,
  "Sleep_Hours_Per_Day": 7,
  "Diabetes": 0,
  "Smoking": 1,
  "Obesity": 0,
  "Family_History": 1,
  "Previous_Heart_Problems": 0,
  "Gender": "Male",
  "Diet": 1
}'
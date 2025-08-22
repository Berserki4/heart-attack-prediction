#  Heart Attack Risk Prediction API

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3776AB?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)

FastAPI веб-приложение для предсказания риска сердечного приступа на основе медицинских данных пациентов с использованием машинного обучения.

## О проекте

Проект разработан в рамках технического задания М1. Включает полный цикл: от исследования данных и обучения модели до развертывания production-ready API.

### Ключевые особенности
- **ML модель**: XGBoost с оптимизированными гиперпараметрами
- **REST API**: FastAPI с автоматической документацией Swagger
- **Предобработка данных**: автоматическая обработка и преобразование признаков
- **Документация**: полное описание API и методов использования

## 📈 Результаты модели

| Модель | ROC AUC | F1 Score |
|--------|---------|----------|
| XGBoost | 0.5693 ± 0.0146 | 0.3183 ± 0.0086 |
| Random Forest | 0.5717 ± 0.0119 | 0.1511 ± 0.0146 |
| CatBoost | 0.5653 ± 0.0080 | 0.2962 ± 0.0072 |

**Выбранная модель**: XGBoost (лучший баланс метрик)

##  Быстрый старт

### Предварительные требования
- Python 3.10+
- pip (менеджер пакетов Python)

### Установка и запуск

1. **Клонирование репозитория**
```bash
git clone https://github.com/Berserki4/heart-attack-prediction.git
cd heart-attack-prediction
```
2. Установка зависимостей
```bash
pip install -r requirements.txt
```
3. Запуск сервера
```bash
python run.py
```
4. Открытие документации API
Перейдите в браузере: http://localhost:8000/docs

## Использование API

### Предсказание для одного пациента
Endpoint: POST /predict

Пример запроса:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
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
```
Пример ответа:
```json
{
  "risk_score": 0.31105682253837585,
  "risk_class": 0,
  "confidence": 0.6889431476593018
}
```
Проверка здоровья сервера
Endpoint: GET /health
Ответ:
```json
{
  "status": "healthy",
  "message": "API работает нормально"
}
```
Оценка модели:
Для тестирования качества модели используйте скрипт оценки:
```bash
python test_model.py --student submission.csv --correct correct_answers.csv
```
Входные данные:
--student: путь к вашему файлу с предсказаниями (submission.csv)

--correct: путь к файлу с правильными ответами

Выходные данные:
Classification report с метриками: precision, recall, f1-score, accuracy

Подробная статистика по классам
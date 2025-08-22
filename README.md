# 🫀 Heart Attack Risk Prediction API

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3776AB?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)

FastAPI веб-приложение для предсказания риска сердечного приступа на основе медицинских данных пациентов с использованием машинного обучения.

##  О проекте

Проект разработан в рамках технического задания М1. Включает полный цикл: от исследования данных и обучения модели до развертывания production-ready API.

###  Ключевые особенности
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
git clone https://github.com/YOUR_USERNAME/heart-attack-prediction.git
cd heart-attack-prediction
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd

from .models.model_wrapper import model
from .schemas.schemas import PatientData, PredictionResult

app = FastAPI(
    title="Heart Attack Prediction API",
    description="API для предсказания риска сердечного приступа",
    version="1.0.0"
)

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head><title>Heart Attack Prediction API</title></head>
        <body>
            <h1>🚀 Heart Attack Risk Prediction API</h1>
            <p>Документация: <a href="/docs">/docs</a></p>
            <p>Health check: <a href="/health">/health</a></p>
        </body>
    </html>
    """

@app.post("/predict", response_model=PredictionResult)
async def predict(patient: PatientData):
    try:
        patient_data = patient.dict()
        
        # ВСЕ преобразования имен признаков
        patient_data['Heart rate'] = patient_data.pop('Heart_rate')
        patient_data['Exercise Hours Per Week'] = patient_data.pop('Exercise_Hours_Per_Week')
        patient_data['Systolic blood pressure'] = patient_data.pop('Systolic_blood_pressure')
        patient_data['Diastolic blood pressure'] = patient_data.pop('Diastolic_blood_pressure')
        patient_data['Sedentary Hours Per Day'] = patient_data.pop('Sedentary_Hours_Per_Day')
        patient_data['Sleep Hours Per Day'] = patient_data.pop('Sleep_Hours_Per_Day')
        patient_data['Family History'] = patient_data.pop('Family_History')
        patient_data['Previous Heart Problems'] = patient_data.pop('Previous_Heart_Problems')
        
        result = model.predict(patient_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API работает нормально"}
from pydantic import BaseModel, Field
from typing import List, Optional

class PatientData(BaseModel):
    Age: float = Field(..., example=45.0)
    Cholesterol: float = Field(..., example=200.0)
    Heart_rate: float = Field(..., example=75.0)
    Exercise_Hours_Per_Week: float = Field(..., example=3.0)
    Systolic_blood_pressure: float = Field(..., example=120.0)
    Diastolic_blood_pressure: float = Field(..., example=80.0)
    Troponin: float = Field(..., example=0.01)
    Sedentary_Hours_Per_Day: float = Field(..., example=8.0)
    Triglycerides: float = Field(..., example=150.0)
    Sleep_Hours_Per_Day: float = Field(..., example=7.0)
    Diabetes: int = Field(..., example=0)
    Smoking: int = Field(..., example=1)
    Obesity: int = Field(..., example=0)
    Family_History: int = Field(..., example=1)
    Previous_Heart_Problems: int = Field(..., example=0)
    Gender: str = Field(..., example="Male")
    Diet: int = Field(..., example=1)

class PredictionResult(BaseModel):
    risk_score: float = Field(..., example=0.75)
    risk_class: int = Field(..., example=1)
    confidence: float = Field(..., example=0.85)

class BatchPredictionRequest(BaseModel):
    patients: List[PatientData]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]
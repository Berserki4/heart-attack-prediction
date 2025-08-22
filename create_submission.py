import pandas as pd
import numpy as np
import os
from app.models.model import model

# Полный путь к тестовым данным
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(BASE_DIR, 'heart_test.csv')

# Загрузи тестовые данные
test_data = pd.read_csv('heart_test.csv')

# Функция подготовки данных (аналогично API)
def prepare_submission_data(df):
    df = df.copy()
    # Приведи названия колонок к нужному формату
    df = df.rename(columns={
        'Heart rate': 'Heart rate',
        'Exercise Hours Per Week': 'Exercise Hours Per Week',
        'Systolic blood pressure': 'Systolic blood pressure', 
        'Diastolic blood pressure': 'Diastolic blood pressure',
        'Sedentary Hours Per Day': 'Sedentary Hours Per Day',
        'Sleep Hours Per Day': 'Sleep Hours Per Day',
        'Family History': 'Family History',
        'Previous Heart Problems': 'Previous Heart Problems'
    })
    df['Troponin_log'] = np.log(df['Troponin'] + 1e-6)
    return df

# Подготовь данные
prepared_data = prepare_submission_data(test_data)

# Сделай предсказания
predictions = []
for _, row in prepared_data.iterrows():
    try:
        result = model.predict(row.to_dict())
        predictions.append(result['risk_score'])
    except:
        predictions.append(0.5)  # значение по умолчанию

# Сохрани submission
submission = pd.DataFrame({
    'id': test_data['id'],
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)
print("✅ Submission file created!")
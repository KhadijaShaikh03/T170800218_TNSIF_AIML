import os
import joblib
import numpy as np

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "best_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR, "models", "scaler.pkl"
)


# --------------------------------------------------
# LOAD MODEL + SCALER
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="CardioPredict",
    description="AI-powered heart disease risk assessment system",
    version="1.0.0"
)


# --------------------------------------------------
# STATIC FILES + TEMPLATES
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)

templates = Jinja2Templates(
    directory=os.path.join(BASE_DIR, "templates")
)


# --------------------------------------------------
# INPUT DATA MODEL
# --------------------------------------------------

class PatientData(BaseModel):
    age: int
    sex: int
    chest_pain_type: int
    resting_blood_pressure: int
    cholesterol: int
    fasting_blood_sugar: int
    resting_ecg: int
    max_heart_rate: int
    exercise_induced_angina: int
    st_depression: float
    st_slope: int
    num_major_vessels: int
    thalassemia: int


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
    request=request,
    name="index.html"
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "Random Forest"
    }


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict(data: PatientData):

    input_data = np.array([
        [
            data.age,
            data.sex,
            data.chest_pain_type,
            data.resting_blood_pressure,
            data.cholesterol,
            data.fasting_blood_sugar,
            data.resting_ecg,
            data.max_heart_rate,
            data.exercise_induced_angina,
            data.st_depression,
            data.st_slope,
            data.num_major_vessels,
            data.thalassemia
        ]
    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    probabilities = model.predict_proba(input_scaled)[0]

    disease_probability = float(probabilities[1] * 100)
    no_disease_probability = float(probabilities[0] * 100)

    if prediction == 1:
        result = "Higher Risk"
    else:
        result = "Lower Risk"

    return {
        "prediction": int(prediction),
        "result": result,
        "heart_disease_probability": round(
            disease_probability, 2
        ),
        "no_heart_disease_probability": round(
            no_disease_probability, 2
        )
    }
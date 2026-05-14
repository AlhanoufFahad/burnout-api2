from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# تحميل الملفات
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.get("/")
def home():
    return {"message": "API running successfully"}


@app.post("/predict")
def predict(data: dict):

    designation = float(data["designation"])
    resource_allocation = float(data["resource_allocation"])
    mental_fatigue = float(data["mental_fatigue"])

    features = np.array([
        [designation, resource_allocation, mental_fatigue]
    ])

    # scaling
    features_scaled = scaler.transform(features)

    # prediction
    prediction = model.predict(features_scaled)

    return {
        "prediction": float(prediction[0])
    }

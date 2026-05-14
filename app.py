from fastapi import FastAPI
import joblib
import numpy as np
import traceback

app = FastAPI()

try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Model loaded successfully")

except Exception as e:
    print("ERROR:")
    traceback.print_exc()


@app.get("/")
def home():
    return {"message": "API running successfully"}


@app.post("/predict")
def predict(data: dict):

    designation = data["designation"]
    resource_allocation = data["resource_allocation"]
    mental_fatigue = data["mental_fatigue"]

    features = np.array([
        [designation, resource_allocation, mental_fatigue]
    ])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    return {
        "prediction": float(prediction[0])
    }

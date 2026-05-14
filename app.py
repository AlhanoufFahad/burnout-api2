from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# تحميل المودل
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: dict):

    designation = data["designation"]
    resource_allocation = data["resource_allocation"]
    mental_fatigue = data["mental_fatigue"]

    # تجهيز البيانات
    features = np.array([[designation,
                          resource_allocation,
                          mental_fatigue]])

    # Scaling
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)

    return {"prediction": float(prediction[0])}

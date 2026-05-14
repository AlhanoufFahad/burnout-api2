from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "API running"}

@app.post("/predict")
def predict(data: dict):

    x = np.array([[
        float(data["designation"]),
        float(data["resource_allocation"]),
        float(data["mental_fatigue"])
    ]])

    prediction = model.predict(x)

    return {"prediction": float(prediction[0])}

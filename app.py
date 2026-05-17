from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import joblib
import numpy as np
import uvicorn

# =========================
# Load model and scaler
# =========================
model = joblib.load("burnout_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Create FastAPI app
# =========================
app = FastAPI(title="Burnout Detection API")

# =========================
# Enable CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Input schema
# =========================
class UserInput(BaseModel):

    designation: int = Field(..., ge=0, le=5)

    resource_allocation: int = Field(..., ge=0, le=100)

    mental_fatigue_score: int = Field(..., ge=4, le=20)

# =========================
# Output schema
# =========================
class BurnoutResponse(BaseModel):

    burnout_level: str
    recommendation: str
    mental_fatigue_score: int
    raw_prediction: int

# =========================
# Recommendation function
# =========================
def get_recommendation(level):

    recommendations = {

        0: (
            "Low burnout detected. "
            "Maintain work-life balance and healthy work habits."
        ),

        1: (
            "Moderate burnout detected. "
            "Try reducing stress and taking regular breaks."
        ),

        2: (
            "High burnout detected. "
            "We recommend rest and workload reduction."
        )
    }

    return recommendations.get(
        level,
        "Please consult a professional."
    )

# =========================
# Root endpoint
# =========================
@app.get("/")
async def root():

    return {
        "message": "Burnout Detection API is running"
    }

# =========================
# Health endpoint
# =========================
@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }

# =========================
# Prediction endpoint
# =========================
@app.post("/predict", response_model=BurnoutResponse)
async def predict_burnout(input_data: UserInput):

    try:

        # -------------------------
        # Medium burnout rule only
        # -------------------------
        if 6 < input_data.mental_fatigue_score <= 13:

            prediction = 1

        # -------------------------
        # Machine Learning prediction
        # -------------------------
        else:

            features = np.array([[
                input_data.designation,
                input_data.resource_allocation,
                input_data.mental_fatigue_score
            ]])

            # Apply scaler
            features_scaled = scaler.transform(features)

            # ML prediction
            prediction = int(
                model.predict(features_scaled)[0]
            )

        # -------------------------
        # Label mapping
        # -------------------------
        level_map = {
            0: "Low Burnout",
            1: "Medium Burnout",
            2: "High Burnout"
        }

        # -------------------------
        # Return response
        # -------------------------
        return BurnoutResponse(
            burnout_level=level_map[prediction],
            recommendation=get_recommendation(prediction),
            mental_fatigue_score=input_data.mental_fatigue_score,
            raw_prediction=prediction
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =========================
# Run server
# =========================
if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )

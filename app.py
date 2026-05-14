from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Burnout Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
scaler = None

# Try to load model and scaler
try:
    import joblib
    import pickle
    
    # Try different loading methods
    if os.path.exists("burnout_model.pkl"):
        # Method 1: Try pickle
        try:
            with open("burnout_model.pkl", "rb") as f:
                model = pickle.load(f)
            print("✅ Model loaded with pickle")
        except:
            # Method 2: Try joblib
            model = joblib.load("burnout_model.pkl")
            print("✅ Model loaded with joblib")
    
    if os.path.exists("scaler.pkl"):
        try:
            with open("scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            print("✅ Scaler loaded with pickle")
        except:
            scaler = joblib.load("scaler.pkl")
            print("✅ Scaler loaded with joblib")
            
except Exception as e:
    print(f"⚠️ Warning: {e}")
    print("API will run with fallback predictions")

# Fallback prediction function (if model fails)
def fallback_predict(designation, resource_allocation, mental_fatigue):
    """Simple rule-based prediction"""
    score = (designation * 0.3) + (resource_allocation * 0.2) + (mental_fatigue * 0.5)
    if score < 3:
        return 0  # Low
    elif score < 6:
        return 1  # Medium
    else:
        return 2  # High

# Request model
class PredictRequest(BaseModel):
    designation: int
    resource_allocation: int
    mental_fatigue_score: float

# Response model
class PredictResponse(BaseModel):
    burnout_level: str
    recommendation: str

def get_recommendation(level):
    if level == "Low Burnout":
        return "✅ Keep up the good work! Maintain work-life balance and take regular breaks."
    elif level == "Medium Burnout":
        return "⚠️ You may be experiencing moderate burnout. Consider taking short breaks and managing workload."
    else:
        return "🔥 High burnout detected. Please take time off and seek support if needed."

@app.get("/")
async def root():
    return {
        "message": "Burnout Prediction API", 
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        # Prepare input
        input_data = np.array([[
            request.designation,
            request.resource_allocation,
            request.mental_fatigue_score
        ]])
        
        # Make prediction
        if model is not None and scaler is not None:
            # Use ML model with scaling
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
        else:
            # Use fallback prediction
            prediction = fallback_predict(
                request.designation,
                request.resource_allocation,
                request.mental_fatigue_score
            )
        
        # Convert to level
        level_map = {0: "Low Burnout", 1: "Medium Burnout", 2: "High Burnout"}
        burnout_level = level_map.get(prediction, "Medium Burnout")
        
        return PredictResponse(
            burnout_level=burnout_level,
            recommendation=get_recommendation(burnout_level)
        )
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Return fallback response instead of error
        burnout_level = "Medium Burnout"
        return PredictResponse(
            burnout_level=burnout_level,
            recommendation=get_recommendation(burnout_level)
        )

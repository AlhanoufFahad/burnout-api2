from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
import os
import sys
from pathlib import Path

app = FastAPI(
    title="Burnout Prediction API",
    description="API for predicting employee burnout levels",
    version="1.0.0"
)

# Global variables for models
model = None
scaler = None

# Get the absolute path to the current directory
BASE_DIR = Path(__file__).resolve().parent

def load_models():
    """Load models with proper error handling"""
    global model, scaler
    
    try:
        # Try multiple possible paths
        model_paths = [
            BASE_DIR / "burnout_model.pkl",
            Path("/app/burnout_model.pkl"),  # Docker path
            Path("./burnout_model.pkl"),
        ]
        
        scaler_paths = [
            BASE_DIR / "scaler.pkl",
            Path("/app/scaler.pkl"),  # Docker path
            Path("./scaler.pkl"),
        ]
        
        # Find and load model
        model_loaded = False
        for model_path in model_paths:
            if model_path.exists():
                print(f"Loading model from: {model_path}")
                model = joblib.load(model_path)
                model_loaded = True
                break
        
        # Find and load scaler
        scaler_loaded = False
        for scaler_path in scaler_paths:
            if scaler_path.exists():
                print(f"Loading scaler from: {scaler_path}")
                scaler = joblib.load(scaler_path)
                scaler_loaded = True
                break
        
        if not model_loaded:
            print("ERROR: Could not find burnout_model.pkl")
            print(f"Current directory: {BASE_DIR}")
            print(f"Files in directory: {os.listdir(BASE_DIR)}")
        
        if not scaler_loaded:
            print("ERROR: Could not find scaler.pkl")
        
        return model_loaded and scaler_loaded
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

# Load models on startup
@app.on_event("startup")
async def startup_event():
    success = load_models()
    if success:
        print("✅ Models loaded successfully!")
    else:
        print("❌ Failed to load models")

# Request/Response Models
class BurnoutPredictionRequest(BaseModel):
    designation: int = Field(..., ge=0, le=5, description="Designation level (0-5)")
    resource_allocation: int = Field(..., ge=1, le=10, description="Resource allocation score (1-10)")
    mental_fatigue_score: float = Field(..., ge=0, le=10, description="Mental fatigue score (0-10)")

class BurnoutPredictionResponse(BaseModel):
    burnout_level: str
    confidence_scores: Dict[str, float]
    recommendation: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    model_path: Optional[str] = None
    scaler_path: Optional[str] = None

# Helper Functions
def classify_burnout(score: int) -> str:
    if score == 0:
        return "Low Burnout"
    elif score == 1:
        return "Medium Burnout"
    else:
        return "High Burnout"

def get_recommendation(level: str) -> str:
    recommendations = {
        "Low Burnout": "Your stress level is low. Keep maintaining a healthy work-life balance. Continue your current healthy habits and take regular breaks to prevent burnout.",
        "Medium Burnout": "You may be experiencing moderate burnout. Consider taking short breaks throughout the day, managing your workload better, practicing mindfulness, and speaking with your supervisor about workload concerns.",
        "High Burnout": "High burnout detected. We strongly recommend: taking time off work, reducing workload immediately, seeking professional support (counselor or therapist), practicing self-care activities, and discussing your situation with HR or management."
    }
    return recommendations.get(level, "Please consult with a healthcare professional.")

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None and scaler is not None else "unhealthy",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None,
        model_path=str(BASE_DIR / "burnout_model.pkl") if (BASE_DIR / "burnout_model.pkl").exists() else None,
        scaler_path=str(BASE_DIR / "scaler.pkl") if (BASE_DIR / "scaler.pkl").exists() else None
    )

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "model_loaded": True,
        "scaler_loaded": scaler is not None,
        "features": ["Designation", "Resource Allocation", "Mental Fatigue Score"],
        "burnout_levels": ["Low Burnout", "Medium Burnout", "High Burnout"]
    }

@app.post("/predict", response_model=BurnoutPredictionResponse)
async def predict_burnout(request: BurnoutPredictionRequest):
    """Predict burnout level"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        # Prepare input data
        input_data = np.array([[
            request.designation,
            request.resource_allocation,
            request.mental_fatigue_score
        ]])
        
        # Apply scaling
        input_data_scaled = scaler.transform(input_data)
        
        # Get prediction
        prediction = model.predict(input_data_scaled)[0]
        
        # Get confidence scores if available
        confidence_scores = {}
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data_scaled)[0]
            confidence_scores = {
                "Low Burnout": float(probabilities[0]),
                "Medium Burnout": float(probabilities[1]),
                "High Burnout": float(probabilities[2]) if len(probabilities) > 2 else 0.0
            }
        else:
            confidence_scores = {"prediction_confidence": 1.0}
        
        # Get burnout level and recommendation
        burnout_level = classify_burnout(prediction)
        recommendation = get_recommendation(burnout_level)
        
        return BurnoutPredictionResponse(
            burnout_level=burnout_level,
            confidence_scores=confidence_scores,
            recommendation=recommendation
        )
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import os
import sys
from pathlib import Path
import pickle

app = FastAPI(
    title="Burnout Prediction API",
    description="API for predicting employee burnout levels",
    version="1.0.0"
)

# Try to import joblib, fallback to pickle if not available
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("Warning: joblib not available, using pickle")

# Global variables
model = None
scaler = None

# Fallback rule-based prediction (used if model files are missing)
def rule_based_prediction(designation, resource_allocation, mental_fatigue_score):
    """Fallback prediction using business rules"""
    # Simple weighted scoring
    score = (
        designation * 0.3 +
        resource_allocation * 0.2 +
        mental_fatigue_score * 0.5
    )
    
    # Normalize to 0-2 scale
    if score < 3:
        return 0  # Low
    elif score < 6:
        return 1  # Medium
    else:
        return 2  # High

def load_models():
    """Load models with multiple fallback methods"""
    global model, scaler
    
    # Try different methods to load the model
    current_dir = Path(__file__).resolve().parent
    
    # Method 1: Using joblib
    if HAS_JOBLIB:
        try:
            model_path = current_dir / "burnout_model.pkl"
            if model_path.exists():
                print(f"Loading model from {model_path}")
                model = joblib.load(model_path)
                print("Model loaded successfully with joblib")
        except Exception as e:
            print(f"Failed to load with joblib: {e}")
    
    # Method 2: Using pickle (if joblib failed)
    if model is None:
        try:
            model_path = current_dir / "burnout_model.pkl"
            if model_path.exists():
                print(f"Loading model with pickle from {model_path}")
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print("Model loaded successfully with pickle")
        except Exception as e:
            print(f"Failed to load with pickle: {e}")
    
    # Load scaler similarly
    if HAS_JOBLIB:
        try:
            scaler_path = current_dir / "scaler.pkl"
            if scaler_path.exists():
                print(f"Loading scaler from {scaler_path}")
                scaler = joblib.load(scaler_path)
                print("Scaler loaded successfully with joblib")
        except Exception as e:
            print(f"Failed to load scaler with joblib: {e}")
    
    if scaler is None:
        try:
            scaler_path = current_dir / "scaler.pkl"
            if scaler_path.exists():
                print(f"Loading scaler with pickle from {scaler_path}")
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print("Scaler loaded successfully with pickle")
        except Exception as e:
            print(f"Failed to load scaler with pickle: {e}")
    
    # If model still None, use rule-based fallback
    if model is None:
        print("WARNING: Using rule-based fallback prediction")
        model = "rule_based"
    
    return model is not None or model == "rule_based"

def predict_with_model(features):
    """Make prediction using loaded model or fallback"""
    global model, scaler
    
    if model == "rule_based":
        # Use rule-based prediction
        designation, resource_allocation, mental_fatigue = features[0]
        prediction = rule_based_prediction(designation, resource_allocation, mental_fatigue)
        
        # Generate confidence scores for rule-based
        confidence_scores = {
            "Low Burnout": 0.7 if prediction == 0 else 0.2,
            "Medium Burnout": 0.7 if prediction == 1 else 0.2,
            "High Burnout": 0.7 if prediction == 2 else 0.2
        }
        return prediction, confidence_scores
    
    # Use actual ML model
    try:
        # Apply scaling if available
        if scaler is not None:
            features = scaler.transform(features)
        
        prediction = model.predict(features)[0]
        
        # Get confidence scores
        confidence_scores = {}
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence_scores = {
                "Low Burnout": float(probabilities[0]),
                "Medium Burnout": float(probabilities[1]),
                "High Burnout": float(probabilities[2]) if len(probabilities) > 2 else 0.0
            }
        else:
            confidence_scores = {"confidence": 0.8}
        
        return prediction, confidence_scores
    except Exception as e:
        print(f"Model prediction error: {e}")
        # Fallback to rule-based
        designation, resource_allocation, mental_fatigue = features[0]
        prediction = rule_based_prediction(designation, resource_allocation, mental_fatigue)
        confidence_scores = {"confidence": 0.5, "note": "Using fallback prediction"}
        return prediction, confidence_scores

# Load models on startup
@app.on_event("startup")
async def startup_event():
    success = load_models()
    if success:
        print("✅ Models loaded successfully!")
    else:
        print("⚠️ Running in fallback mode with rule-based predictions")

# Request/Response Models
class BurnoutPredictionRequest(BaseModel):
    designation: int = Field(..., ge=0, le=5)
    resource_allocation: int = Field(..., ge=1, le=10)
    mental_fatigue_score: float = Field(..., ge=0, le=10)

class BurnoutPredictionResponse(BaseModel):
    burnout_level: str
    confidence_scores: Dict[str, float]
    recommendation: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    scaler_loaded: bool

# Helper Functions
def classify_burnout(score) -> str:
    if score == 0:
        return "Low Burnout"
    elif score == 1:
        return "Medium Burnout"
    else:
        return "High Burnout"

def get_recommendation(level: str) -> str:
    recommendations = {
        "Low Burnout": "✅ Your stress level is low. Keep maintaining a healthy work-life balance.",
        "Medium Burnout": "⚠️ You may be experiencing moderate burnout. Consider taking short breaks and managing workload.",
        "High Burnout": "🚨 High burnout detected. We recommend rest, reducing workload, and seeking support if needed."
    }
    return recommendations.get(level, "Please consult with a healthcare professional.")

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Burnout Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_type=str(type(model).__name__) if model != "rule_based" else "rule_based_fallback",
        scaler_loaded=scaler is not None
    )

@app.get("/debug")
async def debug_info():
    """Debug endpoint to diagnose issues"""
    current_dir = Path(__file__).resolve().parent
    files = list(current_dir.glob("*"))
    
    model_files = [f.name for f in files if f.suffix == '.pkl']
    
    return {
        "current_directory": str(current_dir),
        "available_model_files": model_files,
        "joblib_available": HAS_JOBLIB,
        "model_loaded": model is not None,
        "model_type": str(type(model).__name__) if model != "rule_based" else "rule_based",
        "scaler_loaded": scaler is not None,
        "python_version": sys.version
    }

@app.post("/predict", response_model=BurnoutPredictionResponse)
async def predict_burnout(request: BurnoutPredictionRequest):
    """Predict burnout level"""
    try:
        # Prepare input data
        input_data = np.array([[
            request.designation,
            request.resource_allocation,
            request.mental_fatigue_score
        ]])
        
        # Get prediction
        prediction, confidence_scores = predict_with_model(input_data)
        
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

# Additional endpoints for convenience
@app.post("/predict/batch")
async def batch_predict(requests: List[BurnoutPredictionRequest]):
    """Batch prediction"""
    results = []
    for req in requests:
        input_data = np.array([[
            req.designation,
            req.resource_allocation,
            req.mental_fatigue_score
        ]])
        prediction, confidence_scores = predict_with_model(input_data)
        burnout_level = classify_burnout(prediction)
        
        results.append({
            "input": req.dict(),
            "burnout_level": burnout_level,
            "recommendation": get_recommendation(burnout_level)
        })
    
    return {"predictions": results, "total": len(results)}
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Burnout Prediction AI API")

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

# Force numpy compatibility
import numpy as np
np.__version__ = '1.24.3'

# Load model with compatibility fixes
def load_models():
    global model, scaler
    
    try:
        import joblib
        import pickle
        
        # Set environment for compatibility
        os.environ['XGBOOST_LOADER'] = 'pickle'
        
        # Load model
        if os.path.exists("burnout_model.pkl"):
            try:
                # Try joblib first
                model = joblib.load("burnout_model.pkl")
                print("✅ Model loaded with joblib")
            except Exception as e:
                print(f"Joblib failed: {e}")
                # Try pickle
                with open("burnout_model.pkl", "rb") as f:
                    model = pickle.load(f)
                print("✅ Model loaded with pickle")
        
        # Load scaler
        if os.path.exists("scaler.pkl"):
            try:
                scaler = joblib.load("scaler.pkl")
                print("✅ Scaler loaded with joblib")
            except:
                with open("scaler.pkl", "rb") as f:
                    scaler = pickle.load(f)
                print("✅ Scaler loaded with pickle")
        
        # Test the model
        if model is not None:
            test_input = np.array([[2, 5, 4.5]]).astype(np.float32)
            test_pred = model.predict(test_input)
            print(f"✅ Model test successful: {test_pred[0]}")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()

# Load models on startup
@app.on_event("startup")
async def startup_event():
    load_models()
    print(f"Files in directory: {os.listdir('.')}")
    print(f"Model loaded: {model is not None}")
    print(f"Scaler loaded: {scaler is not None}")

# Request model
class PredictRequest(BaseModel):
    designation: int = Field(ge=0, le=5, description="Designation level (0-5)")
    resource_allocation: int = Field(ge=1, le=10, description="Resource allocation (1-10)")
    mental_fatigue_score: float = Field(ge=0, le=10, description="Mental fatigue score (0-10)")

# Response model
class PredictResponse(BaseModel):
    burnout_level: str
    recommendation: str
    confidence: float = None

def get_recommendation(level):
    recommendations = {
        "Low Burnout": "✅ You're doing great! Keep maintaining work-life balance and take regular breaks.",
        "Medium Burnout": "⚠️ You may be experiencing moderate burnout. Consider taking short breaks, delegating tasks, and practicing mindfulness.",
        "High Burnout": "🔥 High burnout detected. We strongly recommend taking time off, reducing workload, and seeking professional support."
    }
    return recommendations.get(level, "Please consult a professional.")

@app.get("/")
async def root():
    return {
        "message": "Burnout Prediction AI API", 
        "status": "running",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_type": "XGBoost" if model is not None else "None"
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="AI Model not loaded. Please try again later."
        )
    
    try:
        # Prepare input as float32 for XGBoost
        input_data = np.array([[
            float(request.designation),
            float(request.resource_allocation),
            float(request.mental_fatigue_score)
        ]], dtype=np.float32)
        
        # Apply scaling if scaler is available
        if scaler is not None:
            input_scaled = scaler.transform(input_data)
        else:
            input_scaled = input_data
        
        # Get prediction
        prediction = model.predict(input_scaled)[0]
        
        # Get confidence if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(input_scaled)[0]
            confidence = float(max(probs) * 100)
        
        # Convert to level
        level_map = {0: "Low Burnout", 1: "Medium Burnout", 2: "High Burnout"}
        burnout_level = level_map.get(prediction, "Medium Burnout")
        
        return PredictResponse(
            burnout_level=burnout_level,
            recommendation=get_recommendation(burnout_level),
            confidence=confidence
        )
        
    except Exception as e:
        print(f"Prediction error details: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/debug-predict")
async def debug_predict(request: PredictRequest):
    """Debug endpoint to see what's happening"""
    return {
        "input": request.dict(),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "message": "Check server logs for details"
    }

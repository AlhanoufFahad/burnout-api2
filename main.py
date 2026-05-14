from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
from pathlib import Path

app = FastAPI(title="Burnout Detection API")

# CORS for frontend
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

# Load models on startup
@app.on_event("startup")
async def startup_event():
    global model, scaler
    
    # Get the directory where this file is located
    current_dir = Path(__file__).parent
    
    # Paths to model files
    model_path = current_dir / "burnout_model.pkl"
    scaler_path = current_dir / "scaler.pkl"
    
    print(f"Looking for model at: {model_path}")
    print(f"Looking for scaler at: {scaler_path}")
    
    # Check if files exist
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"❌ Model file not found at {model_path}")
    
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
    else:
        print(f"❌ Scaler file not found at {scaler_path}")
    
    # List all files in directory for debugging
    print("\n📁 Files in current directory:")
    for file in current_dir.iterdir():
        print(f"   - {file.name}")

class UserInput(BaseModel):
    designation: int = Field(ge=0, le=5)
    resource_allocation: int = Field(ge=1, le=10)
    mental_fatigue_score: float = Field(ge=0, le=10)

class BurnoutResponse(BaseModel):
    burnout_level: str
    recommendation: str
    mental_fatigue_score: float
    raw_prediction: int

def get_recommendation(level):
    recommendations = {
        0: "✅ Low Burnout: Keep up the good work! Maintain work-life balance and take regular breaks.",
        1: "⚠️ Moderate Burnout: Consider taking short breaks, delegating tasks, and practicing mindfulness.",
        2: "🔥 High Burnout: Take time off, reduce workload, and seek professional support."
    }
    return recommendations.get(level, "Please consult a professional.")

@app.get("/")
async def root():
    return {
        "message": "Burnout Detection API is running!",
        "status": "active",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model and scaler else "degraded",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=BurnoutResponse)
async def predict_burnout(input_data: UserInput):
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Models not loaded. Model: {model is not None}, Scaler: {scaler is not None}"
        )
    
    try:
        # Prepare input
        raw_input = np.array([[
            input_data.designation,
            input_data.resource_allocation,
            input_data.mental_fatigue_score
        ]])
        
        # Scale the input
        scaled_input = scaler.transform(raw_input)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        
        level_map = {0: "Low Burnout", 1: "Medium Burnout", 2: "High Burnout"}
        
        return BurnoutResponse(
            burnout_level=level_map[prediction],
            recommendation=get_recommendation(prediction),
            mental_fatigue_score=input_data.mental_fatigue_score,
            raw_prediction=int(prediction)
        )
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

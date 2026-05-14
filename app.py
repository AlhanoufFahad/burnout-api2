from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Burnout Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = None
scaler = None

try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# Request model
class PredictRequest(BaseModel):
    designation: int
    resource_allocation: int
    mental_fatigue_score: float

# Response model
class PredictResponse(BaseModel):
    burnout_level: str
    recommendation: str

# Helper function
def get_recommendation(level):
    if level == "Low Burnout":
        return "✅ Keep up the good work! Maintain work-life balance."
    elif level == "Medium Burnout":
        return "⚠️ Take short breaks and manage your workload better."
    else:
        return "🔥 Please take time off and seek support if needed."

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Burnout Prediction API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input
        input_data = np.array([[
            request.designation,
            request.resource_allocation,
            request.mental_fatigue_score
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Convert to level
        level_map = {0: "Low Burnout", 1: "Medium Burnout", 2: "High Burnout"}
        burnout_level = level_map[prediction]
        
        return PredictResponse(
            burnout_level=burnout_level,
            recommendation=get_recommendation(burnout_level)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# تحميل النموذج والسكالر
model_path = "burnout_model.pkl"
scaler_path = "scaler.pkl"

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and Scaler loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Scaler type: {type(scaler).__name__}")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    scaler = None

app = FastAPI(title="Burnout Detection API")

# إضافة CORS - يسمح للواجهة باستدعاء الـ API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    designation: int = Field(ge=0, le=5, description="Designation level (0-5)")
    resource_allocation: int = Field(ge=1, le=10, description="Resource allocation (1-10)")
    mental_fatigue_score: float = Field(ge=0, le=10, description="Mental fatigue score (0-10)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "designation": 2,
                "resource_allocation": 5,
                "mental_fatigue_score": 4.5
            }
        }

class BurnoutResponse(BaseModel):
    burnout_level: str
    recommendation: str
    mental_fatigue_score: float
    raw_prediction: int
    confidence_scores: dict = None

def get_recommendation(level):
    recommendations = {
        0: "✅ Low Burnout: Keep up the good work! Maintain work-life balance and take regular breaks. Continue your healthy habits.",
        1: "⚠️ Moderate Burnout: You may be experiencing moderate burnout. Consider taking short breaks, delegating tasks, practicing mindfulness, and setting clear boundaries.",
        2: "🔥 High Burnout: High burnout detected. We strongly recommend taking time off, reducing workload immediately, seeking professional support, and prioritizing self-care."
    }
    return recommendations.get(level, "Please consult a healthcare professional for personalized advice.")

@app.get("/")
async def root():
    return {
        "message": "Burnout Detection API is running!", 
        "status": "active",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "features": ["Designation", "Resource Allocation", "Mental Fatigue Score"]
    }

@app.post("/predict", response_model=BurnoutResponse)
async def predict_burnout(input_data: UserInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or Scaler not loaded. Please check server.")
    
    try:
        # تحضير البيانات الخام
        raw_features = np.array([[
            input_data.designation,
            input_data.resource_allocation,
            input_data.mental_fatigue_score
        ]])
        
        # تطبيق Scaler (مهم جداً!)
        scaled_features = scaler.transform(raw_features)
        
        # التنبؤ
        prediction = model.predict(scaled_features)[0]
        
        # الحصول على نسب الثقة إذا كان النموذج يدعمها
        confidence_scores = {}
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(scaled_features)[0]
            confidence_scores = {
                "Low Burnout": float(probabilities[0]),
                "Medium Burnout": float(probabilities[1]),
                "High Burnout": float(probabilities[2]) if len(probabilities) > 2 else 0.0
            }
        
        # تعيين مستوى الاحتراق
        level_map = {0: "Low Burnout", 1: "Medium Burnout", 2: "High Burnout"}
        
        return BurnoutResponse(
            burnout_level=level_map[prediction],
            recommendation=get_recommendation(prediction),
            mental_fatigue_score=input_data.mental_fatigue_score,
            raw_prediction=int(prediction),
            confidence_scores=confidence_scores
        )
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None and scaler is not None else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_type": type(model).__name__ if model else None,
        "scaler_type": type(scaler).__name__ if scaler else None
    }

@app.get("/model-info")
async def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "model_loaded": True,
        "scaler_loaded": scaler is not None,
        "expected_features": ["Designation", "Resource Allocation", "Mental Fatigue Score"],
        "output_classes": ["Low Burnout (0)", "Medium Burnout (1)", "High Burnout (2)"]
    }

# لتشغيل الخادم - مهم لـ Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

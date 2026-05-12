from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # مهم للربط مع الواجهة
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# تحميل النموذج
model_path = "burnout_model.pkl"
model = joblib.load(model_path)

app = FastAPI(title="Burnout Detection API")

# 🔓 إضافة CORS - يسمح للواجهة باستدعاء الـ API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في الإنتاج، استبدلي بـ URL الواجهة الحقيقي
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    designation: int = Field(ge=1, le=5)
    resource_allocation: int = Field(ge=1, le=10)
    mental_fatigue_score: int = Field(ge=4, le=20)

class BurnoutResponse(BaseModel):
    burnout_level: str
    recommendation: str
    mental_fatigue_score: int
    raw_prediction: int

def get_recommendation(level):
    recommendations = {
        0: "Keep up the good work! Maintain work-life balance and take regular breaks. 🌿",
        1: "Moderate burnout risk. Try to delegate tasks, practice mindfulness, and set boundaries. ⚠️",
        2: "High burnout risk. Consider taking time off, reducing workload, and seeking support. 🔥"
    }
    return recommendations.get(level, "Please consult a professional.")

@app.get("/")
async def root():
    return {"message": "Burnout Detection API is running!", "status": "active"}

@app.post("/predict", response_model=BurnoutResponse)
async def predict_burnout(input_data: UserInput):
    try:
        features = np.array([[
            input_data.designation,
            input_data.resource_allocation,
            input_data.mental_fatigue_score
        ]])
        
        prediction = model.predict(features)[0]
        
        level_map = {0: "Low Burnout", 1: "Medium Burnout", 2: "High Burnout"}
        
        return BurnoutResponse(
            burnout_level=level_map[prediction],
            recommendation=get_recommendation(prediction),
            mental_fatigue_score=input_data.mental_fatigue_score,
            raw_prediction=int(prediction)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

# لتشغيل الخادم - مهم لـ Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import joblib
import numpy as np
import pandas as pd
from enum import Enum

# Initialize FastAPI app
app = FastAPI(
    title="Burnout Prediction API",
    description="API for predicting employee burnout levels based on work-related factors",
    version="1.0.0"
)

# Load models and scaler
try:
    model = joblib.load("burnout_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    scaler = None

# Enums for validation
class LikertScale(str, Enum):
    STRONGLY_AGREE = "Strongly Agree"
    AGREE = "Agree"
    NEUTRAL = "Neutral"
    DISAGREE = "Disagree"
    STRONGLY_DISAGREE = "Strongly Disagree"

class BurnoutLevel(str, Enum):
    LOW = "Low Burnout"
    MEDIUM = "Medium Burnout"
    HIGH = "High Burnout"

# Request Models
class BurnoutPredictionRequest(BaseModel):
    designation: int = Field(..., ge=0, le=5, description="Designation level (0-5)")
    resource_allocation: int = Field(..., ge=1, le=10, description="Resource allocation score (1-10)")
    mental_fatigue_score: float = Field(..., ge=0, le=10, description="Mental fatigue score (0-10)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "designation": 2,
                "resource_allocation": 5,
                "mental_fatigue_score": 4.5
            }
        }

class MentalFatigueAnswers(BaseModel):
    answers: List[LikertScale] = Field(..., min_length=4, max_length=4, description="4 Likert scale answers")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answers": ["Strongly Agree", "Agree", "Neutral", "Agree"]
            }
        }

class ChatbotRequest(BaseModel):
    designation: int = Field(..., ge=0, le=5)
    resource_allocation: int = Field(..., ge=1, le=10)
    answers: List[LikertScale] = Field(..., min_length=4, max_length=4)

# Response Models
class BurnoutPredictionResponse(BaseModel):
    burnout_level: BurnoutLevel
    confidence_scores: Dict[str, float]
    recommendation: str

class MentalFatigueScoreResponse(BaseModel):
    mental_fatigue_score: float
    burnout_level: BurnoutLevel
    recommendation: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool

# Helper Functions
def classify_burnout(score: float) -> BurnoutLevel:
    """Classify burnout level based on predicted class"""
    if score == 0:
        return BurnoutLevel.LOW
    elif score == 1:
        return BurnoutLevel.MEDIUM
    else:
        return BurnoutLevel.HIGH

def get_recommendation(level: BurnoutLevel) -> str:
    """Generate recommendation based on burnout level"""
    recommendations = {
        BurnoutLevel.LOW: "Your stress level is low. Keep maintaining a healthy work-life balance. Continue your current healthy habits and take regular breaks to prevent burnout.",
        BurnoutLevel.MEDIUM: "You may be experiencing moderate burnout. Consider taking short breaks throughout the day, managing your workload better, practicing mindfulness, and speaking with your supervisor about workload concerns.",
        BurnoutLevel.HIGH: "High burnout detected. We strongly recommend: taking time off work, reducing workload immediately, seeking professional support (counselor or therapist), practicing self-care activities, and discussing your situation with HR or management."
    }
    return recommendations.get(level, "Please consult with a healthcare professional.")

def calculate_mental_fatigue_score(answers: List[str]) -> float:
    """Calculate mental fatigue score from Likert scale answers"""
    score_map = {
        "Strongly Agree": 5,
        "Agree": 4,
        "Neutral": 3,
        "Disagree": 2,
        "Strongly Disagree": 1
    }
    
    total_score = sum(score_map[answer] for answer in answers)
    # Convert to 0-10 scale (max score is 20, min is 4)
    mental_fatigue_score = ((total_score - 4) / 16) * 10
    return round(mental_fatigue_score, 2)

def get_prediction_with_confidence(features: np.ndarray) -> tuple:
    """Get prediction and confidence scores from model"""
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        prediction = np.argmax(probabilities)
        confidence_scores = {
            "Low Burnout": float(probabilities[0]),
            "Medium Burnout": float(probabilities[1]),
            "High Burnout": float(probabilities[2]) if len(probabilities) > 2 else 0.0
        }
        return prediction, confidence_scores
    else:
        prediction = model.predict(features)[0]
        return prediction, {}

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        scaler_loaded=scaler is not None
    )

@app.post("/predict", response_model=BurnoutPredictionResponse)
async def predict_burnout(request: BurnoutPredictionRequest):
    """
    Predict burnout level based on designation, resource allocation, and mental fatigue score
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input data
        input_data = np.array([[
            request.designation,
            request.resource_allocation,
            request.mental_fatigue_score
        ]])
        
        # Apply scaling
        input_data_scaled = scaler.transform(input_data)
        
        # Get prediction and confidence
        prediction, confidence_scores = get_prediction_with_confidence(input_data_scaled)
        
        # Get burnout level and recommendation
        burnout_level = classify_burnout(prediction)
        recommendation = get_recommendation(burnout_level)
        
        return BurnoutPredictionResponse(
            burnout_level=burnout_level,
            confidence_scores=confidence_scores,
            recommendation=recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/mental-fatigue-score", response_model=MentalFatigueScoreResponse)
async def calculate_mental_fatigue(answers: MentalFatigueAnswers):
    """
    Calculate mental fatigue score from Likert scale answers and predict burnout
    """
    try:
        # Calculate mental fatigue score
        mental_fatigue_score = calculate_mental_fatigue_score(answers.answers)
        
        # Use default values for other features (since this endpoint only uses mental fatigue)
        # In a real scenario, you'd collect designation and resource allocation too
        request = BurnoutPredictionRequest(
            designation=2,  # Default value
            resource_allocation=5,  # Default value
            mental_fatigue_score=mental_fatigue_score
        )
        
        # Get burnout prediction
        result = await predict_burnout(request)
        
        return MentalFatigueScoreResponse(
            mental_fatigue_score=mental_fatigue_score,
            burnout_level=result.burnout_level,
            recommendation=result.recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")

@app.post("/chatbot-predict", response_model=MentalFatigueScoreResponse)
async def chatbot_predict(request: ChatbotRequest):
    """
    Complete chatbot prediction endpoint that takes all inputs and returns burnout level
    """
    try:
        # Calculate mental fatigue score from answers
        mental_fatigue_score = calculate_mental_fatigue_score(request.answers)
        
        # Prepare prediction request
        prediction_request = BurnoutPredictionRequest(
            designation=request.designation,
            resource_allocation=request.resource_allocation,
            mental_fatigue_score=mental_fatigue_score
        )
        
        # Get burnout prediction
        result = await predict_burnout(prediction_request)
        
        return MentalFatigueScoreResponse(
            mental_fatigue_score=mental_fatigue_score,
            burnout_level=result.burnout_level,
            recommendation=result.recommendation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(requests: List[BurnoutPredictionRequest]):
    """
    Batch prediction for multiple employees
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for req in requests:
            input_data = np.array([[
                req.designation,
                req.resource_allocation,
                req.mental_fatigue_score
            ]])
            
            input_data_scaled = scaler.transform(input_data)
            prediction, confidence_scores = get_prediction_with_confidence(input_data_scaled)
            burnout_level = classify_burnout(prediction)
            
            results.append({
                "input": req.dict(),
                "burnout_level": burnout_level.value,
                "confidence_scores": confidence_scores,
                "recommendation": get_recommendation(burnout_level)
            })
        
        return {"predictions": results, "total": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "model_loaded": True,
        "scaler_loaded": scaler is not None,
        "features": ["Designation", "Resource Allocation", "Mental Fatigue Score"],
        "burnout_levels": ["Low Burnout", "Medium Burnout", "High Burnout"]
    }

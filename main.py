from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

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

# Load model with compatibility handling
@app.on_event("startup")
async def startup_event():
    global model
    
    current_dir = Path(__file__).parent
    model_path = current_dir / "burnout_model.pkl"
    
    print(f"Looking for model at: {model_path}")
    
    if model_path.exists():
        try:
            # Load model with compatibility for older XGBoost versions
            model = joblib.load(model_path)
            
            # Fix for XGBoost 2.0+ compatibility
            if hasattr(model, 'use_label_encoder'):
                model.use_label_encoder = False
            
            print("✅ Model loaded successfully!")
            print(f"Model type: {type(model).__name__}")
            
            # Test prediction with sample data
            test_input = np.array([[2, 5, 4.5]])
            test_pred = model.predict(test_input)
            print(f"Test prediction successful: {test_pred[0]}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            
            # Try loading with different method
            try:
                import xgboost as xgb
                # Try loading as native XGBoost model
                if str(model_path).endswith('.pkl'):
                    model = joblib.load(model_path)
                    # Force compatibility
                    if hasattr(model, 'use_label_encoder'):
                        model.use_label_encoder = False
                    print("✅ Model loaded with compatibility mode!")
            except Exception as e2:
                print(f"❌ Alternative loading also failed: {e2}")
                model = None
    else:
        print(f"❌ Model file not found at {model_path}")
    
    # List files for debugging
    print("\n📁 Files in current directory:")
    for file in current_dir.iterdir():
        if not file.name.startswith('.'):
            print(f"   - {file.name}")

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
        "model_type": type(model).__name__ if model else None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None
    }

@app.post("/predict", response_model=BurnoutResponse)
async def predict_burnout(input_data: UserInput):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Prepare input
        raw_input = np.array([[
            input_data.designation,
            input_data.resource_allocation,
            input_data.mental_fatigue_score
        ]]).astype(np.float32)
        
        # Make prediction
        prediction = model.predict(raw_input)[0]
        
        # Get confidence scores if available
        confidence_scores = {}
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(raw_input)[0]
                confidence_scores = {
                    "Low Burnout": float(probabilities[0]),
                    "Medium Burnout": float(probabilities[1]),
                    "High Burnout": float(probabilities[2]) if len(probabilities) > 2 else 0.0
                }
            except:
                confidence_scores = {"note": "Probability scores not available"}
        
        level_map = {0: "Low Burnout", 1: "Medium Burnout", 2: "High Burnout"}
        
        return BurnoutResponse(
            burnout_level=level_map.get(prediction, "Unknown"),
            recommendation=get_recommendation(prediction),
            mental_fatigue_score=input_data.mental_fatigue_score,
            raw_prediction=int(prediction),
            confidence_scores=confidence_scores if confidence_scores else None
        )
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/debug")
async def debug():
    """Debug endpoint to check model status"""
    current_dir = Path(__file__).parent
    files = [f.name for f in current_dir.iterdir() if not f.name.startswith('.')]
    
    model_info = None
    if model:
        try:
            model_info = {
                "type": type(model).__name__,
                "has_predict_proba": hasattr(model, 'predict_proba')
            }
        except:
            model_info = {"type": "Unknown"}
    
    return {
        "current_directory": str(current_dir),
        "files": files,
        "model_loaded": model is not None,
        "model_info": model_info,
        "model_file_exists": (current_dir / "burnout_model.pkl").exists()
    }

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

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

# Load model with compatibility handling
@app.on_event("startup")
async def startup_event():
    global model
    
    current_dir = Path(__file__).parent
    model_path = current_dir / "burnout_model.pkl"
    
    print(f"Looking for model at: {model_path}")
    
    if model_path.exists():
        try:
            # Load model with compatibility for older XGBoost versions
            model = joblib.load(model_path)
            
            # Fix for XGBoost 2.0+ compatibility
            if hasattr(model, 'use_label_encoder'):
                model.use_label_encoder = False
            
            print("✅ Model loaded successfully!")
            print(f"Model type: {type(model).__name__}")
            
            # Test prediction with sample data
            test_input = np.array([[2, 5, 4.5]])
            test_pred = model.predict(test_input)
            print(f"Test prediction successful: {test_pred[0]}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            
            # Try loading with different method
            try:
                import xgboost as xgb
                # Try loading as native XGBoost model
                if str(model_path).endswith('.pkl'):
                    model = joblib.load(model_path)
                    # Force compatibility
                    if hasattr(model, 'use_label_encoder'):
                        model.use_label_encoder = False
                    print("✅ Model loaded with compatibility mode!")
            except Exception as e2:
                print(f"❌ Alternative loading also failed: {e2}")
                model = None
    else:
        print(f"❌ Model file not found at {model_path}")
    
    # List files for debugging
    print("\n📁 Files in current directory:")
    for file in current_dir.iterdir():
        if not file.name.startswith('.'):
            print(f"   - {file.name}")

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
        "model_type": type(model).__name__ if model else None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model else "degraded",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None
    }

@app.post("/predict", response_model=BurnoutResponse)
async def predict_burnout(input_data: UserInput):
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Prepare input
        raw_input = np.array([[
            input_data.designation,
            input_data.resource_allocation,
            input_data.mental_fatigue_score
        ]]).astype(np.float32)
        
        # Make prediction
        prediction = model.predict(raw_input)[0]
        
        # Get confidence scores if available
        confidence_scores = {}
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(raw_input)[0]
                confidence_scores = {
                    "Low Burnout": float(probabilities[0]),
                    "Medium Burnout": float(probabilities[1]),
                    "High Burnout": float(probabilities[2]) if len(probabilities) > 2 else 0.0
                }
            except:
                confidence_scores = {"note": "Probability scores not available"}
        
        level_map = {0: "Low Burnout", 1: "Medium Burnout", 2: "High Burnout"}
        
        return BurnoutResponse(
            burnout_level=level_map.get(prediction, "Unknown"),
            recommendation=get_recommendation(prediction),
            mental_fatigue_score=input_data.mental_fatigue_score,
            raw_prediction=int(prediction),
            confidence_scores=confidence_scores if confidence_scores else None
        )
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/debug")
async def debug():
    """Debug endpoint to check model status"""
    current_dir = Path(__file__).parent
    files = [f.name for f in current_dir.iterdir() if not f.name.startswith('.')]
    
    model_info = None
    if model:
        try:
            model_info = {
                "type": type(model).__name__,
                "has_predict_proba": hasattr(model, 'predict_proba')
            }
        except:
            model_info = {"type": "Unknown"}
    
    return {
        "current_directory": str(current_dir),
        "files": files,
        "model_loaded": model is not None,
        "model_info": model_info,
        "model_file_exists": (current_dir / "burnout_model.pkl").exists()
    }

# For local testing
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

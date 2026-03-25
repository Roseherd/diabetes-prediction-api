# -*- coding: utf-8 -*-
"""
FastAPI Application for Diabetes Prediction
Serves the XGBoost model as a REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import joblib
import numpy as np
import os
import uvicorn
import logging

# ============================================================================
# LOGGING SETUP
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# FASTAPI APP INITIALIZATION
# ============================================================================
app = FastAPI(
    title="Diabetes Prediction API",
    description="Machine Learning API for diabetes risk prediction using XGBoost",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================================
MODEL_DIR = 'models'

logger.info("Loading model artifacts...")

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    feature_names = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))
    metadata = joblib.load(os.path.join(MODEL_DIR, 'model_metadata.pkl'))
    logger.info("All model artifacts loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

# ============================================================================
# PYDANTIC MODELS (REQUEST/RESPONSE SCHEMAS)
# ============================================================================

class PredictionRequest(BaseModel):
    """Schema for prediction request"""
    gender: str = Field(..., description="Gender (e.g., 'Male', 'Female', 'Other')")
    age: float = Field(..., ge=0, le=120, description="Age in years (0-120)")
    hypertension: int = Field(..., ge=0, le=1, description="Hypertension status (0 or 1)")
    heart_disease: int = Field(..., ge=0, le=1, description="Heart disease status (0 or 1)")
    smoking_history: str = Field(..., description="Smoking history category")
    bmi: float = Field(..., ge=10, le=100, description="Body Mass Index (10-100)")
    HbA1c_level: float = Field(..., ge=4, le=10, description="HbA1c level (4-10)")
    blood_glucose_level: int = Field(..., ge=80, le=300, description="Blood glucose level (80-300)")

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "age": 45.0,
                "hypertension": 0,
                "heart_disease": 0,
                "smoking_history": "never",
                "bmi": 25.5,
                "HbA1c_level": 5.8,
                "blood_glucose_level": 120
            }
        }

class BatchPredictionRequest(BaseModel):
    """Schema for batch predictions"""
    predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")

class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    input_data: Dict[str, Any]
    prediction: int = Field(..., ge=0, le=1, description="Prediction (0: No diabetes, 1: Has diabetes)")
    probability: float = Field(..., ge=0, le=1, description="Confidence probability")
    risk_level: str = Field(..., description="Risk level (Low, Medium, High)")

class ModelInfo(BaseModel):
    """Schema for model information response"""
    model_type: str
    n_features: int
    feature_names: List[str]
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def preprocess_input(input_data: PredictionRequest) -> np.ndarray:
    """
    Preprocess input data for prediction
    
    Args:
        input_data: PredictionRequest object
        
    Returns:
        Preprocessed numpy array ready for model prediction
    """
    # Create a dictionary with the input data
    data_dict = input_data.dict()
    
    # Encode categorical features
    try:
        if 'gender' in label_encoders:
            data_dict['gender'] = label_encoders['gender'].transform([data_dict['gender']])[0]
        
        if 'smoking_history' in label_encoders:
            data_dict['smoking_history'] = label_encoders['smoking_history'].transform([data_dict['smoking_history']])[0]
    except ValueError as e:
        raise ValueError(f"Invalid categorical value: {e}")
    
    # Create feature array in the correct order
    X = np.array([data_dict[feature] for feature in feature_names]).reshape(1, -1)
    
    # Scale the features
    X_scaled = scaler.transform(X)
    
    return X_scaled

def get_risk_level(probability: float) -> str:
    """Determine risk level based on prediction probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Serve the custom prediction UI"""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    else:
        return {
            "message": "Diabetes Prediction API is running",
            "status": "OK",
            "endpoints": {
                "ui": "/ (visit in browser)",
                "health": "/health",
                "model_info": "/model-info",
                "predict": "/predict",
                "batch_predict": "/batch-predict",
                "docs": "/docs",
                "redoc": "/redoc"
            }
        }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "OK",
        "message": "API is healthy"
    }

@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get model information and performance metrics"""
    return ModelInfo(
        model_type=metadata['model_type'],
        n_features=metadata['n_features'],
        feature_names=metadata['feature_names'],
        accuracy=metadata['accuracy'],
        precision=metadata['precision'],
        recall=metadata['recall'],
        f1_score=metadata['f1_score'],
        roc_auc=metadata['roc_auc']
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Make a single diabetes prediction
    
    Args:
        request: Patient data for prediction
        
    Returns:
        Prediction result with confidence score and risk level
    """
    try:
        # Preprocess input
        X_scaled = preprocess_input(request)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0, 1]
        risk_level = get_risk_level(probability)
        
        return PredictionResponse(
            input_data=request.dict(),
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction")

@app.post("/batch-predict", tags=["Predictions"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple patients
    
    Args:
        request: List of patient data for predictions
        
    Returns:
        List of prediction results
    """
    try:
        results = []
        
        for patient_data in request.predictions:
            # Preprocess input
            X_scaled = preprocess_input(patient_data)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0, 1]
            risk_level = get_risk_level(probability)
            
            results.append({
                "input_data": patient_data.dict(),
                "prediction": int(prediction),
                "probability": float(probability),
                "risk_level": risk_level
            })
        
        return {
            "total_predictions": len(results),
            "predictions": results
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input data: {str(e)}")
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during batch prediction")

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return {
        "error": True,
        "status_code": exc.status_code,
        "detail": exc.detail
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

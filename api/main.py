"""
FastAPI Application for Citizen Grievance Analysis
Production-ready API for sentiment analysis and priority classification
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

import uvicorn


# CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "trained_models"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Priority mapping
PRIORITY_MAPPING = {
    "Critical": {"score": 5, "label": "URGENT", "description": "Requires immediate action"},
    "Negative": {"score": 4, "label": "HIGH", "description": "Should be addressed soon"},
    "Neutral": {"score": 3, "label": "MEDIUM", "description": "Standard processing"},
    "Positive": {"score": 2, "label": "LOW", "description": "Routine handling"}
}


# REQUEST/RESPONSE MODELS
class ComplaintRequest(BaseModel):
    """Request model for complaint analysis"""
    complaint_text: str = Field(..., min_length=10, max_length=5000, description="Complaint description")
    
    class Config:
        json_schema_extra = {
            "example": {
                "complaint_text": "The streetlights have been out for three days and the road feels unsafe."
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    sentiment: str = Field(..., description="Predicted sentiment classification")
    priority_score: int = Field(..., description="Priority score from 1-5")
    priority_label: str = Field(..., description="Priority label (LOW, MEDIUM, HIGH, URGENT)")
    description: str = Field(..., description="Priority description")
    confidence: float = Field(..., description="Confidence level (0-1)")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of complaint texts")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions")
    total_processed: int = Field(..., description="Total complaints processed")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_type: str
    vectorizer_type: str
    accuracy: float
    f1_score: float
    sentiment_classes: List[str]
    class_distribution: Dict[str, int]
    training_date: str


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    models_loaded: bool
    models_available: List[str]


# MODEL LOADING
def load_models():
    """Load trained models and metadata"""
    try:
        if not SENTIMENT_MODEL_PATH.exists():
            raise FileNotFoundError(f"Sentiment model not found at {SENTIMENT_MODEL_PATH}")
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}")
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata not found at {METADATA_PATH}")
        
        model = joblib.load(SENTIMENT_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
        
        logger.info("Models loaded successfully")
        return model, vectorizer, metadata
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


# GLOBAL STATE
class AppState:
    """Application state management"""
    model = None
    vectorizer = None
    metadata = None
    models_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    # Startup
    try:
        AppState.model, AppState.vectorizer, AppState.metadata = load_models()
        AppState.models_loaded = True
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Failed to load models during startup: {str(e)}")
        AppState.models_loaded = False
    
    yield
    
    # Shutdown
    logger.info("Application shutdown")


# FASTAPI APPLICATION
app = FastAPI(
    title="Citizen Grievance Analysis API",
    description="Production-ready API for sentiment analysis and priority classification",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ENDPOINTS
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy" if AppState.models_loaded else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        models_loaded=AppState.models_loaded,
        models_available=[
            "sentiment_model.joblib",
            "tfidf_vectorizer.joblib"
        ] if AppState.models_loaded else []
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model information and performance metrics"""
    if not AppState.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. API is in degraded state."
        )
    
    return ModelInfoResponse(
        model_type=AppState.metadata.get("model_type", "Unknown"),
        vectorizer_type=AppState.metadata.get("vectorizer_type", "Unknown"),
        accuracy=AppState.metadata.get("test_accuracy", 0.0),
        f1_score=AppState.metadata.get("test_macro_f1", 0.0),
        sentiment_classes=AppState.metadata.get("sentiment_classes", []),
        class_distribution=AppState.metadata.get("class_distribution", {}),
        training_date=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_sentiment(request: ComplaintRequest):
    """
    Predict sentiment and priority for a single complaint
    
    Returns sentiment classification (Critical/Negative/Neutral/Positive) and priority score
    """
    if not AppState.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not available. Service temporarily unavailable."
        )
    
    try:
        # Preprocess and vectorize
        features = AppState.vectorizer.transform([request.complaint_text])
        
        # Predict
        prediction = AppState.model.predict(features)[0]
        decision_scores = AppState.model.decision_function(features)[0]
        
        # Calculate confidence (approximate)
        max_score = abs(decision_scores).max()
        confidence = min(max_score / 10.0, 1.0)  # Normalize to 0-1
        
        # Get priority information
        priority_info = PRIORITY_MAPPING.get(prediction, {})
        
        logger.info(f"Prediction: {prediction} (confidence: {confidence:.2f})")
        
        return PredictionResponse(
            sentiment=prediction,
            priority_score=priority_info.get("score", 3),
            priority_label=priority_info.get("label", "MEDIUM"),
            description=priority_info.get("description", "Unknown"),
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict sentiment for multiple complaints in batch
    
    Processes up to 100 complaints at once
    """
    if not AppState.models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not available. Service temporarily unavailable."
        )
    
    try:
        import time
        start_time = time.time()
        
        # Vectorize all texts
        features = AppState.vectorizer.transform(request.texts)
        predictions = AppState.model.predict(features)
        decision_scores = AppState.model.decision_function(features)
        
        # Process predictions
        results = []
        for text, prediction, scores in zip(request.texts, predictions, decision_scores):
            confidence = min(abs(scores).max() / 10.0, 1.0)
            priority_info = PRIORITY_MAPPING.get(prediction, {})
            
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "sentiment": prediction,
                "priority_score": priority_info.get("score", 3),
                "priority_label": priority_info.get("label", "MEDIUM"),
                "confidence": confidence
            })
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        logger.info(f"Batch prediction: {len(request.texts)} texts in {processing_time:.2f}ms")
        
        return BatchPredictionResponse(
            predictions=results,
            total_processed=len(request.texts),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    """API information endpoint"""
    return {
        "name": "Citizen Grievance Analysis API",
        "version": "1.0.0",
        "status": "operational" if AppState.models_loaded else "degraded",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict (POST)",
            "batch_predict": "/batch_predict (POST)",
            "documentation": "/docs"
        }
    }


# STARTUP EVENTS
@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    logger.info("API Server Starting")
    logger.info(f"Models Directory: {MODELS_DIR}")
    logger.info(f"Models Loaded: {AppState.models_loaded}")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

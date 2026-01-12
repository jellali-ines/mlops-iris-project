"""
FastAPI for the model - API for Iris Classification
"""
import os
import pickle
import time
from typing import List

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram


# Pydantic models
class IrisFeatures(BaseModel):
    """Iris flower features"""
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal Length (cm)")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal Width (cm)")
    petal_length: float = Field(..., ge=0, le=10, description="Petal Length (cm)")
    petal_width: float = Field(..., ge=0, le=10, description="Petal Width (cm)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionResponse(BaseModel):
    """Prediction Response"""
    prediction: int
    class_name: str
    probability: List[float]
    model_version: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: str


# Initialize FastAPI
app = FastAPI(
    title="Iris Classification API",
    description="API for Iris flower classification using ML",
    version="1.0.0"
)

# Enable Prometheus metrics
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Global variables
model = None
scaler = None
model_version = os.getenv('MODEL_VERSION', 'optuna_best')  # This is the most accurate model (Optuna output)
model_path = os.getenv('MODEL_PATH', 'models/model_optuna_best.pkl')
scaler_path = model_path.replace('model_', 'scaler_')

CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica']

# Custom Prometheus Metrics
PREDICTION_COUNTER = Counter(
    "iris_predictions_total", 
    "Total number of predictions", 
    ["model_version", "class_name"]
)
INFERENCE_LATENCY = Histogram(
    "iris_inference_latency_seconds", 
    "Inference latency in seconds",
    ["model_version"]
)
ERROR_COUNTER = Counter(
    "iris_prediction_errors_total", 
    "Total number of prediction errors",
    ["model_version", "error_type"]
)


def load_model_artifacts():
    """Load model and scaler artifacts"""
    global model, scaler
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Fix for scikit-learn 1.5+ compatibility with older models
        if hasattr(model, 'coef_') and not hasattr(model, 'multi_class'):
            model.multi_class = 'auto'
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


# Load model at startup
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model_artifacts()
    if success:
        print(f"✅ Loaded model {model_version} from {model_path}")
    else:
        print(f"⚠️ Warning: Model not loaded from {model_path}")


@app.get("/", tags=["Root"])
async def root():
    """Home page"""
    return {
        "message": "Iris Classification API",
        "version": model_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version
    )


@app.get("/info", tags=["Info"])
async def model_info():
    """Model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": model_version,
        "model_type": type(model).__name__,
        "model_path": model_path,
        "classes": CLASS_NAMES,
        "n_features": 4,
        "monitoring": "/metrics",
        "retrain": "/retrain"
    }


@app.get("/retrain", tags=["Training"])
@app.post("/retrain", tags=["Training"])
async def trigger_retrain(request: Request):
    """
    Request model retraining
    Triggers a simulated retraining process
    """
    # In a real scenario, this would trigger a GitHub Action or a ZenML pipeline
    # Here we simulate it by returning the command to run
    print(f"🔄 Retrain request received from {request.client.host}")
    return {
        "status": "Accepted",
        "message": "Retraining pipeline triggered (Simulated)",
        "command": "python src/models/train.py --config configs/best_optuna.yaml",
        "timestamp": time.time()
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: IrisFeatures):
    """
    Predict Iris flower species
    
    - **sepal_length**: Sepal length in cm
    - **sepal_width**: Sepal width in cm
    - **petal_length**: Petal length in cm
    - **petal_width**: Petal width in cm
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Start timing
    start_time = time.time()
    
    try:
        # Prepare input
        input_data = [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]]
        
        # Scale features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = int(model.predict(input_scaled)[0])
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_scaled)[0].tolist()
        else:
            # For SVM without probability
            probabilities = [0.0] * 3
            probabilities[prediction] = 1.0
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Custom Metrics Logging
        PREDICTION_COUNTER.labels(
            model_version=model_version, 
            class_name=CLASS_NAMES[prediction]
        ).inc()
        
        INFERENCE_LATENCY.labels(
            model_version=model_version
        ).observe(inference_time / 1000)
        
        return PredictionResponse(
            prediction=prediction,
            class_name=CLASS_NAMES[prediction],
            probability=probabilities,
            model_version=model_version,
            inference_time_ms=round(inference_time, 2)
        )
    
    except Exception as e:
        ERROR_COUNTER.labels(
            model_version=model_version,
            error_type=type(e).__name__
        ).inc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(features_list: List[IrisFeatures]):
    """Multiple predictions in batch"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    for features in features_list:
        pred = await predict(features)
        predictions.append(pred)
    
    return {"predictions": predictions, "count": len(predictions)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

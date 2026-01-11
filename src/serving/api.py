"""
FastAPI للنموذج - API for Iris Classification
"""
import os
import pickle
import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Pydantic models
class IrisFeatures(BaseModel):
    """مميزات زهرة Iris"""
    sepal_length: float = Field(..., ge=0, le=10, description="طول الكأس (cm)")
    sepal_width: float = Field(..., ge=0, le=10, description="عرض الكأس (cm)")
    petal_length: float = Field(..., ge=0, le=10, description="طول البتلة (cm)")
    petal_width: float = Field(..., ge=0, le=10, description="عرض البتلة (cm)")
    
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionResponse(BaseModel):
    """استجابة التنبؤ"""
    prediction: int
    class_name: str
    probability: List[float]
    model_version: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """استجابة الصحة"""
    status: str
    model_loaded: bool
    model_version: str


# Initialize FastAPI
app = FastAPI(
    title="Iris Classification API",
    description="API لتصنيف زهور Iris باستخدام ML",
    version="1.0.0"
)

# Global variables
model = None
scaler = None
model_version = os.getenv('MODEL_VERSION', 'v2')  # v2 لأنه الأفضل (100%)
model_path = os.getenv('MODEL_PATH', 'models/model_v2.pkl')
scaler_path = model_path.replace('model_', 'scaler_')

CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica']


def load_model_artifacts():
    """تحميل النموذج والـ scaler"""
    global model, scaler
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return True
    except Exception as e:
        print(f"❌ خطأ في تحميل النموذج: {e}")
        return False


# Load model at startup
@app.on_event("startup")
async def startup_event():
    """تحميل النموذج عند البدء"""
    success = load_model_artifacts()
    if success:
        print(f"✅ تم تحميل النموذج {model_version} من {model_path}")
    else:
        print(f"⚠️ تحذير: لم يتم تحميل النموذج من {model_path}")


@app.get("/", tags=["Root"])
async def root():
    """الصفحة الرئيسية"""
    return {
        "message": "Iris Classification API",
        "version": model_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """فحص صحة الـ API"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_version
    )


@app.get("/info", tags=["Info"])
async def model_info():
    """معلومات النموذج"""
    if model is None:
        raise HTTPException(status_code=503, detail="النموذج غير محمل")
    
    return {
        "model_version": model_version,
        "model_type": type(model).__name__,
        "model_path": model_path,
        "classes": CLASS_NAMES,
        "n_features": 4
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: IrisFeatures):
    """
    التنبؤ بنوع زهرة Iris
    
    - **sepal_length**: طول الكأس بالسنتيمتر
    - **sepal_width**: عرض الكأس بالسنتيمتر
    - **petal_length**: طول البتلة بالسنتيمتر
    - **petal_width**: عرض البتلة بالسنتيمتر
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="النموذج غير محمل")
    
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
        
        return PredictionResponse(
            prediction=prediction,
            class_name=CLASS_NAMES[prediction],
            probability=probabilities,
            model_version=model_version,
            inference_time_ms=round(inference_time, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في التنبؤ: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(features_list: List[IrisFeatures]):
    """تنبؤات متعددة دفعة واحدة"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="النموذج غير محمل")
    
    predictions = []
    for features in features_list:
        pred = await predict(features)
        predictions.append(pred)
    
    return {"predictions": predictions, "count": len(predictions)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
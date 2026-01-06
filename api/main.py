from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
import os
import time
from typing import Dict, Any

from .schemas import PredictionRequest, BatchPredictionResponse, PredictionResponse, InterventionRecommendation

# Global model store
models: Dict[str, Any] = {}
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    print("Loading models...")
    try:
        models["churn_model"] = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
        models["causal_model"] = joblib.load(os.path.join(MODEL_DIR, "causal_forest.pkl"))
        models["label_encoders"] = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl")) if os.path.exists(os.path.join(MODEL_DIR, "label_encoders.pkl")) else None
        # scaler might be needed if not part of pipeline
        models["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")) if os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl")) else None
        models["feature_columns"] = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # In production, we might want to crash if models fail to load
    yield
    # Clean up on shutdown
    models.clear()

app = FastAPI(title="Churn Prediction & Intervention API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_features(customers_data, feature_columns, scaler=None):
    df = pd.DataFrame([c.features for c in customers_data])
    
    # Ensure all expected columns exist, fill missing with 0 or appropriate value
    # logic depends on training. Assuming alignment for now.
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0 # Default handling
            
    # Reorder to match training
    df = df[feature_columns]
    
    # Scale if scaler exists
    if scaler:
        # Check if scaler expects specific columns or whole DF
        # Assuming whole DF for simplicity based on standard pipelines
        try:
             df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
             return df_scaled
        except Exception:
             # Fallback if scaler fails or mismatch
             return df
    return df

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": list(models.keys())}

@app.get("/")
def root():
    return {
        "message": "Welcome to the Churn Prediction & Intervention API",
        "docs_url": "/docs",
        "health_check": "/health",
        "predict_endpoint": "/predict"
    }

@app.post("/predict", response_model=BatchPredictionResponse)
async def predict_churn(request: PredictionRequest):
    start_time = time.time()
    
    if "churn_model" not in models:
        raise HTTPException(status_code=503, detail="Models not loaded")
        
    feature_cols = models.get("feature_columns", [])
    
    # Preprocess
    try:
        X = preprocess_features(request.customers, feature_cols, models.get("scaler"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

    # Make predictions
    try:
        # LightGBM usually returns prob via predict_proba or predict depending on wrapper
        # Using .predict() for pure lgb, .predict_proba() for sklearn wrapper
        model = models["churn_model"]
        if hasattr(model, "predict_proba"):
            churn_probs = model.predict_proba(X)[:, 1]
        else:
            churn_probs = model.predict(X)
            
        # Decision Logic (Simplified)
        # Using 0.5 threshold for 'At Risk'
        # For 'At Risk', calculate CATE
        
        causal_model = models.get("causal_model")
        predictions = []
        
        # CATE (Conditional Average Treatment Effect)
        # We need to run this for all or just at risk? 
        # Usually we want to know intervention effect for everyone to decide who to target.
        # But CausalForest might be heavy. Let's run for all for now.
        
        cates = [0] * len(X)
        if causal_model:
            # EconML models typically take X and T? Or just X for CATE(X)?
            # const_marginal_effect gives CATE(X)
            try:
                cates = causal_model.const_marginal_effect(X)
                # If returns array of arrays, flatten
                if hasattr(cates, "flatten"):
                    cates = cates.flatten()
            except Exception as e:
                print(f"CATE prediction failed: {e}")
                
        for i, customer in enumerate(request.customers):
            prob = float(churn_probs[i])
            is_at_risk = prob > 0.5
            cate = float(cates[i]) if i < len(cates) else 0.0
            
            # Simple policy: If at risk AND cate is negative (reduces churn), recommend
            recommendation = None
            if is_at_risk and cate < -0.05: # Threshold for significant effect
                 recommendation = InterventionRecommendation(
                     intervention_type="standard_retention_offer", # Placeholder
                     predicted_lift=abs(cate),
                     description="Standard retention discount recommended based on high churn risk and positive expected response."
                 )
            
            predictions.append(PredictionResponse(
                customer_id=customer.customer_id,
                churn_probability=prob,
                is_at_risk=is_at_risk,
                recommended_intervention=recommendation,
                meta={"cate": cate}
            ))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return BatchPredictionResponse(
        predictions=predictions,
        processing_time_ms=(time.time() - start_time) * 1000
    )

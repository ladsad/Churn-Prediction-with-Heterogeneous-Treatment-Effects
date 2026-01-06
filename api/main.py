"""
FastAPI application for real-time intervention recommendations.

Provides endpoints for:
- Single customer recommendation
- Batch prediction
- Health check
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pickle
from pathlib import Path

app = FastAPI(
    title="Churn HTE Intervention API",
    description="Real-time intervention recommendations using causal ML",
    version="1.0.0"
)

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "models"


# Pydantic schemas
class CustomerFeatures(BaseModel):
    """Input features for a single customer."""
    customer_id: str = Field(..., description="Unique customer identifier")
    tenure: float = Field(..., ge=0, description="Months with company")
    monthly_charges: float = Field(..., ge=0, description="Monthly subscription cost")
    total_charges: float = Field(..., ge=0, description="Total charges to date")
    contract_type: str = Field(..., description="Month-to-month, One year, Two year")
    internet_service: str = Field(..., description="DSL, Fiber optic, No")
    senior_citizen: int = Field(0, ge=0, le=1, description="1 if senior citizen")
    tech_support: str = Field("No", description="Yes/No")
    online_security: str = Field("No", description="Yes/No")
    payment_method: str = Field("", description="Payment method")
    paperless_billing: str = Field("No", description="Yes/No")

    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST-001",
                "tenure": 12,
                "monthly_charges": 70.50,
                "total_charges": 846.00,
                "contract_type": "Month-to-month",
                "internet_service": "Fiber optic",
                "senior_citizen": 0,
                "tech_support": "No",
                "online_security": "No",
                "payment_method": "Electronic check",
                "paperless_billing": "Yes"
            }
        }


class InterventionOption(BaseModel):
    """Single intervention option with predicted effect."""
    intervention_type: str
    predicted_effect: float = Field(..., description="Negative = reduces churn")
    cost: float
    confidence: float = Field(0.85, ge=0, le=1)


class RecommendationResponse(BaseModel):
    """Full recommendation response."""
    customer_id: str
    churn_probability: float
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH")
    recommended_intervention: str
    predicted_churn_reduction: float
    confidence_interval: List[float]
    alternatives: List[InterventionOption]
    should_intervene: bool


class BatchRequest(BaseModel):
    """Batch prediction request."""
    customers: List[CustomerFeatures]


class BatchResponse(BaseModel):
    """Batch prediction response."""
    recommendations: List[RecommendationResponse]
    summary: Dict[str, Any]


# Model loading 
class ModelManager:
    """Manages model loading and inference."""
    
    def __init__(self):
        self.churn_model = None
        self.causal_forest = None
        self.scaler = None
        self.feature_columns = None
        self.loaded = False
    
    def load_models(self):
        """Load all models from disk."""
        try:
            with open(MODELS_DIR / 'churn_model.pkl', 'rb') as f:
                self.churn_model = pickle.load(f)
            
            with open(MODELS_DIR / 'causal_forest.pkl', 'rb') as f:
                self.causal_forest = pickle.load(f)
            
            with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open(MODELS_DIR / 'feature_columns.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            self.loaded = True
            return True
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            return False
    
    def prepare_features(self, customer: CustomerFeatures) -> np.ndarray:
        """Convert customer data to feature array."""
        # Create feature dict
        features = {
            'tenure': customer.tenure,
            'tenure_years': customer.tenure / 12,
            'MonthlyCharges': customer.monthly_charges,
            'TotalCharges': customer.total_charges,
            'SeniorCitizen': customer.senior_citizen,
            'high_value': 1 if customer.monthly_charges > 70 else 0,
            'long_tenure': 1 if customer.tenure > 24 else 0,
            'new_customer': 1 if customer.tenure < 6 else 0,
        }
        
        # Encode categoricals (simplified)
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        internet_map = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
        yes_no_map = {'No': 0, 'Yes': 1}
        
        features['Contract_encoded'] = contract_map.get(customer.contract_type, 0)
        features['InternetService_encoded'] = internet_map.get(customer.internet_service, 0)
        features['OnlineSecurity_encoded'] = yes_no_map.get(customer.online_security, 0)
        features['TechSupport_encoded'] = yes_no_map.get(customer.tech_support, 0)
        features['PaperlessBilling_encoded'] = yes_no_map.get(customer.paperless_billing, 0)
        features['PaymentMethod_encoded'] = 0  # Simplified
        
        # Build feature array in correct order
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        
        # Scale
        X_scaled = self.scaler.transform(X)
        return X_scaled


# Global model manager
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    success = model_manager.load_models()
    if not success:
        print("WARNING: Models not loaded. Run notebooks first to generate models.")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": model_manager.loaded,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=RecommendationResponse)
async def predict_single(customer: CustomerFeatures):
    """Get intervention recommendation for a single customer."""
    
    if not model_manager.loaded:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded. Run notebooks to generate models first."
        )
    
    # Prepare features
    X = model_manager.prepare_features(customer)
    
    # Predict churn probability
    churn_prob = float(model_manager.churn_model.predict(X)[0])
    
    # Determine risk level
    if churn_prob < 0.3:
        risk_level = "LOW"
    elif churn_prob < 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"
    
    # If low risk, no intervention needed
    if churn_prob < 0.3:
        return RecommendationResponse(
            customer_id=customer.customer_id,
            churn_probability=churn_prob,
            risk_level=risk_level,
            recommended_intervention="none",
            predicted_churn_reduction=0.0,
            confidence_interval=[0.0, 0.0],
            alternatives=[],
            should_intervene=False
        )
    
    # Estimate treatment effect
    base_effect = float(model_manager.causal_forest.effect(X)[0])
    
    # Generate recommendations for each intervention type
    intervention_costs = {
        'discount_99': 99,
        'discount_199': 199,
        'feature_upsell': 0,
        'support_call': 50
    }
    
    # Estimate effects for different intervention types
    alternatives = []
    for itype, cost in intervention_costs.items():
        # Scale effect based on intervention type
        effect_multiplier = {
            'discount_99': 0.6,
            'discount_199': 1.0,
            'feature_upsell': 0.4,
            'support_call': 0.5
        }
        effect = base_effect * effect_multiplier[itype]
        
        alternatives.append(InterventionOption(
            intervention_type=itype,
            predicted_effect=effect,
            cost=cost,
            confidence=0.85
        ))
    
    # Sort by effect (most negative = best)
    alternatives.sort(key=lambda x: x.predicted_effect)
    
    # Best intervention
    best = alternatives[0]
    
    return RecommendationResponse(
        customer_id=customer.customer_id,
        churn_probability=churn_prob,
        risk_level=risk_level,
        recommended_intervention=best.intervention_type,
        predicted_churn_reduction=-best.predicted_effect,
        confidence_interval=[-best.predicted_effect - 0.05, -best.predicted_effect + 0.05],
        alternatives=alternatives,
        should_intervene=True
    )


@app.post("/batch-predict", response_model=BatchResponse)
async def predict_batch(request: BatchRequest):
    """Get intervention recommendations for multiple customers."""
    
    if not model_manager.loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run notebooks to generate models first."
        )
    
    recommendations = []
    for customer in request.customers:
        rec = await predict_single(customer)
        recommendations.append(rec)
    
    # Summary statistics
    high_risk_count = sum(1 for r in recommendations if r.risk_level == "HIGH")
    should_intervene_count = sum(1 for r in recommendations if r.should_intervene)
    avg_churn_prob = np.mean([r.churn_probability for r in recommendations])
    
    summary = {
        "total_customers": len(recommendations),
        "high_risk_count": high_risk_count,
        "should_intervene_count": should_intervene_count,
        "average_churn_probability": float(avg_churn_prob)
    }
    
    return BatchResponse(
        recommendations=recommendations,
        summary=summary
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

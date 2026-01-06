from pydantic import BaseModel, Field, conlist
from typing import List, Optional, Dict, Any

class CustomerFeatures(BaseModel):
    """
    Input schema for customer features.
    
    Expected features should match the training data columns.
    For this specific project, we expect features like:
    - tenure (months)
    - MonthlyCharges
    - TotalCharges
    - ... and categorical features encoded or passed as is if the pipeline handles them.
    
    Note: Ideally, this should dynamically match the feature list. 
    For now, we accept a dictionary or specific critical fields to be flexible w/ the pipeline.
    """
    customer_id: str = Field(..., description="Unique identifier for the customer")
    features: Dict[str, Any] = Field(..., description="Dictionary of feature names and values")

class PredictionRequest(BaseModel):
    customers: List[CustomerFeatures]

class InterventionRecommendation(BaseModel):
    intervention_type: str
    predicted_lift: float
    description: str

class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    is_at_risk: bool
    recommended_intervention: Optional[InterventionRecommendation] = None
    meta: Optional[Dict[str, Any]] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    processing_time_ms: float

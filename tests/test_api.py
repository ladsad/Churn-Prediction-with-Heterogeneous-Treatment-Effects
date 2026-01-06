from fastapi.testclient import TestClient
from api.main import app, models
import pytest
from unittest.mock import MagicMock
import numpy as np

client = TestClient(app)

# Mock data
mock_features = {
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
    "Contract_Month-to-month": 1,
    "PaymentMethod_Electronic check": 1
    # Add other encoded columns as expected by the model logic
}

@pytest.fixture
def mock_models(monkeypatch):
    """
    Mock the models dictionary in api.main to avoid loading real files.
    """
    
    # Mock Churn Model
    churn_model = MagicMock()
    # Setup predict_proba to return high risk for one, low for another
    # Input shape will be (N, features)
    churn_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2]]) 
    # if using predict: churn_model.predict.return_value = np.array([0.9, 0.2])
    
    # Mock Causal Model
    causal_model = MagicMock()
    # const_marginal_effect returns shape (N, 1) or (N,)
    causal_model.const_marginal_effect.return_value = np.array([-0.08, -0.01])
    
    # Mock Scaler and Cols
    feature_columns = ["tenure", "MonthlyCharges", "TotalCharges", "Contract_Month-to-month", "PaymentMethod_Electronic check"]
    
    # Patch the global models dict
    # We need to manually set it because TestClient with lifespan might override or run concurrently
    # But since we're testing the endpoints which access `models`, patching the dict object itself works if it's mutable.
    
    models["churn_model"] = churn_model
    models["causal_model"] = causal_model
    models["feature_columns"] = feature_columns
    # models["scaler"] = ... optional
    
    yield models
    
    models.clear()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_churn_endpoint(mock_models):
    # Prepare request
    payload = {
        "customers": [
            {"customer_id": "cust_1", "features": mock_features}, # Should be high risk (0.9)
            {"customer_id": "cust_2", "features": mock_features}  # Should be low risk (0.2)
        ]
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["predictions"]) == 2
    
    # Check Cust 1 (High Risk)
    p1 = data["predictions"][0]
    assert p1["customer_id"] == "cust_1"
    assert p1["churn_probability"] == 0.9
    assert p1["is_at_risk"] is True
    # CATE was -0.08, which is < -0.05, so valid intervention
    assert p1["recommended_intervention"] is not None
    assert p1["recommended_intervention"]["intervention_type"] == "standard_retention_offer"
    
    # Check Cust 2 (Low Risk)
    p2 = data["predictions"][1]
    assert p2["customer_id"] == "cust_2"
    assert p2["churn_probability"] == 0.2
    assert p2["is_at_risk"] is False
    assert p2["recommended_intervention"] is None

def test_predict_missing_models():
    # Clear models to simulate failure
    models.clear()
    payload = {"customers": [{"customer_id": "1", "features": {}}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 503

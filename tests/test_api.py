"""
Tests for FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_returns_200(self):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_structure(self):
        """Test health response has expected fields."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "models_loaded" in data
        assert "version" in data


class TestPredictEndpoint:
    """Tests for predict endpoint."""
    
    def test_predict_valid_request(self):
        """Test predict with valid request."""
        payload = {
            "customer_id": "TEST-001",
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
        
        response = client.post("/predict", json=payload)
        
        # May return 503 if models not loaded
        assert response.status_code in [200, 503]
    
    def test_predict_invalid_request(self):
        """Test predict with missing required fields."""
        payload = {
            "customer_id": "TEST-002"
            # Missing required fields
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error


class TestBatchPredictEndpoint:
    """Tests for batch predict endpoint."""
    
    def test_batch_predict_valid_request(self):
        """Test batch predict with valid request."""
        payload = {
            "customers": [
                {
                    "customer_id": "TEST-001",
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
            ]
        }
        
        response = client.post("/batch-predict", json=payload)
        
        # May return 503 if models not loaded
        assert response.status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# System Architecture

## Overview

The production system follows a request-response pattern for real-time intervention recommendations.

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│    Customer Trigger Event               │
│    (login, billing, usage, etc.)        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  1. Fetch Customer Data                 │
│     - Tenure, MRR, usage metrics        │
│     - Recent behavior (30-day window)   │
│     - Contract and service details      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  2. Churn Risk Scoring                  │
│     - LightGBM churn model              │
│     - If churn_prob > 30%, continue     │
│     - Otherwise: no intervention        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  3. Treatment Effect Estimation         │
│     - Causal Forest model               │
│     - Predict effect for each           │
│       intervention type                 │
│     - Include confidence intervals      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  4. Optimal Intervention Selection      │
│     - Pick highest-effect intervention  │
│     - Apply cost constraints            │
│     - Business rules (eligibility)      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  5. Execute & Log                       │
│     - Return recommendation to caller   │
│     - Log prediction for monitoring     │
│     - Trigger intervention workflow     │
└─────────────────────────────────────────┘
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single customer recommendation |
| `/batch-predict` | POST | Bulk customer scoring |
| `/health` | GET | Health check |

## Request/Response Example

**Request**:
```json
{
  "customer_id": "CUST-001",
  "tenure": 12,
  "monthly_charges": 70.50,
  "contract_type": "Month-to-month",
  "internet_service": "Fiber optic"
}
```

**Response**:
```json
{
  "customer_id": "CUST-001",
  "churn_probability": 0.45,
  "risk_level": "MEDIUM",
  "recommended_intervention": "discount_199",
  "predicted_churn_reduction": 0.15,
  "should_intervene": true,
  "alternatives": [...]
}
```

## Deployment

### Local Development
```bash
uvicorn api.main:app --reload
```

### Production
- Container: Docker with uvicorn + gunicorn
- Scaling: Horizontal with load balancer
- Latency target: < 100ms p99

## Model Serving

Models are loaded at startup from `models/` directory:
- `churn_model.pkl` - LightGBM churn predictor
- `causal_forest.pkl` - CausalForestDML
- `scaler.pkl` - Feature StandardScaler
- `feature_columns.pkl` - Feature list

## Monitoring

### Metrics to Track
1. Prediction latency (p50, p95, p99)
2. Churn prediction calibration
3. Treatment effect drift
4. Feature distribution drift

### Alerting
- AUC drops below 0.75
- Feature drift detected (KS test p < 0.01)
- Latency exceeds 200ms

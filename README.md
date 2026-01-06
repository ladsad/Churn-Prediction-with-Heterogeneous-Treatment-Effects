# Churn Prediction with Heterogeneous Treatment Effects

A causal ML system that predicts customer churn **and** determines the best personalized intervention for each customer using causal inference.

## Project Overview

Most churn prediction systems just predict "who churns" and offer everyone the same intervention. This project goes further:

1. **Churn Prediction**: LightGBM model to identify at-risk customers
2. **Causal Effect Estimation**: Doubly robust estimation to isolate true intervention effects
3. **Heterogeneous Treatment Effects**: Causal forests to estimate per-customer treatment effects
4. **A/B Test Validation**: Statistical validation of causal estimates
5. **Production API**: FastAPI service for real-time recommendations

## Key Innovation

Rather than treating all at-risk customers the same, we estimate **Conditional Average Treatment Effects (CATE)** - understanding that different customer segments respond to different interventions.

## Project Structure

```
churn-hte/
├── data/                   # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for each phase
├── src/                    # Python source modules
├── models/                 # Saved model artifacts
├── api/                    # FastAPI inference service
├── docs/                   # Documentation
└── tests/                  # Unit and integration tests
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd churn-hte

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running Notebooks

```bash
jupyter notebook notebooks/
```

Run notebooks in order: `01_eda.ipynb` → `02_churn_prediction.ipynb` → ...

### Running the API

```bash
uvicorn api.main:app --reload
```

## Methodology

- **Propensity Score Estimation**: Model treatment assignment to handle selection bias
- **Doubly Robust Estimation**: Combine propensity scores + outcome models for robust ATE
- **Causal Forest (econml)**: Estimate heterogeneous treatment effects across customer segments

## Expected Results

| Metric | Value |
|--------|-------|
| Churn Prediction AUC-ROC | ~0.82 |
| Average Treatment Effect | -12% (intervention reduces churn) |
| Relative Churn Reduction | ~24% |

## License

MIT

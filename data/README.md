# Data Directory

## Structure

- `raw/` - Original, unmodified datasets
- `processed/` - Cleaned and feature-engineered datasets

## Data Sources

### Telco Customer Churn Dataset
- **Source**: IBM Sample Data Sets / Kaggle
- **Size**: 7,043 customers, 21 features
- **Churn Rate**: ~26.5%

### Synthetic Interventions
Generated via `src/simulation.py` to create realistic confounding for causal inference.

## Data Dictionary

| Column | Type | Description |
|--------|------|-------------|
| customerID | string | Unique customer identifier |
| tenure | int | Months with company |
| MonthlyCharges | float | Monthly subscription cost |
| TotalCharges | float | Cumulative charges |
| Contract | string | Month-to-month, One year, Two year |
| Churn | string | Yes/No churn label |
| intervention_received | int | 0/1 synthetic treatment flag |

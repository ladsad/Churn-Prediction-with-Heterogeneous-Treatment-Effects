"""
Tests for churn prediction and causal models.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPreprocessing:
    """Tests for preprocessing module."""
    
    def test_encode_categoricals(self):
        """Test categorical encoding."""
        from src.preprocess import encode_categoricals
        
        df = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'Churn': ['No', 'Yes', 'No']
        })
        
        result = encode_categoricals(df)
        
        assert 'gender_encoded' in result.columns
        assert 'Contract_encoded' in result.columns
        assert result['gender_encoded'].dtype in [np.int64, np.int32]
    
    def test_engineer_features(self):
        """Test feature engineering."""
        from src.preprocess import engineer_features
        
        df = pd.DataFrame({
            'tenure': [5, 25, 50],
            'MonthlyCharges': [30, 70, 100],
            'Churn': ['No', 'Yes', 'No']
        })
        
        result = engineer_features(df)
        
        assert 'tenure_years' in result.columns
        assert 'high_value' in result.columns
        assert 'new_customer' in result.columns
        assert 'long_tenure' in result.columns
        assert result['tenure_years'].iloc[0] == 5 / 12
    
    def test_get_feature_columns(self):
        """Test feature column list."""
        from src.preprocess import get_feature_columns
        
        columns = get_feature_columns()
        
        assert isinstance(columns, list)
        assert 'tenure' in columns
        assert 'MonthlyCharges' in columns


class TestSimulation:
    """Tests for simulation module."""
    
    def test_compute_churn_risk(self):
        """Test churn risk computation."""
        from src.simulation import compute_churn_risk
        
        df = pd.DataFrame({
            'tenure': [5, 25, 50],
            'MonthlyCharges': [30, 70, 100],
            'TotalCharges': [150, 1750, 5000],
            'Churn': ['Yes', 'No', 'No']
        })
        
        risk = compute_churn_risk(df)
        
        assert len(risk) == 3
        assert all(0 <= r <= 1 for r in risk)
    
    def test_assign_interventions(self):
        """Test intervention assignment."""
        from src.simulation import assign_interventions
        
        risk = np.array([0.1, 0.5, 0.9])
        interventions = assign_interventions(risk)
        
        assert len(interventions) == 3
        assert all(i in [0, 1] for i in interventions)
    
    def test_simulate_treatment_effects(self):
        """Test treatment effect simulation."""
        from src.simulation import simulate_treatment_effects
        
        effects = simulate_treatment_effects(100)
        
        assert len(effects) == 100
        assert all(0 <= e <= 0.5 for e in effects)


class TestEvaluation:
    """Tests for evaluation module."""
    
    def test_compute_auc_roc(self):
        """Test AUC-ROC computation."""
        from src.evaluation import compute_auc_roc
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.3, 0.7, 0.9])
        
        auc = compute_auc_roc(y_true, y_pred)
        
        assert 0 <= auc <= 1
        assert auc == 1.0  # Perfect separation
    
    def test_check_common_support(self):
        """Test common support check."""
        from src.evaluation import check_common_support
        
        ps = np.array([0.2, 0.3, 0.5, 0.6, 0.7, 0.8])
        treatment = np.array([0, 0, 0, 1, 1, 1])
        
        has_overlap, measure = check_common_support(ps, treatment)
        
        assert isinstance(has_overlap, bool)
        assert isinstance(measure, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

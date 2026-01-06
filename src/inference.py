"""
Inference module for real-time intervention recommendations.

Contains the InterventionRecommender class for production use.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass
class InterventionRecommendation:
    """Result of intervention recommendation."""
    customer_id: str
    churn_probability: float
    recommended_intervention: str
    predicted_churn_reduction: float
    confidence_interval: Tuple[float, float]
    alternatives: List[Dict[str, Any]]


class InterventionRecommender:
    """
    Real-time intervention recommendation engine.
    
    Loads trained models and provides recommendations for individual customers.
    """
    
    def __init__(
        self,
        churn_model: Any,
        causal_forest: Any,
        scaler: Any,
        feature_columns: List[str],
        churn_threshold: float = 0.3
    ):
        """
        Initialize recommender with trained models.
        
        Args:
            churn_model: Trained LightGBM churn predictor.
            causal_forest: Trained CausalForestDML.
            scaler: Fitted StandardScaler.
            feature_columns: List of feature column names.
            churn_threshold: Minimum churn probability to trigger intervention.
        """
        self.churn_model = churn_model
        self.causal_forest = causal_forest
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.churn_threshold = churn_threshold
        
        # Intervention costs (can be configured)
        self.intervention_costs = {
            'discount_99': 99,
            'discount_199': 199,
            'feature_upsell': 0,  # Revenue positive
            'support_call': 50,
            'none': 0
        }
    
    def predict_churn(self, X: np.ndarray) -> float:
        """
        Predict churn probability for a customer.
        
        Args:
            X: Feature vector (scaled).
            
        Returns:
            Churn probability.
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self.churn_model.predict(X)[0]
    
    def estimate_treatment_effects(self, X: np.ndarray) -> Dict[str, float]:
        """
        Estimate treatment effects for each intervention type.
        
        Args:
            X: Feature vector.
            
        Returns:
            Dictionary mapping intervention type to predicted effect.
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # For single binary treatment model, get base effect
        base_effect = self.causal_forest.effect(X)[0]
        
        # Simulate different intervention magnitudes
        # (In production, train separate models for each type)
        effects = {
            'discount_99': base_effect * 0.6,
            'discount_199': base_effect * 1.0,
            'feature_upsell': base_effect * 0.4,
            'support_call': base_effect * 0.5,
            'none': 0.0
        }
        
        return effects
    
    def get_confidence_intervals(
        self,
        X: np.ndarray
    ) -> Tuple[float, float]:
        """
        Get confidence interval for treatment effect.
        
        Args:
            X: Feature vector.
            
        Returns:
            Tuple of (lower, upper) bounds.
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        try:
            lb, ub = self.causal_forest.effect_interval(X, alpha=0.05)
            return (lb[0], ub[0])
        except:
            # Fallback if interval not available
            effect = self.causal_forest.effect(X)[0]
            return (effect - 0.05, effect + 0.05)
    
    def recommend_intervention(
        self,
        customer_id: str,
        features: Dict[str, Any]
    ) -> InterventionRecommendation:
        """
        Get intervention recommendation for a customer.
        
        Args:
            customer_id: Customer identifier.
            features: Dictionary of customer features.
            
        Returns:
            InterventionRecommendation with best action.
        """
        # Prepare features
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        X_scaled = self.scaler.transform(X)
        
        # Predict churn
        churn_prob = self.predict_churn(X_scaled)
        
        # If low risk, no intervention needed
        if churn_prob < self.churn_threshold:
            return InterventionRecommendation(
                customer_id=customer_id,
                churn_probability=churn_prob,
                recommended_intervention='none',
                predicted_churn_reduction=0.0,
                confidence_interval=(0.0, 0.0),
                alternatives=[]
            )
        
        # Estimate effects for each intervention
        effects = self.estimate_treatment_effects(X_scaled)
        
        # Find best intervention (most negative effect = biggest churn reduction)
        best_intervention = min(
            [k for k in effects if k != 'none'],
            key=lambda k: effects[k]  # More negative = better
        )
        
        # Get confidence interval
        ci = self.get_confidence_intervals(X_scaled)
        
        # Build alternatives list
        alternatives = [
            {
                'intervention_type': itype,
                'predicted_effect': effects[itype],
                'cost': self.intervention_costs[itype]
            }
            for itype in effects
            if itype != 'none'
        ]
        alternatives = sorted(alternatives, key=lambda x: x['predicted_effect'])
        
        return InterventionRecommendation(
            customer_id=customer_id,
            churn_probability=churn_prob,
            recommended_intervention=best_intervention,
            predicted_churn_reduction=-effects[best_intervention],
            confidence_interval=(-ci[1], -ci[0]),
            alternatives=alternatives
        )
    
    @classmethod
    def load(cls, models_dir: Optional[Path] = None) -> 'InterventionRecommender':
        """
        Load recommender with all saved models.
        
        Args:
            models_dir: Path to models directory.
            
        Returns:
            Configured InterventionRecommender.
        """
        import pickle
        
        if models_dir is None:
            models_dir = MODELS_DIR
        
        with open(models_dir / 'churn_model.pkl', 'rb') as f:
            churn_model = pickle.load(f)
        
        with open(models_dir / 'causal_forest.pkl', 'rb') as f:
            causal_forest = pickle.load(f)
        
        with open(models_dir / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open(models_dir / 'feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        return cls(
            churn_model=churn_model,
            causal_forest=causal_forest,
            scaler=scaler,
            feature_columns=feature_columns
        )

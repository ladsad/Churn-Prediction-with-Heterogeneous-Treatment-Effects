"""
Model training module for Churn HTE project.

Contains functions for training:
- Churn prediction model (LightGBM)
- Propensity score model (Logistic Regression)
- Outcome models for doubly robust estimation
- Causal Forest for heterogeneous treatment effects
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, Any

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[dict] = None
) -> lgb.Booster:
    """
    Train LightGBM churn prediction model.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (optional).
        y_val: Validation labels (optional).
        params: LightGBM parameters (optional).
        
    Returns:
        Trained LightGBM Booster.
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'random_state': 42,
            'verbosity': -1
        }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    valid_sets = [train_data]
    if X_val is not None and y_val is not None:
        valid_data = lgb.Dataset(X_val, label=y_val)
        valid_sets.append(valid_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=valid_sets,
        callbacks=[lgb.log_evaluation(period=0)]
    )
    
    return model


def train_propensity_model(
    X: pd.DataFrame,
    treatment: pd.Series,
    max_iter: int = 1000
) -> Tuple[LogisticRegression, np.ndarray]:
    """
    Train propensity score model (probability of receiving treatment).
    
    Args:
        X: Feature matrix.
        treatment: Binary treatment indicator.
        max_iter: Maximum iterations for logistic regression.
        
    Returns:
        Tuple of (fitted model, propensity scores).
    """
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(X, treatment)
    
    propensity_scores = model.predict_proba(X)[:, 1]
    
    return model, propensity_scores


def train_outcome_models(
    X: pd.DataFrame,
    y: pd.Series,
    treatment: pd.Series,
    n_estimators: int = 50,
    max_depth: int = 5
) -> Tuple[RandomForestRegressor, RandomForestRegressor]:
    """
    Train separate outcome models for treatment and control groups.
    
    Used in doubly robust estimation.
    
    Args:
        X: Feature matrix.
        y: Outcome variable (churn).
        treatment: Binary treatment indicator.
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        
    Returns:
        Tuple of (treatment outcome model, control outcome model).
    """
    # Treatment group
    X_treatment = X[treatment == 1]
    y_treatment = y[treatment == 1]
    
    outcome_model_treatment = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    outcome_model_treatment.fit(X_treatment, y_treatment)
    
    # Control group
    X_control = X[treatment == 0]
    y_control = y[treatment == 0]
    
    outcome_model_control = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    outcome_model_control.fit(X_control, y_control)
    
    return outcome_model_treatment, outcome_model_control


def train_causal_forest(
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    n_trees: int = 100,
    min_leaf_size: int = 10
) -> Any:
    """
    Train Causal Forest for heterogeneous treatment effects.
    
    Uses econml's CausalForestDML.
    
    Args:
        X: Feature matrix.
        y: Outcome variable.
        treatment: Treatment indicator.
        n_trees: Number of trees in forest.
        min_leaf_size: Minimum samples in leaf.
        
    Returns:
        Fitted CausalForestDML model.
    """
    from econml.dml import CausalForestDML
    
    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
        model_t=RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42),
        n_estimators=n_trees,
        min_samples_leaf=min_leaf_size,
        random_state=42
    )
    
    cf.fit(Y=y, T=treatment, X=X)
    
    return cf


def save_model(model: Any, name: str) -> Path:
    """
    Save model to pickle file.
    
    Args:
        model: Model object to save.
        name: Filename (without extension).
        
    Returns:
        Path to saved file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / f"{name}.pkl"
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    return filepath


def load_model(name: str) -> Any:
    """
    Load model from pickle file.
    
    Args:
        name: Filename (without extension).
        
    Returns:
        Loaded model object.
    """
    filepath = MODELS_DIR / f"{name}.pkl"
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    return model

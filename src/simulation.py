"""
Simulation module for generating synthetic interventions.

Creates confounded intervention data to enable causal inference problem setup.
The interventions are assigned with selection bias (high-risk customers more likely
to receive interventions), which creates the confounding we need to address.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Optional


def compute_churn_risk(
    df: pd.DataFrame, 
    features: list = ['tenure', 'MonthlyCharges', 'TotalCharges'],
    random_state: int = 42
) -> np.ndarray:
    """
    Compute initial churn risk scores using a simple model.
    
    This simulates historical decision-making that drives selection bias.
    
    Args:
        df: DataFrame with features and churn outcomes.
        features: Columns to use for risk scoring.
        random_state: Random seed.
        
    Returns:
        Array of churn risk probabilities.
    """
    X = df[features].fillna(0)
    
    # Create binary churn target
    if df['Churn'].dtype == object:
        y = (df['Churn'] == 'Yes').astype(int)
    else:
        y = df['Churn']
    
    # Train quick RF model
    risk_model = RandomForestClassifier(
        n_estimators=10, 
        max_depth=5,
        random_state=random_state
    )
    risk_model.fit(X, y)
    
    # Predict probabilities
    churn_risk = risk_model.predict_proba(X)[:, 1]
    
    return churn_risk


def assign_interventions(
    churn_risk: np.ndarray,
    base_prob: float = 0.2,
    risk_weight: float = 0.6,
    random_state: int = 42
) -> np.ndarray:
    """
    Assign interventions with selection bias based on churn risk.
    
    High-risk customers are more likely to receive interventions,
    creating the confounding problem for causal inference.
    
    Args:
        churn_risk: Array of churn risk probabilities.
        base_prob: Baseline probability of receiving intervention.
        risk_weight: How much churn risk affects intervention probability.
        random_state: Random seed.
        
    Returns:
        Binary array indicating intervention assignment (1=received, 0=not).
    """
    np.random.seed(random_state)
    
    # Higher risk -> higher treatment probability (confounding!)
    treatment_prob = base_prob + risk_weight * churn_risk
    treatment_prob = np.clip(treatment_prob, 0, 1)
    
    # Random assignment based on probability
    intervention_received = np.random.binomial(1, treatment_prob)
    
    return intervention_received


def simulate_treatment_effects(
    n: int,
    mean_effect: float = 0.25,
    std_effect: float = 0.10,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate heterogeneous treatment effects.
    
    Different customers have different responsiveness to intervention.
    
    Args:
        n: Number of customers.
        mean_effect: Average reduction in churn probability from intervention.
        std_effect: Standard deviation of effect heterogeneity.
        random_state: Random seed.
        
    Returns:
        Array of treatment effect magnitudes (positive = reduces churn).
    """
    np.random.seed(random_state)
    
    treatment_effects = np.random.normal(mean_effect, std_effect, n)
    treatment_effects = np.clip(treatment_effects, 0, 0.5)  # Cap at 50% reduction
    
    return treatment_effects


def apply_treatment_effects(
    churn_observed: np.ndarray,
    intervention_received: np.ndarray,
    treatment_effects: np.ndarray,
    random_state: int = 42
) -> np.ndarray:
    """
    Apply treatment effects to compute counterfactual outcomes.
    
    For customers who received intervention, their churn probability
    is reduced by their individual treatment effect.
    
    Args:
        churn_observed: Original binary churn outcomes.
        intervention_received: Binary intervention indicator.
        treatment_effects: Individual treatment effect magnitudes.
        random_state: Random seed.
        
    Returns:
        Array of counterfactual churn outcomes (what happened after intervention).
    """
    np.random.seed(random_state)
    
    churn_counterfactual = churn_observed.copy()
    
    for i in range(len(churn_observed)):
        if intervention_received[i] == 1:
            # With intervention, reduce churn probability
            original_prob = churn_observed[i]
            reduced_prob = max(0, original_prob - treatment_effects[i])
            churn_counterfactual[i] = np.random.binomial(1, reduced_prob * 0.5)  # Reduced outcome
    
    return churn_counterfactual


def create_confounded_dataset(
    df: pd.DataFrame,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Master function to create a dataset with synthetic confounded interventions.
    
    This creates the causal inference problem: observed differences between
    treatment and control groups are biased because high-risk customers
    are more likely to receive treatment.
    
    Args:
        df: Raw DataFrame with churn outcomes.
        random_state: Random seed.
        
    Returns:
        DataFrame with added intervention columns:
        - intervention_received: Binary treatment indicator
        - churn_observed: Original churn outcome
        - treatment_effect: True (unobserved) individual effect
        - churn_counterfactual: Outcome after applying treatment effects
    """
    df = df.copy()
    n = len(df)
    
    # Step 1: Compute churn risk scores
    churn_risk = compute_churn_risk(df, random_state=random_state)
    df['churn_risk'] = churn_risk
    
    # Step 2: Assign interventions with selection bias
    intervention_received = assign_interventions(churn_risk, random_state=random_state)
    df['intervention_received'] = intervention_received
    
    # Step 3: Store original churn outcome
    if df['Churn'].dtype == object:
        df['churn_observed'] = (df['Churn'] == 'Yes').astype(int)
    else:
        df['churn_observed'] = df['Churn']
    
    # Step 4: Generate heterogeneous treatment effects
    treatment_effects = simulate_treatment_effects(n, random_state=random_state)
    df['treatment_effect'] = treatment_effects
    
    # Step 5: Apply treatment effects
    df['churn_counterfactual'] = apply_treatment_effects(
        df['churn_observed'].values,
        df['intervention_received'].values,
        treatment_effects,
        random_state=random_state
    )
    
    # Summary statistics
    print("=== Synthetic Intervention Summary ===")
    print(f"Total customers: {n}")
    print(f"Intervention rate: {df['intervention_received'].mean():.1%}")
    print(f"Churn rate (no intervention): {df[df['intervention_received']==0]['churn_observed'].mean():.1%}")
    print(f"Churn rate (with intervention): {df[df['intervention_received']==1]['churn_observed'].mean():.1%}")
    print(f"Mean treatment effect: {df['treatment_effect'].mean():.3f}")
    print("")
    print("NOTE: The observed difference is CONFOUNDED - high-risk customers")
    print("are more likely to receive intervention. Use causal methods!")
    
    return df

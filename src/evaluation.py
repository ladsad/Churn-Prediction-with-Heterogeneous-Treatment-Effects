"""
Evaluation module for Churn HTE project.

Contains functions for:
- Model performance metrics (AUC-ROC, precision-recall)
- Causal effect estimation (doubly robust ATE)
- A/B test statistical analysis
- Segment-level treatment effect analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve


def compute_auc_roc(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """
    Compute AUC-ROC score for churn prediction.
    
    Args:
        y_true: True binary labels.
        y_pred_prob: Predicted probabilities.
        
    Returns:
        AUC-ROC score.
    """
    return roc_auc_score(y_true, y_pred_prob)


def compute_precision_at_recall(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    target_recall: float = 0.5
) -> float:
    """
    Compute precision at a specific recall threshold.
    
    Args:
        y_true: True binary labels.
        y_pred_prob: Predicted probabilities.
        target_recall: Target recall value.
        
    Returns:
        Precision at target recall.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    idx = np.argmin(np.abs(recall - target_recall))
    return precision[idx]


def compute_doubly_robust_ate(
    y: np.ndarray,
    treatment: np.ndarray,
    propensity_scores: np.ndarray,
    y_pred_treated: np.ndarray,
    y_pred_control: np.ndarray
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Compute Average Treatment Effect using Doubly Robust estimation.
    
    ATE_DR = E[ T*(y - mu1(X))/ps - (1-T)*(y - mu0(X))/(1-ps) + mu1(X) - mu0(X) ]
    
    This estimator is robust: it's consistent if either the propensity score
    model OR the outcome models are correctly specified.
    
    Args:
        y: Observed outcomes.
        treatment: Binary treatment indicator.
        propensity_scores: P(T=1 | X).
        y_pred_treated: E[Y | T=1, X] predictions for all X.
        y_pred_control: E[Y | T=0, X] predictions for all X.
        
    Returns:
        Tuple of (ATE, standard error, 95% confidence interval).
    """
    ps = propensity_scores
    
    # Clip propensity scores to avoid division issues
    ps = np.clip(ps, 0.01, 0.99)
    
    # Doubly robust terms
    ate_terms = (
        (treatment * (y - y_pred_treated) / ps) -
        ((1 - treatment) * (y - y_pred_control) / (1 - ps)) +
        (y_pred_treated - y_pred_control)
    )
    
    ate = ate_terms.mean()
    ate_se = ate_terms.std() / np.sqrt(len(ate_terms))
    ate_ci = (ate - 1.96 * ate_se, ate + 1.96 * ate_se)
    
    return ate, ate_se, ate_ci


def run_ab_test(
    treatment_outcomes: np.ndarray,
    control_outcomes: np.ndarray
) -> Dict[str, Any]:
    """
    Run A/B test statistical analysis using two-proportion z-test.
    
    Args:
        treatment_outcomes: Binary outcomes for treatment group.
        control_outcomes: Binary outcomes for control group.
        
    Returns:
        Dictionary with test results.
    """
    treatment_n = len(treatment_outcomes)
    treatment_successes = treatment_outcomes.sum()
    treatment_rate = treatment_successes / treatment_n
    
    control_n = len(control_outcomes)
    control_successes = control_outcomes.sum()
    control_rate = control_successes / control_n
    
    # Two-proportion z-test
    count = np.array([treatment_successes, control_successes])
    nobs = np.array([treatment_n, control_n])
    
    from statsmodels.stats.proportion import proportions_ztest
    z_stat, p_value = proportions_ztest(count, nobs)
    
    # Effect metrics
    absolute_diff = control_rate - treatment_rate
    relative_uplift = absolute_diff / control_rate if control_rate > 0 else 0
    
    return {
        'treatment_n': treatment_n,
        'treatment_rate': treatment_rate,
        'control_n': control_n,
        'control_rate': control_rate,
        'absolute_difference': absolute_diff,
        'relative_uplift': relative_uplift,
        'z_statistic': z_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def analyze_segment_effects(
    df: pd.DataFrame,
    treatment_effects: np.ndarray,
    segment_column: str
) -> pd.DataFrame:
    """
    Analyze treatment effects by customer segment.
    
    Args:
        df: DataFrame with customer data.
        treatment_effects: Array of individual treatment effects.
        segment_column: Column name to segment by.
        
    Returns:
        DataFrame with segment-level effect statistics.
    """
    df = df.copy()
    df['treatment_effect'] = treatment_effects
    
    segment_stats = df.groupby(segment_column)['treatment_effect'].agg([
        'count',
        'mean',
        'std',
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(4)
    
    return segment_stats


def check_common_support(
    propensity_scores: np.ndarray,
    treatment: np.ndarray,
    threshold: float = 0.1
) -> Tuple[bool, float]:
    """
    Check overlap in propensity score distributions (common support).
    
    Good overlap is required for credible causal estimates.
    
    Args:
        propensity_scores: P(T=1 | X).
        treatment: Binary treatment indicator.
        threshold: Minimum acceptable overlap.
        
    Returns:
        Tuple of (has_good_overlap, overlap_measure).
    """
    ps_treated = propensity_scores[treatment == 1]
    ps_control = propensity_scores[treatment == 0]
    
    # Compute overlap region
    min_overlap = min(ps_treated.min(), ps_control.min())
    max_overlap = max(ps_treated.max(), ps_control.max())
    
    # Proportion of samples in overlap region
    overlap_range = (max(ps_treated.min(), ps_control.min()),
                     min(ps_treated.max(), ps_control.max()))
    
    overlap_measure = overlap_range[1] - overlap_range[0]
    has_good_overlap = overlap_measure > threshold
    
    return has_good_overlap, overlap_measure

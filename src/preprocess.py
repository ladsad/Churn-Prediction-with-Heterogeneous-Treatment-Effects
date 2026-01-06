"""
Data preprocessing module for Churn HTE project.

Handles loading, cleaning, encoding, and feature engineering for the Telco Churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, List, Optional


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_telco_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Telco Customer Churn dataset.
    
    Args:
        filepath: Path to CSV file. If None, looks in data/raw/.
        
    Returns:
        Raw DataFrame with all columns.
    """
    if filepath is None:
        filepath = DATA_RAW / "telco_churn.csv"
    
    df = pd.read_csv(filepath)
    
    # Convert TotalCharges to numeric (has some empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    return df


def encode_categoricals(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Label encode categorical columns.
    
    Args:
        df: Input DataFrame.
        columns: List of columns to encode. If None, auto-detect.
        
    Returns:
        DataFrame with encoded columns added (suffix '_encoded').
    """
    df = df.copy()
    
    if columns is None:
        columns = ['gender', 'Contract', 'InternetService', 'OnlineSecurity',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 
                   'PaymentMethod', 'PaperlessBilling']
    
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            # Handle missing values
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for modeling.
    
    Args:
        df: Input DataFrame with raw features.
        
    Returns:
        DataFrame with additional engineered features.
    """
    df = df.copy()
    
    # Tenure in years
    df['tenure_years'] = df['tenure'] / 12
    
    # Customer value segments
    df['high_value'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
    
    # Tenure segments
    df['long_tenure'] = (df['tenure'] > 24).astype(int)
    df['new_customer'] = (df['tenure'] < 6).astype(int)
    
    # Churn binary (if string)
    if df['Churn'].dtype == object:
        df['churn_binary'] = (df['Churn'] == 'Yes').astype(int)
    else:
        df['churn_binary'] = df['Churn']
    
    return df


def get_feature_columns() -> List[str]:
    """
    Get the list of feature columns for modeling.
    
    Returns:
        List of feature column names.
    """
    return [
        'tenure', 'tenure_years', 'MonthlyCharges', 'TotalCharges',
        'gender_encoded', 'Contract_encoded', 'InternetService_encoded',
        'OnlineSecurity_encoded', 'TechSupport_encoded',
        'high_value', 'long_tenure', 'new_customer'
    ]


def scale_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize features using StandardScaler.
    
    Args:
        X: Feature matrix.
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled, scaler


def split_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.3, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with stratification.
    
    Args:
        X: Feature matrix.
        y: Target variable.
        test_size: Fraction for test set.
        random_state: Random seed.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def prepare_modeling_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full pipeline: encode, engineer, select features.
    
    Args:
        df: Raw DataFrame.
        
    Returns:
        Tuple of (feature matrix X, target y).
    """
    # Encode categoricals
    df = encode_categoricals(df)
    
    # Engineer features
    df = engineer_features(df)
    
    # Select feature columns
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[available_cols].fillna(0)
    y = df['churn_binary']
    
    return X, y

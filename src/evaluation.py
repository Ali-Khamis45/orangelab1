"""
Evaluation module for computing cross-validation scores and detailed metrics.
Supports Stratified K-Fold for classification and K-Fold for regression.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.utils.class_weight import compute_sample_weight


def cross_validate_classifier(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    use_sample_weight: bool = False
) -> Dict[str, Tuple[float, float]]:
    """
    Performs Stratified K-Fold cross-validation on a classifier.
    Computes Mean and Std Dev for Accuracy, Precision, Recall, F1, and ROC-AUC.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        # Train model
        if use_sample_weight:
            sample_w = compute_sample_weight(class_weight='balanced', y=y_tr)
            model.fit(X_tr, y_tr, sample_weight=sample_w)
        else:
            model.fit(X_tr, y_tr)
            
        preds = model.predict(X_va)
        probs = model.predict_proba(X_va)[:, 1]
        
        metrics['accuracy'].append(accuracy_score(y_va, preds))
        metrics['precision'].append(precision_score(y_va, preds, zero_division=0))
        metrics['recall'].append(recall_score(y_va, preds, zero_division=0))
        metrics['f1'].append(f1_score(y_va, preds, zero_division=0))
        metrics['roc_auc'].append(roc_auc_score(y_va, probs))
        
    results = {}
    for name, vals in metrics.items():
        results[name] = (float(np.mean(vals)), float(np.std(vals)))
        
    return results


def cross_validate_regressor(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5
) -> Dict[str, Tuple[float, float]]:
    """
    Performs K-Fold cross-validation on a regressor.
    Computes Mean and Std Dev for MAE, MSE, RMSE, and R2.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    metrics = {
        'mae': [],
        'mse': [],
        'rmse': [],
        'r2': []
    }
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        
        mae_val = mean_absolute_error(y_va, preds)
        mse_val = mean_squared_error(y_va, preds)
        r2_val = r2_score(y_va, preds)
        
        metrics['mae'].append(mae_val)
        metrics['mse'].append(mse_val)
        metrics['rmse'].append(np.sqrt(mse_val))
        metrics['r2'].append(r2_val)
        
    results = {}
    for name, vals in metrics.items():
        results[name] = (float(np.mean(vals)), float(np.std(vals)))
        
    return results


def evaluate_classifier_test_set(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_sample_weight: bool = False
) -> Dict[str, Any]:
    """
    Evaluates classifier on holdout test set.
    """
    if use_sample_weight:
        sample_w = compute_sample_weight(class_weight='balanced', y=y_train)
        model.fit(X_train, y_train, sample_weight=sample_w)
    else:
        model.fit(X_train, y_train)
        
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    return {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probs),
        'predictions': preds,
        'probabilities': probs
    }


def evaluate_regressor_test_set(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """
    Evaluates regressor on holdout test set.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    
    return {
        'mae': mean_absolute_error(y_test, preds),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2_score(y_test, preds),
        'predictions': preds
    }

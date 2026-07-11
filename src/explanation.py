"""
Model explainability module.
Implements Logistic Regression coefficient analysis and Permutation Importance
computations for tree-based models.
"""

import os
from typing import List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

# Color configurations
COEFF_POS_COLOR = '#4f46e5'  # Indigo (Positive risk/coefficient)
COEFF_NEG_COLOR = '#10b981'  # Emerald (Negative risk/coefficient)
IMPORTANCE_COLOR = '#2b5c8f' # Slate Blue


def plot_logistic_regression_coefficients(
    coefs: np.ndarray, 
    feature_names: List[str], 
    save_path: str,
    top_n: int = 15
):
    """
    Plots Logistic Regression coefficients sorted by absolute magnitude.
    Colors represent positive vs negative impact on target risk.
    """
    df_coef = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs[0]
    })
    df_coef['Abs_Coefficient'] = df_coef['Coefficient'].abs()
    df_coef = df_coef.sort_values(by='Abs_Coefficient', ascending=False).head(top_n)
    
    # Sort again by actual coefficient value for visualization ordering
    df_coef = df_coef.sort_values(by='Coefficient', ascending=True)
    
    # Define colors based on positive vs negative risk impact
    colors = [COEFF_POS_COLOR if c >= 0 else COEFF_NEG_COLOR for c in df_coef['Coefficient']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_coef['Feature'], df_coef['Coefficient'], color=colors, edgecolor='none', height=0.7)
    
    # Custom axes lines and formatting
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel('Coefficient Value (Impact on Log Odds)')
    plt.ylabel('Feature')
    plt.title('Top Feature Coefficients (Logistic Regression)')
    
    # Add annotation labels for positive and negative impacts
    plt.text(
        df_coef['Coefficient'].max() * 0.1 if df_coef['Coefficient'].max() > 0 else 0.1,
        len(df_coef) - 1.5,
        "Increases Risk (Positive Coef)",
        color=COEFF_POS_COLOR,
        fontweight='bold',
        fontsize=9
    )
    plt.text(
        df_coef['Coefficient'].min() * 0.8 if df_coef['Coefficient'].min() < 0 else -0.8,
        0.5,
        "Decreases Risk (Negative Coef)",
        color=COEFF_NEG_COLOR,
        fontweight='bold',
        fontsize=9
    )
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_permutation_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    save_path: str,
    top_n: int = 15
):
    """
    Computes and plots Permutation Importance for any trained model on test data.
    Provides a statistically robust explainability alternative to default Gini importance.
    """
    print("Computing Permutation Importance (this may take a few seconds)...")
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    
    sorted_importances_idx = result.importances_mean.argsort()[::-1]
    
    top_idx = sorted_importances_idx[:top_n]
    
    df_imp = pd.DataFrame({
        'Feature': [feature_names[i] for i in top_idx],
        'Importance_Mean': result.importances_mean[top_idx],
        'Importance_Std': result.importances_std[top_idx]
    })
    
    df_imp = df_imp.sort_values(by='Importance_Mean', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(
        df_imp['Feature'], 
        df_imp['Importance_Mean'], 
        xerr=df_imp['Importance_Std'],
        color=IMPORTANCE_COLOR,
        error_kw={'ecolor': 'gray', 'capsize': 3, 'linewidth': 0.8},
        height=0.6
    )
    
    plt.xlabel('Decrease in Model Score (Accuracy/AUC)')
    plt.ylabel('Feature')
    plt.title('Permutation Feature Importances (On Test Dataset)')
    
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

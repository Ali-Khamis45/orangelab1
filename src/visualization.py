"""
Visualization module for generating and saving production-grade evaluation plots.
Supports ROC Curves, Precision-Recall, Calibration, Confusion Matrix, and Regression diagnostics.
"""

import os
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve

# Color Palette Config (recruiter-friendly professional dashboard colors)
PRIMARY_COLOR = '#4f46e5'  # Indigo
ACCENT_COLOR = '#de425b'   # Coral
SAFE_COLOR = '#10b981'     # Emerald
SLATE_COLOR = '#2b5c8f'    # Slate Blue
GRAY_COLOR = '#94a3b8'     # Slate Gray


def setup_style():
    """Sets up global matplotlib/seaborn plotting themes."""
    sns.set_theme(style='whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Outfit', 'Inter', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10


def plot_roc_curves(
    model_curves: Dict[str, Tuple[np.ndarray, np.ndarray]], 
    save_path: str
):
    """
    Plots ROC curves for multiple models on the same plot.
    `model_curves` is a dict of name -> (y_test, y_probs)
    """
    setup_style()
    plt.figure(figsize=(8, 6))
    
    colors = [PRIMARY_COLOR, SLATE_COLOR, SAFE_COLOR, '#f59e0b', '#8b5cf6']
    
    for idx, (name, (y_test, y_probs)) in enumerate(model_curves.items()):
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        color = colors[idx % len(colors)]
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC = {roc_auc:.4f})')
        
    plt.plot([0, 1], [0, 1], color=GRAY_COLOR, lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right", frameon=True, facecolor='white', edgecolor=GRAY_COLOR)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_precision_recall_curves(
    model_curves: Dict[str, Tuple[np.ndarray, np.ndarray]], 
    save_path: str
):
    """
    Plots Precision-Recall curves for multiple models.
    """
    setup_style()
    plt.figure(figsize=(8, 6))
    
    colors = [PRIMARY_COLOR, SLATE_COLOR, SAFE_COLOR, '#f59e0b', '#8b5cf6']
    
    for idx, (name, (y_test, y_probs)) in enumerate(model_curves.items()):
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        avg_prec = average_precision_score(y_test, y_probs)
        color = colors[idx % len(colors)]
        plt.plot(recall, precision, color=color, lw=2.5, label=f'{name} (AP = {avg_prec:.4f})')
        
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left", frameon=True, facecolor='white', edgecolor=GRAY_COLOR)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_calibration_curves(
    model_curves: Dict[str, Tuple[np.ndarray, np.ndarray]], 
    save_path: str
):
    """
    Plots Calibration curves (reliability diagrams) for multiple models.
    """
    setup_style()
    plt.figure(figsize=(8, 6))
    
    colors = [PRIMARY_COLOR, SLATE_COLOR, SAFE_COLOR, '#f59e0b', '#8b5cf6']
    
    for idx, (name, (y_test, y_probs)) in enumerate(model_curves.items()):
        prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
        color = colors[idx % len(colors)]
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color=color, label=name)
        
    plt.plot([0, 1], [0, 1], color=GRAY_COLOR, lw=1.5, linestyle='--', label='Perfect Calibration')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Probability Calibration Curves (Reliability Diagrams)')
    plt.legend(loc="upper left", frameon=True, facecolor='white', edgecolor=GRAY_COLOR)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    y_test: np.ndarray, 
    y_preds: np.ndarray, 
    save_path: str
):
    """
    Plots a highly stylized confusion matrix heatmap.
    """
    setup_style()
    cm = confusion_matrix(y_test, y_preds)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        xticklabels=['Retained (0)', 'Terminated (1)'],
        yticklabels=['Retained (0)', 'Terminated (1)'],
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Model Confusion Matrix')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residuals(
    y_test: np.ndarray, 
    y_preds: np.ndarray, 
    save_path: str
):
    """
    Plots regression diagnostics: Residuals vs. Predictions and Prediction vs. Actual.
    """
    setup_style()
    residuals = y_test - y_preds
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Prediction vs Actual Plot
    sns.scatterplot(x=y_preds, y=y_test, ax=axes[0], color=SLATE_COLOR, alpha=0.6, edgecolor='white')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color=ACCENT_COLOR, lw=2, linestyle='--')
    axes[0].set_xlabel('Predicted Performance Rating')
    axes[0].set_ylabel('Actual Performance Rating')
    axes[0].set_title('Prediction vs. Actual Ratings')
    
    # 2. Residuals Plot
    sns.scatterplot(x=y_preds, y=residuals, ax=axes[1], color=PRIMARY_COLOR, alpha=0.6, edgecolor='white')
    axes[1].axhline(y=0, color=ACCENT_COLOR, lw=2, linestyle='--')
    axes[1].set_xlabel('Predicted Performance Rating')
    axes[1].set_ylabel('Residual (Actual - Predicted)')
    axes[1].set_title('Residuals Analysis')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_learning_curves(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    save_path: str,
    cv: int = 5,
    scoring: str = 'roc_auc'
):
    """
    Plots learning curves showing train vs validation scores over training sizes.
    """
    setup_style()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', color=PRIMARY_COLOR, label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color=PRIMARY_COLOR)
    
    plt.plot(train_sizes, test_mean, 'o-', color=SAFE_COLOR, label='Cross-Validation Score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color=SAFE_COLOR)
    
    plt.xlabel('Training Set Size')
    plt.ylabel(scoring.upper())
    plt.title('Model Learning Curves')
    plt.legend(loc='best')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

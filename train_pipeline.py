"""
End-to-end training and evaluation pipeline script.
Runs cross-validation comparisons, tunes hyperparameters, generates diagnostics charts,
and serializes the production model artifacts.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from src.preprocessing import get_attrition_features, get_rating_features
from src.training import (
    get_classification_models, get_regression_models,
    tune_logistic_regression, tune_random_forest_classifier
)
from src.evaluation import (
    cross_validate_classifier, cross_validate_regressor,
    evaluate_classifier_test_set
)
from src.visualization import (
    plot_roc_curves, plot_precision_recall_curves, plot_calibration_curves,
    plot_confusion_matrix, plot_residuals, plot_learning_curves
)
from src.explanation import (
    plot_logistic_regression_coefficients, plot_permutation_importance
)

PLOT_DIR = 'static/plots'
MODEL_DIR = 'ml_models'


def run_classification_pipeline(df: pd.DataFrame):
    """
    Runs classification benchmarking, tuning, and plots generation for Attrition.
    """
    print("\n--- RUNNING ATTRITION CLASSIFICATION PIPELINE ---")
    
    # 1. Load and split data
    X, y, num_cols = get_attrition_features(df, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical columns
    scaler = StandardScaler()
    # Fit scaler on training feature set
    X_train_scaled_vals = scaler.fit_transform(X_train)
    X_test_scaled_vals = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled_vals, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled_vals, columns=X.columns)
    
    # 2. Benchmarking default models with Stratified K-Fold Cross Validation
    print("\n[CV Benchmarking] Evaluating default classifiers (5-Fold CV)...")
    models = get_classification_models()
    cv_records = []
    
    for name, clf in models.items():
        use_sw = ('Default' not in name) and ('Balanced' not in name)  # Use sample weights for general classifiers
        cv_res = cross_validate_classifier(
            clf, X_train_scaled.values, y_train.values, cv=5, use_sample_weight=use_sw
        )
        
        cv_records.append({
            'Model': name,
            'CV ROC-AUC (Mean)': f"{cv_res['roc_auc'][0]:.4f} ± {cv_res['roc_auc'][1]:.4f}",
            'CV Recall (Mean)': f"{cv_res['recall'][0]:.4f} ± {cv_res['recall'][1]:.4f}",
            'CV F1 (Mean)': f"{cv_res['f1'][0]:.4f} ± {cv_res['f1'][1]:.4f}",
            'CV Accuracy (Mean)': f"{cv_res['accuracy'][0]:.4f} ± {cv_res['accuracy'][1]:.4f}"
        })
        
    df_cv = pd.DataFrame(cv_records)
    print("\n--- 5-Fold Cross Validation Results (Classifiers) ---")
    print(df_cv.to_string(index=False))
    
    # 3. Hyperparameter Tuning using GridSearchCV
    print("\n[Tuning] Running GridSearch for Logistic Regression...")
    best_lr, lr_params = tune_logistic_regression(X_train_scaled.values, y_train.values)
    print(f"Best LR Params: {lr_params}")
    
    print("\n[Tuning] Running GridSearch for Random Forest...")
    best_rf, rf_params = tune_random_forest_classifier(X_train_scaled.values, y_train.values)
    print(f"Best RF Params: {rf_params}")
    
    # 4. Evaluate Tuned Models on Holdout Test Set
    print("\n[Test Set] Evaluating tuned classifiers on unseen data...")
    test_lr = evaluate_classifier_test_set(best_lr, X_train_scaled.values, y_train.values, X_test_scaled.values, y_test.values)
    test_rf = evaluate_classifier_test_set(best_rf, X_train_scaled.values, y_train.values, X_test_scaled.values, y_test.values)
    
    # Train tuned Gradient Boosting Classifier with sample weight balancing
    gb_tuned = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    test_gb = evaluate_classifier_test_set(gb_tuned, X_train_scaled.values, y_train.values, X_test_scaled.values, y_test.values, use_sample_weight=True)
    
    print("\n--- Test Set Performance Metrics Comparison ---")
    metrics_comp = [
        {
            'Model': 'Tuned Logistic Regression (Balanced)',
            'Accuracy': f"{test_lr['accuracy']:.4f}",
            'ROC-AUC': f"{test_lr['roc_auc']:.4f}",
            'Recall (Class 1)': f"{test_lr['recall']:.4f}",
            'F1 Score (Class 1)': f"{test_lr['f1']:.4f}"
        },
        {
            'Model': 'Tuned Random Forest (Balanced)',
            'Accuracy': f"{test_rf['accuracy']:.4f}",
            'ROC-AUC': f"{test_rf['roc_auc']:.4f}",
            'Recall (Class 1)': f"{test_rf['recall']:.4f}",
            'F1 Score (Class 1)': f"{test_rf['f1']:.4f}"
        },
        {
            'Model': 'Tuned Gradient Boosting (Balanced)',
            'Accuracy': f"{test_gb['accuracy']:.4f}",
            'ROC-AUC': f"{test_gb['roc_auc']:.4f}",
            'Recall (Class 1)': f"{test_gb['recall']:.4f}",
            'F1 Score (Class 1)': f"{test_gb['f1']:.4f}"
        }
    ]
    print(pd.DataFrame(metrics_comp).to_string(index=False))
    
    # 5. Save Diagnostic Visualizations
    print(f"\n[Plots] Saving diagnostic charts to {PLOT_DIR}...")
    model_curves = {
        'Logistic Regression (Tuned)': (y_test.values, test_lr['probabilities']),
        'Random Forest (Tuned)': (y_test.values, test_rf['probabilities']),
        'Gradient Boosting (Tuned)': (y_test.values, test_gb['probabilities'])
    }
    
    plot_roc_curves(model_curves, f"{PLOT_DIR}/roc_curves.png")
    plot_precision_recall_curves(model_curves, f"{PLOT_DIR}/pr_curves.png")
    plot_calibration_curves(model_curves, f"{PLOT_DIR}/calibration.png")
    plot_confusion_matrix(y_test.values, test_gb['predictions'], f"{PLOT_DIR}/confusion_matrix.png")
    plot_learning_curves(gb_tuned, X_train_scaled.values, y_train.values, f"{PLOT_DIR}/learning_curves.png", cv=5)
    
    # Save explainability plots
    plot_logistic_regression_coefficients(best_lr.coef_, X.columns.tolist(), f"{PLOT_DIR}/coefficients.png")
    plot_permutation_importance(gb_tuned, X_test_scaled.values, y_test.values, X.columns.tolist(), f"{PLOT_DIR}/feature_importance.png")
    
    # 6. Model Serialization
    # Deploy Tuned Gradient Boosting model with attached Logistic Regression coefficients
    print(f"\n[Save] Serializing production models to {MODEL_DIR}...")
    gb_tuned.coef_ = best_lr.coef_
    gb_tuned.intercept_ = best_lr.intercept_
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/model.pkl", "wb") as f:
        pickle.dump(gb_tuned, f)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{MODEL_DIR}/feature_columns.json", "w") as f:
        json.dump(X.columns.tolist(), f)
        
    # Prepare defaults.json from training median/mode
    defaults = {}
    for col in num_cols:
        defaults[col] = float(X_train[col].median())
    
    # Re-extract cat cols
    cat_cols = [c for c in df.columns if c not in num_cols + ['StartDate', 'ExitDate', 'DOB', 'Training Date', 'Employee ID', 'ADEmail', 'EmployeeStatus', 'Is_Terminated', 'Current Employee Rating']]
    for col in cat_cols:
        if col in df.columns:
            defaults[col] = str(df[col].mode()[0])
            
    with open(f"{MODEL_DIR}/defaults.json", "w") as f:
        json.dump(defaults, f)
        
    print("Serialization completed successfully.")


def run_regression_pipeline(df: pd.DataFrame):
    """
    Runs regression benchmarking and residuals plots generation for Employee Ratings.
    """
    print("\n--- RUNNING PERFORMANCE RATING REGRESSION PIPELINE ---")
    
    X, y, num_cols = get_rating_features(df, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Benchmarking default models with K-Fold Cross Validation
    print("\n[CV Benchmarking] Evaluating regressors (5-Fold CV)...")
    models = get_regression_models()
    cv_records = []
    
    for name, reg in models.items():
        cv_res = cross_validate_regressor(reg, X_train_scaled, y_train.values, cv=5)
        cv_records.append({
            'Model': name,
            'CV RMSE (Mean)': f"{cv_res['rmse'][0]:.4f} ± {cv_res['rmse'][1]:.4f}",
            'CV R² (Mean)': f"{cv_res['r2'][0]:.4f} ± {cv_res['r2'][1]:.4f}",
            'CV MAE (Mean)': f"{cv_res['mae'][0]:.4f} ± {cv_res['mae'][1]:.4f}"
        })
        
    df_cv = pd.DataFrame(cv_records)
    print("\n--- 5-Fold Cross Validation Results (Regressors) ---")
    print(df_cv.to_string(index=False))
    
    # Fit best baseline regressor (Gradient Boosting)
    print("\n[Test Set] Evaluating final regressor on test set...")
    gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_reg.fit(X_train_scaled, y_train)
    preds = gb_reg.predict(X_test_scaled)
    
    # Plot residuals diagnostics
    print(f"\n[Plots] Saving residuals diagnostic plots to {PLOT_DIR}...")
    plot_residuals(y_test.values, preds, f"{PLOT_DIR}/residuals.png")


if __name__ == '__main__':
    # Load dataset
    data = pd.read_csv("cleaned_data_HR.csv")
    
    # Create output directories
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Run pipelines
    run_classification_pipeline(data)
    run_regression_pipeline(data)

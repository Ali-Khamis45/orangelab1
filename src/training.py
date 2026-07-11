"""
Model training and hyperparameter tuning module.
Implements GridSearchCV searches and comparisons for various estimators.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def tune_logistic_regression(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    cv: int = 5,
    scoring: str = 'roc_auc'
) -> Tuple[LogisticRegression, Dict[str, Any]]:
    """
    Performs GridSearchCV to tune Logistic Regression.
    """
    param_grid = [
        {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga'],
            'class_weight': ['balanced', None]
        },
        {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None]
        }
    ]
    grid = GridSearchCV(
        LogisticRegression(max_iter=3000, random_state=42),
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tune_random_forest_classifier(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    cv: int = 5,
    scoring: str = 'roc_auc'
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Performs GridSearchCV to tune RandomForestClassifier.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 10, 12, None],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tune_random_forest_regressor(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    cv: int = 5,
    scoring: str = 'neg_mean_squared_error'
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """
    Performs GridSearchCV to tune RandomForestRegressor.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def get_classification_models(
    sample_weight: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Returns a dictionary of classification models including Decision Tree,
    Random Forest, Extra Trees, and Gradient Boosting.
    Includes XGBoost/LightGBM if available.
    """
    models = {
        'LogisticRegression_Default': LogisticRegression(max_iter=2000, random_state=42),
        'LogisticRegression_Balanced': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
        'DecisionTree_Default': DecisionTreeClassifier(random_state=42),
        'DecisionTree_Balanced': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'RandomForest_Default': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'RandomForest_Balanced': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'ExtraTrees_Default': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'ExtraTrees_Balanced': ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'GradientBoosting_Default': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    # Attempt to load XGBoost
    try:
        import xgboost as xgb
        # Compute scale_pos_weight if sample_weight is provided
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    except ImportError:
        pass

    # Attempt to load LightGBM
    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
    except ImportError:
        pass

    return models


def get_regression_models() -> Dict[str, Any]:
    """
    Returns a dictionary of regression models for Performance Rating prediction.
    """
    models = {
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    try:
        import xgboost as xgb
        models['XGBoost'] = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    except ImportError:
        pass

    try:
        import lightgbm as lgb
        models['LightGBM'] = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    except ImportError:
        pass

    return models

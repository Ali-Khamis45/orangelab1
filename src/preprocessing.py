"""
Preprocessing module for HR analytics data pipeline.
Handles date parsing, feature engineering (Age, Tenure), target construction,
and categorical dummy encoding.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses date columns into standard Pandas datetime objects.
    """
    df = df.copy()
    date_cols = ['StartDate', 'ExitDate', 'DOB', 'Training Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def engineer_features(df: pd.DataFrame, reference_date_str: str = '2026-07-11') -> pd.DataFrame:
    """
    Engineers Age, Tenure, and Is_Terminated columns.
    Ensures correct datetime formats before performing calculations.
    """
    df = parse_dates(df)
    reference_date = pd.to_datetime(reference_date_str)

    # Age calculation in years
    if 'DOB' in df.columns:
        df['Age'] = (reference_date - df['DOB']).dt.days / 365.25

    # Tenure calculation in years
    if 'StartDate' in df.columns and 'ExitDate' in df.columns:
        df['Tenure'] = np.where(
            df['ExitDate'].isna(),
            (reference_date - df['StartDate']).dt.days / 365.25,
            (df['ExitDate'] - df['StartDate']).dt.days / 365.25
        )

    # Binary Attrition Target creation
    if 'EmployeeStatus' in df.columns:
        df['Is_Terminated'] = np.where(
            df['EmployeeStatus'].str.contains('Terminated', na=False), 1, 0
        )

    return df


def get_attrition_features(
    df: pd.DataFrame, 
    drop_first: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Extracts features and target for the Employee Attrition classification task.
    Performs dummy encoding on categorical features and aligns with selected inputs.
    """
    # Ensure engineered columns are present
    df = engineer_features(df)

    num_cols = [
        'Age', 'Tenure', 'Engagement Score', 'Satisfaction Score', 
        'Work-Life Balance Score', 'Training Duration(Days)', 'Training Cost'
    ]
    cat_cols = [
        'GenderCode', 'MaritalDesc', 'RaceDesc', 'DepartmentType', 
        'BusinessUnit', 'EmployeeType', 'PayZone', 
        'EmployeeClassificationType', 'Training Type', 'Training Program Name'
    ]

    X_numeric = df[num_cols]
    X_categorical = pd.get_dummies(df[cat_cols], drop_first=drop_first, dtype=float)

    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = df['Is_Terminated']

    return X, y, num_cols


def get_rating_features(
    df: pd.DataFrame, 
    drop_first: bool = True
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Extracts features and target for the Employee Performance Rating prediction task.
    """
    df = parse_dates(df)

    num_cols = [
        'Engagement Score', 'Satisfaction Score', 'Work-Life Balance Score', 
        'Training Duration(Days)', 'Training Cost'
    ]
    cat_cols = [
        'GenderCode', 'MaritalDesc', 'RaceDesc', 'DepartmentType', 
        'BusinessUnit', 'EmployeeType', 'PayZone', 
        'EmployeeClassificationType', 'Training Type', 'Training Program Name'
    ]

    X_numeric = df[num_cols]
    X_categorical = pd.get_dummies(df[cat_cols], drop_first=drop_first, dtype=float)

    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = df['Current Employee Rating']

    return X, y, num_cols

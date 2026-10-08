"""
Data Preprocessing Module for Ames Housing Dataset.
Handles outlier removal, target transformations, and leakage-free imputation.
"""

from typing import Tuple
import numpy as np
import pandas as pd


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove influential outliers identified by dataset author Dean De Cock.
    Two partial sales (Id 524, 1299) have GrLivArea > 4000 sq ft but sold below $300k.
    Only applied to the training dataset.
    """
    if 'SalePrice' in df.columns:
        outlier_mask = (df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)
        cleaned_df = df[~outlier_mask].copy().reset_index(drop=True)
        return cleaned_df
    return df.copy()


def transform_target(y: pd.Series) -> pd.Series:
    """
    Apply log1p transformation to stabilize variance and directly optimize RMSLE.
    """
    return np.log1p(y)


def inverse_transform_target(y_log: np.ndarray) -> np.ndarray:
    """
    Invert predictions from log space back to raw dollar values.
    """
    return np.expm1(y_log)


def impute_missing_values(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values across train and test sets without data leakage.
    
    1. Structural Categorical NAs (amenity absent) -> 'None'
    2. Structural Numerical NAs (amenity absent) -> 0
    3. Stochastic LotFrontage -> Median of house's Neighborhood (learned on train)
    4. Stochastic Categorical NAs -> Mode (learned on train)
    """
    train = train_df.copy()
    test = test_df.copy()

    # 1. Structural Categoricals: NA means absent amenity
    structural_cats = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in structural_cats:
        train[col] = train[col].fillna('None')
        test[col] = test[col].fillna('None')

    # 2. Structural Numerics: NA means 0 sq ft or 0 count
    structural_nums = [
        'GarageYrBlt', 'GarageCars', 'GarageArea',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in structural_nums:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

    # 3. LotFrontage: Impute by Neighborhood median learned strictly on train
    neighborhood_medians = train.groupby('Neighborhood')['LotFrontage'].median()
    global_median = train['LotFrontage'].median()

    train['LotFrontage'] = train['LotFrontage'].fillna(train['Neighborhood'].map(neighborhood_medians)).fillna(global_median)
    test['LotFrontage'] = test['LotFrontage'].fillna(test['Neighborhood'].map(neighborhood_medians)).fillna(global_median)

    # 4. Mode Imputations: categorical columns with minor missingness in test/train
    mode_cols = [
        'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 
        'KitchenQual', 'SaleType', 'Functional', 'Electrical'
    ]
    for col in mode_cols:
        train_mode = train[col].mode()[0]
        train[col] = train[col].fillna(train_mode)
        test[col] = test[col].fillna(train_mode)

    return train, test

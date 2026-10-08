"""
Feature Engineering Module for Ames Housing Dataset.
Constructs domain-specific composite metrics, encodes ordinals, and one-hot encodes nominals.
"""

from typing import Tuple
import pandas as pd
import numpy as np


def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct high-signal engineered features based on real estate domain principles:
    - TotalSF: Combined livable area across basement, 1st, and 2nd floors
    - TotalBath: Fractional sum of above-ground and basement bathrooms
    - TotalPorchSF: Aggregated exterior leisure area
    - HouseAge / RemodAge: Effective age at the time of sale
    - Interaction and binary amenity flags
    """
    data = df.copy()

    # 1. Square footage aggregations
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
    data['TotalPorchSF'] = data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']

    # 2. Bathroom aggregations
    data['TotalBath'] = (
        data['FullBath'] + 
        (0.5 * data['HalfBath']) + 
        data['BsmtFullBath'] + 
        (0.5 * data['BsmtHalfBath'])
    )

    # 3. Temporal and age calculations
    data['HouseAge'] = (data['YrSold'] - data['YearBuilt']).clip(lower=0)
    data['RemodAge'] = (data['YrSold'] - data['YearRemodAdd']).clip(lower=0)
    data['IsRemodeled'] = (data['YearRemodAdd'] != data['YearBuilt']).astype(int)

    # 4. Interaction metrics
    data['OverallGrade'] = data['OverallQual'] * data['OverallCond']

    # 5. Binary amenity indicators
    data['HasPool'] = (data['PoolArea'] > 0).astype(int)
    data['Has2ndFloor'] = (data['2ndFlrSF'] > 0).astype(int)
    data['HasGarage'] = (data['GarageArea'] > 0).astype(int)
    data['HasBsmt'] = (data['TotalBsmtSF'] > 0).astype(int)
    data['HasFireplace'] = (data['Fireplaces'] > 0).astype(int)

    return data


def encode_ordinal_features(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map categorical features with inherent natural rank to numeric integers.
    """
    train = train_df.copy()
    test = test_df.copy()

    # Standard 5-tier quality/condition map
    qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    qual_cols = [
        'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 
        'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 
        'GarageCond', 'PoolQC'
    ]
    for col in qual_cols:
        train[col] = train[col].map(qual_map).fillna(0).astype(int)
        test[col] = test[col].map(qual_map).fillna(0).astype(int)

    # Basement exposure
    bsmt_exp_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    train['BsmtExposure'] = train['BsmtExposure'].map(bsmt_exp_map).fillna(0).astype(int)
    test['BsmtExposure'] = test['BsmtExposure'].map(bsmt_exp_map).fillna(0).astype(int)

    # Basement finish ratings
    bsmt_fin_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    for col in ['BsmtFinType1', 'BsmtFinType2']:
        train[col] = train[col].map(bsmt_fin_map).fillna(0).astype(int)
        test[col] = test[col].map(bsmt_fin_map).fillna(0).astype(int)

    # Garage finish
    garage_fin_map = {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    train['GarageFinish'] = train['GarageFinish'].map(garage_fin_map).fillna(0).astype(int)
    test['GarageFinish'] = test['GarageFinish'].map(garage_fin_map).fillna(0).astype(int)

    # Central Air and Paved Drive
    train['CentralAir'] = (train['CentralAir'] == 'Y').astype(int)
    test['CentralAir'] = (test['CentralAir'] == 'Y').astype(int)

    paved_map = {'N': 0, 'P': 1, 'Y': 2}
    train['PavedDrive'] = train['PavedDrive'].map(paved_map).fillna(0).astype(int)
    test['PavedDrive'] = test['PavedDrive'].map(paved_map).fillna(0).astype(int)

    # Functional
    func_map = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}
    train['Functional'] = train['Functional'].map(func_map).fillna(8).astype(int)
    test['Functional'] = test['Functional'].map(func_map).fillna(8).astype(int)

    # Fence
    fence_map = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
    train['Fence'] = train['Fence'].map(fence_map).fillna(0).astype(int)
    test['Fence'] = test['Fence'].map(fence_map).fillna(0).astype(int)

    return train, test


def encode_nominal_features(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode remaining nominal categorical features and enforce identical column schema.
    """
    train = pd.get_dummies(train_df, drop_first=True)
    test = pd.get_dummies(test_df, drop_first=True)

    # Align columns: ensure test has all columns present in train, filling absent dummies with 0
    test = test.reindex(columns=train.columns, fill_value=0)

    return train, test

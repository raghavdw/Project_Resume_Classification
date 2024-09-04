from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class labelEncoder(BaseEstimator, TransformerMixin):
    """encode categorical variables"""
    def __init__(self, column_name="Category"):
        self.column_name = column_name
        self.le = LabelEncoder()

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        
        self.le.fit(X[self.column_name])
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        
        X_copy = X.copy()  # Create a copy to avoid modifying the original DataFrame
        X_copy["label"] = self.le.transform(X_copy[self.column_name])
        return X_copy
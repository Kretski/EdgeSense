import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor

class AutoFeatureSelector:
    """Автоматичен избор на фичъри с VarianceThreshold + RandomForest важност"""

    def __init__(self, variance_thresh=0.0001):
        self.var_filter = VarianceThreshold(threshold=variance_thresh)
        self.model = RandomForestRegressor(n_estimators=50)
        self.selected_idx = None

    def fit(self, X, y):
        X_var = self.var_filter.fit_transform(X)
        self.model.fit(X_var, y)
        importances = self.model.feature_importances_
        self.selected_idx = np.where(importances > np.mean(importances))[0]
        return self

    def transform(self, X):
        X_var = self.var_filter.transform(X)
        if self.selected_idx is not None and len(self.selected_idx) > 0:
            return X_var[:, self.selected_idx]
        return X_var

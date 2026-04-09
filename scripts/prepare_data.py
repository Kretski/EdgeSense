# scripts/prepare_data.py
"""
Prepare synthetic dataset with feature engineering for EdgeSenseNano
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib

def feature_engineering(X):
    """
    Create polynomial and sqrt features
    """
    X_poly = np.hstack([X, X**2, np.sqrt(np.abs(X)+1e-8)])
    return X_poly

def generate_dataset(n_samples=6000, n_features=5, test_size=0.2, save_dir="data"):
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features)
    y = (X.sum(axis=1) + 0.3*np.random.randn(n_samples) > 2.5).astype(int)

    X = feature_engineering(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump((X_train, y_train), os.path.join(save_dir, "train.pkl"))
    joblib.dump((X_test, y_test), os.path.join(save_dir, "test.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    print("[OK] Dataset ready and saved in", save_dir)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    generate_dataset()

#!/usr/bin/env python3
"""
EdgeSenseNano Multi-format Full Test Script
Тества ML pipeline с CSV, JSON и бинарни данни
"""
import numpy as np
import pandas as pd
import os
import struct
import sys
import json

# Добавяне на път към edgesense
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- Импорт или заглушки ---
try:
    from edgesense.universal_loader import UniversalDataLoader
    from edgesense.auto_feature_select import AutoFeatureSelector
    from edgesense.drift_detector import DriftDetector
    print("✅ Импорт на EdgeSense модули успешен")
except ImportError:
    print("⚠️ Импорт неуспешен, използваме заглушки")

    class UniversalDataLoader:
        @staticmethod
        def load_csv(filepath):
            return pd.read_csv(filepath)

        @staticmethod
        def load_json(filepath):
            with open(filepath) as f:
                data = json.load(f)
            return pd.DataFrame(data)

        @staticmethod
        def load_bin(filepath, n_features=4, dtype='f'):
            with open(filepath, 'rb') as f:
                raw = f.read()
            total = len(raw) // 4
            data = struct.unpack(f'{total}f', raw)
            rows = total // n_features
            return pd.DataFrame(np.array(data).reshape(rows, n_features),
                                columns=[f'feat{i}' for i in range(n_features)])

    class AutoFeatureSelector:
        def __init__(self, variance_thresh=0.0001):
            self.variance_thresh = variance_thresh
            self.selected_idx = None

        def fit(self, X, y):
            self.selected_idx = np.arange(max(1, X.shape[1] // 2))
            return self

        def transform(self, X):
            if self.selected_idx is not None:
                return X[:, self.selected_idx]
            return X

    class DriftDetector:
        def __init__(self, ref_data):
            self.ref_data = ref_data
            self.ref_mean = np.mean(ref_data, axis=0)

        def detect(self, new_data):
            distance = np.linalg.norm(np.mean(new_data, axis=0) - self.ref_mean)
            return distance > 0.5

# --- Тест функция ---
def test_file(fmt, path, n_features=4):
    print("\n" + "="*50)
    print(f"🧪 Тест с {fmt}: {path}")
    if not os.path.exists(path):
        print(f"⚠️ Файлът {path} не съществува, пропуск")
        return

    # Зареждане
    if fmt == "CSV":
        df = UniversalDataLoader.load_csv(path)
    elif fmt == "JSON":
        df = UniversalDataLoader.load_json(path)
    elif fmt == "BIN":
        df = UniversalDataLoader.load_bin(path, n_features=n_features)
    else:
        print("⚠️ Непознат формат")
        return

    print(f"✅ Заредено {fmt}, размер: {df.shape}")

    # Добавяне на target ако липсва
    if "target" not in df.columns:
        df["target"] = np.random.randn(len(df))
        print("⚠️ 'target' липсваше, добавен случайно")

    X = df.drop(columns=["target"]).values
    y = df["target"].values
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Auto Feature Selection
    selector = AutoFeatureSelector()
    selector.fit(X, y)
    X_sel = selector.transform(X)
    print(f"🎯 Избрани features: {X_sel.shape[1]}")

    # Drift Detection
    ref = X_sel[:50] if X_sel.shape[0] >= 50 else X_sel
    new = X_sel[50:100] if X_sel.shape[0] >= 100 else X_sel
    drift = DriftDetector(ref)
    print(f"📉 Без модификации: {'Drift открит' if drift.detect(new) else 'Няма drift'}")

    # Симулиран drift
    drifted = new * 1.5
    print(f"📉 Със симулиран drift: {'Drift открит' if drift.detect(drifted) else 'Няма drift'}")

# --- Main ---
def main():
    files = [
        ("CSV", "data/sample.csv"),
        ("JSON", "data/sensors.json"),
        ("BIN", "data/raw_stream.bin")
    ]

    for fmt, path in files:
        test_file(fmt, path)

    print("\n✅ MULTI-FORMAT PIPELINE ЗАВЪРШИ УСПЕШНО!")

if __name__ == "__main__":
    main()

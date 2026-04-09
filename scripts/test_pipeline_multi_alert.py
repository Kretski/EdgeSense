#!/usr/bin/env python3
"""
EdgeSenseNano Full Pipeline с Critical Drift Alerts
Тества CSV, JSON и BIN данни
"""
import os
import sys
import json
import struct
import numpy as np
import pandas as pd

# Добавяне на път към edgesense
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from edgesense.auto_feature_select import AutoFeatureSelector
    print("✅ Импорт на EdgeSense модули успешен")
except ImportError:
    print("⚠️ Импорт неуспешен, използваме заглушки")

    class AutoFeatureSelector:
        def __init__(self, variance_thresh=0.0001):
            self.variance_thresh = variance_thresh
            self.selected_idx = None

        def fit(self, X, y):
            self.selected_idx = np.arange(max(1, X.shape[1] // 2))
            return self

        def transform(self, X):
            return X[:, self.selected_idx]

# --- DriftDetector с critical_threshold ---
class DriftDetector:
    def __init__(self, ref_data, critical_threshold=0.5):
        self.ref_data = ref_data
        self.ref_mean = np.mean(ref_data, axis=0)
        self.critical_threshold = critical_threshold

    def detect(self, new_data):
        new_mean = np.mean(new_data, axis=0)
        distance = np.linalg.norm(new_mean - self.ref_mean)
        return distance > self.critical_threshold

# --- UniversalDataLoader заглушка ---
class UniversalDataLoader:
    @staticmethod
    def load_csv(filepath):
        return pd.read_csv(filepath)

    @staticmethod
    def load_json(filepath):
        with open(filepath) as f:
            data = json.load(f)
        # преобразуваме в DataFrame
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

# --- Main pipeline ---
def main():
    print("="*60)
    print("🧪 EDGESENSE NANO FULL PIPELINE (CRITICAL DRIFT ALERTS)")
    print("="*60)

    files = [
        ("CSV", "data/sample.csv"),
        ("JSON", "data/sensors.json"),
        ("BIN", "data/raw_stream.bin")
    ]

    for fmt, path in files:
        print("="*60)
        print(f"🧪 Тест с {fmt}: {path}")

        if not os.path.exists(path):
            print(f"⚠️ Файлът не съществува, генерирам тестови данни")
            np.random.seed(42)
            df = pd.DataFrame(np.random.randn(50, 4),
                              columns=["temperature", "humidity", "pressure", "status"])
        else:
            try:
                if fmt == "CSV":
                    df = UniversalDataLoader.load_csv(path)
                elif fmt == "JSON":
                    df = UniversalDataLoader.load_json(path)
                elif fmt == "BIN":
                    df = UniversalDataLoader.load_bin(path)
            except Exception as e:
                print(f"⚠️ Грешка при зареждане: {e}")
                continue

        if "target" not in df.columns:
            df["target"] = np.random.randn(len(df))
            print(f"⚠️ 'target' липсваше, добавен случайно")

        X = df.drop(columns=["target"]).values
        y = df["target"].values
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        # --- Feature Selection ---
        selector = AutoFeatureSelector()
        selector.fit(X, y)
        X_sel = selector.transform(X)
        print(f"🎯 Избрани features: {X_sel.shape[1]}")

        # --- Drift Detection ---
        ref_data = X_sel[:25] if X_sel.shape[0] >= 25 else X_sel
        new_data = X_sel[25:50] if X_sel.shape[0] >= 50 else X_sel

        detector = DriftDetector(ref_data, critical_threshold=0.5)
        drift_flag = detector.detect(new_data)
        print(f"📉 Без модификации: {'Drift открит' if drift_flag else 'Няма drift'}")

        # Симулиран drift
        drifted_data = new_data * 1.5
        drift_flag_sim = detector.detect(drifted_data)
        print(f"📉 Със симулиран drift: {'Drift открит' if drift_flag_sim else 'Няма drift'}")

if __name__ == "__main__":
    main()

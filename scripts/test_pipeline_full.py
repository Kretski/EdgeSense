#!/usr/bin/env python3
"""
EdgeSenseNano Full Multi-format Pipeline
- Зареждане: CSV, JSON, BIN
- Feature Selection
- Drift Detection
- Тестов inference
- Генериране на C код за Edge
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
            # вземаме половината фичъри за тест
            self.selected_idx = np.arange(X.shape[1] // 2)
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
            new_mean = np.mean(new_data, axis=0)
            distance = np.linalg.norm(new_mean - self.ref_mean)
            return distance > 0.5

def load_and_prepare(filepath, fmt):
    if fmt == "CSV":
        df = UniversalDataLoader.load_csv(filepath)
    elif fmt == "JSON":
        df = UniversalDataLoader.load_json(filepath)
    elif fmt == "BIN":
        df = UniversalDataLoader.load_bin(filepath)
    else:
        raise ValueError(f"Unknown format {fmt}")
    
    if "target" not in df.columns:
        df["target"] = np.random.randn(len(df))
        print(f"⚠️ 'target' липсваше в {filepath}, добавен случайно")
    
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    return df, X, y

def main():
    print("="*60)
    print("🧪 EDGESENSE NANO FULL PIPELINE")
    print("="*60)

    files = [
        ("CSV", "data/sample.csv"),
        ("JSON", "data/sensors.json"),
        ("BIN", "data/raw_stream.bin")
    ]

    for fmt, path in files:
        print(f"\n==================================================")
        print(f"🧪 Тест с {fmt}: {path}")

        if not os.path.exists(path):
            print(f"⚠️ Файлът не съществува: {path}")
            continue

        try:
            df, X, y = load_and_prepare(path, fmt)
            print(f"✅ Заредено {fmt}, размер: {df.shape}")
            print(f"X shape: {X.shape}, y shape: {y.shape}")

            # Feature selection
            selector = AutoFeatureSelector()
            selector.fit(X, y)
            X_sel = selector.transform(X)
            print(f"🎯 Избрани features: {X_sel.shape[1]}")

            # Drift detection
            ref_data = X_sel[:50] if X_sel.shape[0] >= 50 else X_sel
            new_data = X_sel[50:100] if X_sel.shape[0] >= 100 else X_sel
            drift_detector = DriftDetector(ref_data)
            print(f"📉 Без модификации: {'Drift открит' if drift_detector.detect(new_data) else 'Няма drift'}")
            drifted_data = new_data * 1.5
            print(f"📉 Със симулиран drift: {'Drift открит' if drift_detector.detect(drifted_data) else 'Няма drift'}")

            # Тест inference (симулиран)
            test_sample = X_sel[:1]
            y_pred = test_sample.sum(axis=1)  # placeholder inference
            print(f"🤖 Тест inference (sum placeholder): {y_pred}")

            # Генериране на C код (линейна регресия placeholder)
            if X_sel.shape[1] >= 1:
                coeffs = np.polyfit(X_sel[:, 0], y, 1)
                os.makedirs("outputs", exist_ok=True)
                c_code = f"""
// EdgeSenseNano Symbolic Model
// Автоматично генериран код
float edge_predict(float feature1) {{
    return {coeffs[0]:.6f} * feature1 + {coeffs[1]:.6f};
}}
"""
                c_path = f"outputs/{fmt.lower()}_model.c"
                with open(c_path, "w") as f:
                    f.write(c_code)
                print(f"💻 C код генериран: {c_path}")
        except Exception as e:
            print(f"⚠️ Грешка при зареждане или обработка на {path}: {e}")

    print("\n✅ FULL MULTI-FORMAT PIPELINE ЗАВЪРШИ УСПЕШНО!")

if __name__ == "__main__":
    main()

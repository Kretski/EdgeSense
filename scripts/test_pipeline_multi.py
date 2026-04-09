#!/usr/bin/env python3
"""
EdgeSenseNano FULL MULTI-FORMAT PIPELINE
Чете CSV, JSON, BIN → Feature Selection → Drift Detection → C код
"""
import os
import json
import struct
import numpy as np
import pandas as pd
import sys

# --- добавяне на път към edgesense ---
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
        def load_csv(path):
            return pd.read_csv(path)

        @staticmethod
        def load_json(path):
            with open(path) as f:
                data = json.load(f)
            return pd.DataFrame(data)

        @staticmethod
        def load_bin(path, n_features=4, dtype='f'):
            with open(path, 'rb') as f:
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
            return X[:, self.selected_idx]

    class DriftDetector:
        def __init__(self, ref_data):
            self.ref_data = ref_data
            self.ref_mean = np.mean(ref_data, axis=0)

        def detect(self, new_data):
            distance = np.linalg.norm(np.mean(new_data, axis=0) - self.ref_mean)
            return distance > 0.5

def ensure_outputs():
    os.makedirs("outputs", exist_ok=True)

def generate_c_code(features, coeffs, out_file):
    code = "// EdgeSenseNano Symbolic Model\n"
    code += "// Автоматично генериран код\n"
    code += "float edge_predict(float feature1) {\n"
    code += f"    return {coeffs[0]:.6f} * feature1 + {coeffs[1]:.6f};\n"
    code += "}\n"
    with open(out_file, "w") as f:
        f.write(code)
    print(f"💻 C код генериран: {out_file}")

def process_file(fmt, path):
    print(f"==================================================")
    print(f"🧪 Тест с {fmt}: {path}")

    if not os.path.exists(path):
        print(f"⚠️ Файлът не съществува: {path}")
        return

    try:
        if fmt == "CSV":
            df = UniversalDataLoader.load_csv(path)
        elif fmt == "JSON":
            df = UniversalDataLoader.load_json(path)
        elif fmt == "BIN":
            df = UniversalDataLoader.load_bin(path)
    except Exception as e:
        print(f"⚠️ Грешка при зареждане или обработка на {path}: {e}")
        return

    if "target" not in df.columns:
        df["target"] = np.random.randn(len(df))
        print(f"⚠️ 'target' липсваше в {path}, добавен случайно")

    X = df.drop(columns=["target"]).values
    y = df["target"].values
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # --- Feature Selection ---
    selector = AutoFeatureSelector()
    selector.fit(X, y)
    X_sel = selector.transform(X)
    print(f"🎯 Избрани features: {X_sel.shape[1]}")

    # --- Drift Detection ---
    ref_data = X_sel[:min(50, len(X_sel))]
    new_data = X_sel[min(50, len(X_sel)):]
    drift_detector = DriftDetector(ref_data)
    print(f"📉 Без модификации: {'Drift открит' if drift_detector.detect(new_data) else 'Няма drift'}")
    drifted_data = new_data * 1.5 if len(new_data) > 0 else X_sel
    print(f"📉 Със симулиран drift: {'Drift открит' if drift_detector.detect(drifted_data) else 'Няма drift'}")

    # --- Generate simple C code (placeholder) ---
    coeffs = [1.0, 0.0]  # просто примерна линейна функция
    generate_c_code(X_sel.shape[1], coeffs, f"outputs/{fmt.lower()}_model.c")

def main():
    print("="*60)
    print("🧪 EDGESENSE NANO FULL PIPELINE")
    print("="*60)

    ensure_outputs()

    files = [
        ("CSV", "data/sample.csv"),
        ("JSON", "data/sensors.json"),
        ("BIN", "data/raw_stream.bin")
    ]

    for fmt, path in files:
        process_file(fmt, path)

    print("\n✅ FULL MULTI-FORMAT PIPELINE ЗАВЪРШИ УСПЕШНО!")

if __name__ == "__main__":
    main()

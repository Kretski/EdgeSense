#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import os

# ===========================
# Конфигурация
# ===========================
DATA_PATH = r"C:\Users\KR\EdgeSenseNano\intelligent_indoor_environment_dataset.csv"
TARGET_COL = None  # Ако не знаеш, ще се избере след проверка
IGNORE_COLS = ["ID", "timestamp"]  # колони за игнориране

# ===========================
# Зареждане на данните
# ===========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Файлът не съществува: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Показване на всички колони
print("Колони в dataset-а:")
for i, col in enumerate(df.columns.tolist()):
    print(f"{i}: {col}")

# Ако няма зададен target, пита потребителя
if TARGET_COL is None:
    target_idx = input("\nВъведи индекса на целевата колона (target): ")
    try:
        TARGET_COL = df.columns[int(target_idx)]
    except:
        raise ValueError("Невалиден индекс на целевата колона.")

print(f"\nИзбрана целева колона: {TARGET_COL}")

# Подготовка на X и y
feature_cols = [c for c in df.columns if c != TARGET_COL and c not in IGNORE_COLS]
X = df[feature_cols].values
y = df[TARGET_COL].values

print(f"\nБрой фичъри: {len(feature_cols)}")
print("Пример от X:", X[:5])
print("Пример от y:", y[:5])

# Запазване в numpy файлове (по избор)
import numpy as np
np.save("X_processed.npy", X)
np.save("y_processed.npy", y)
print("\n✅ Данните са подготвени и записани като X_processed.npy и y_processed.npy")

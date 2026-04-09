import pandas as pd
import json
import struct
import numpy as np
import os

class UniversalDataLoader:
    """Универсален loader за CSV, JSON и бинарни данни"""

    @staticmethod
    def load_csv(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файлът {filepath} не съществува")
        return pd.read_csv(filepath)

    @staticmethod
    def load_json(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файлът {filepath} не съществува")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Ако е списък от dict, нормализираме
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # dict с колони
            return pd.DataFrame({k: v for k, v in data.items()})
        else:
            raise ValueError("JSON форматът не е поддържан")

    @staticmethod
    def load_bin(filepath, n_features=4, dtype='f'):
        """Чете бинарен поток с float32 (или друг dtype)"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файлът {filepath} не съществува")
        with open(filepath, "rb") as f:
            raw = f.read()
        total = len(raw) // struct.calcsize(dtype)
        data = struct.unpack(f"{total}{dtype}", raw)
        rows = total // n_features
        return pd.DataFrame(np.array(data).reshape(rows, n_features),
                            columns=[f'feat{i}' for i in range(n_features)])

import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "sample.csv")
BIN_PATH = os.path.join(DATA_DIR, "raw_stream.bin")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ Липсва {CSV_PATH}")

    # Зареждаме CSV
    df = pd.read_csv(CSV_PATH)
    print(f"✅ CSV зареден: {CSV_PATH}, размер: {df.shape}")

    # Махаме target ако има (pipeline си го добавя сам)
    if "target" in df.columns:
        df = df.drop(columns=["target"])

    # Конвертираме към float32 numpy масив
    data = df.astype("float32").values

    # Записваме като суров бинарен поток
    data.tofile(BIN_PATH)

    print(f"✅ BIN файл създаден: {BIN_PATH}, редове: {data.shape[0]}, фичъри: {data.shape[1]}")

if __name__ == "__main__":
    main()

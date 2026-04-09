# scripts/check_files.py
import os
import time

files = [
    "../data/appliances_anomaly.npz",
    "../models/anomaly_model.pth",
    "../outputs/anomaly_model.onnx",
    "../outputs/confusion_matrix.png"
]

print("Проверка на файлове (обновява се на всеки 5 сек)...")
print("-" * 50)

while True:
    for f in files:
        status = "ОК" if os.path.exists(f) else "ЛИПСВА"
        print(f"{os.path.basename(f):<30} → {status}")
    print("-" * 50)
    time.sleep(5)
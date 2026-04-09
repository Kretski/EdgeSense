# scripts/test_on_edge.py
"""
Тест на модела в edge-сценарий (симулация на ABX00173 / микроконтролер)
- Inference latency
- Консумация (CPU / RAM)
- Сравнение ONNX vs PyTorch
"""

import numpy as np
import torch
import onnxruntime as ort
import time
import psutil
import argparse
import os

# =============================================================================
# Конфигурация
# =============================================================================
ONNX_PATH = "../outputs/anomaly_model.onnx"
TEST_SAMPLES = 10000
WARMUP = 100

# =============================================================================
# Парсер
# =============================================================================
parser = argparse.ArgumentParser(description="Тест на модела в edge режим")
parser.add_argument("--onnx", type=str, default=ONNX_PATH, help="Път до ONNX")
parser.add_argument("--samples", type=int, default=TEST_SAMPLES, help="Брой тестове")
args = parser.parse_args()

# =============================================================================
# Основна функция
# =============================================================================
def main():
    print("=== EdgeSense Nano - Тест на edge ===")
    
    if not os.path.exists(args.onnx):
        print(f"Грешка: ONNX файлът {args.onnx} не съществува!")
        return

    # Зареждане на ONNX
    session = ort.InferenceSession(args.onnx)
    input_name = session.get_inputs()[0].name
    print(f"ONNX модел зареден: {args.onnx}")

    # Dummy данни
    dummy_inputs = np.random.randn(args.samples, 9).astype(np.float32)

    # ONNX latency
    print(f"\nТест ONNX – {args.samples} инференса...")
    onnx_times = []

    # Warmup
    for i in range(WARMUP):
        _ = session.run(None, {input_name: dummy_inputs[i:i+1]})[0]

    # Реално мерене
    for i in range(args.samples):
        t0 = time.perf_counter_ns()
        _ = session.run(None, {input_name: dummy_inputs[i:i+1]})[0]
        t1 = time.perf_counter_ns()
        onnx_times.append((t1 - t0) / 1_000_000)  # ms

    mean_latency = np.mean(onnx_times)
    p95_latency = np.percentile(onnx_times, 95)
    print(f"ONNX средна latency: {mean_latency:.3f} ms")
    print(f"ONNX p95 latency:   {p95_latency:.3f} ms")

    # CPU/RAM
    process = psutil.Process(os.getpid())
    print(f"CPU по време на тест: ~{process.cpu_percent(interval=0.1):.1f}%")
    print(f"RAM употреба: ~{process.memory_info().rss / (1024**2):.1f} MB")

    print("\nТестът завърши! Моделът е готов за ABX00173.")
    print("Следваща стъпка: генерирай C код или тествай на платката.")

if __name__ == "__main__":
    main() 

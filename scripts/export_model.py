# scripts/test_on_edge.py
"""
Тест на модела в edge-сценарий (симулация на ABX00173 / микроконтролер)
- Inference latency
- Консумация (CPU / RAM)
- Сравнение ONNX vs PyTorch
- Псевдокод за embedded inference
"""

import numpy as np
import torch
import onnxruntime as ort
import time
import psutil
import argparse
import os
from src.edgesense_core import TinyAnomalyNN

# =============================================================================
# Конфигурация
# =============================================================================
ONNX_PATH = "../outputs/anomaly_model.onnx"
MODEL_PATH = "../models/anomaly_model.pth"
INPUT_DIM = 9
TEST_SAMPLES = 10000          # колко инференса да симулираме
WARMUP = 100                  # загряване преди мерене

# =============================================================================
# Парсер
# =============================================================================
parser = argparse.ArgumentParser(description="Тест на модела в edge режим")
parser.add_argument("--onnx", type=str, default=ONNX_PATH, help="Път до ONNX модела")
parser.add_argument("--samples", type=int, default=TEST_SAMPLES, help="Брой тестове")
args = parser.parse_args()

# =============================================================================
# 1. Зареждане на моделите
# =============================================================================
print("=== EdgeSense Nano - Тест на edge ===")

# PyTorch модел (за сравнение)
device = torch.device("cpu")
pt_model = TinyAnomalyNN(INPUT_DIM)
pt_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
pt_model.eval()

# ONNX модел
onnx_session = ort.InferenceSession(args.onnx)
input_name = onnx_session.get_inputs()[0].name

print(f"ONNX модел зареден: {args.onnx}")

# =============================================================================
# 2. Подготовка на dummy данни
# =============================================================================
np.random.seed(42)
dummy_inputs = np.random.randn(args.samples, INPUT_DIM).astype(np.float32)

# =============================================================================
# 3. ONNX Inference + latency
# =============================================================================
print(f"\nТест ONNX – {args.samples} инференса...")

onnx_times = []
process = psutil.Process(os.getpid())
cpu_start = process.cpu_percent(interval=None)
ram_start = process.memory_info().rss / (1024 ** 2)  # MB

# Warmup
for i in range(WARMUP):
    _ = onnx_session.run(None, {input_name: dummy_inputs[i:i+1]})[0]

# Реално мерене
for i in range(args.samples):
    t0 = time.perf_counter_ns()
    _ = onnx_session.run(None, {input_name: dummy_inputs[i:i+1]})[0]
    t1 = time.perf_counter_ns()
    onnx_times.append((t1 - t0) / 1_000_000)  # ms

onnx_latency_mean = np.mean(onnx_times)
onnx_latency_p95 = np.percentile(onnx_times, 95)

cpu_end = process.cpu_percent(interval=None)
ram_end = process.memory_info().rss / (1024 ** 2)

print(f"ONNX средна latency: {onnx_latency_mean:.3f} ms")
print(f"ONNX p95 latency:   {onnx_latency_p95:.3f} ms")
print(f"CPU по време на тест: ~{cpu_end:.1f}%")
print(f"RAM по време на тест: ~{ram_end - ram_start:.1f} MB допълнително")

# =============================================================================
# 4. PyTorch сравнение (за референтна стойност)
# =============================================================================
print(f"\nСравнение с PyTorch (за референция)...")
pt_times = []

for i in range(args.samples):
    x_t = torch.from_numpy(dummy_inputs[i:i+1]).float()
    t0 = time.perf_counter_ns()
    with torch.no_grad():
        _ = pt_model(x_t)
    t1 = time.perf_counter_ns()
    pt_times.append((t1 - t0) / 1_000_000)

pt_latency_mean = np.mean(pt_times)
print(f"PyTorch средна latency: {pt_latency_mean:.3f} ms")
print(f"ONNX е {pt_latency_mean / onnx_latency_mean:.2f}× по-бърз от PyTorch")

# =============================================================================
# 5. Псевдокод / C шаблон за ABX00173 (Cortex-M33F)
# =============================================================================
print("\n=== Псевдокод за Cortex-M33F inference ===")
print("/*")
print(" * inference.c - пример за ABX00173 / Cortex-M33F")
print(" * Вход: float input[9]")
print(" * Изход: float probability [0..1]")
print(" */")
print("")
print("#include <math.h>")
print("")
print("float predict_anomaly(float input[9]) {")
print("    // Тук трябва да се вмъкнат реалните тегла от ONNX")
print("    // Примерна структура (замени с реални стойности)")
print("    float h1[32] = {0};")
print("    for (int i = 0; i < 32; i++) {")
print("        h1[i] = bias1[i];")
print("        for (int j = 0; j < 9; j++) h1[i] += input[j] * w1[j][i];")
print("        h1[i] = h1[i] > 0 ? h1[i] : 0;  // ReLU")
print("    }")
print("    // ... същото за следващите слоеве")
print("    float output = sigmoid(final_linear);")
print("    return output;")
print("}")
print("")
print("Забележка: За реален код използвай onnx2c, microTVM или ръчно конвертирай теглата.")

print("\nТестът завърши!")
print("ONNX моделът е готов за тестване на ABX00173 или друг MCU.")
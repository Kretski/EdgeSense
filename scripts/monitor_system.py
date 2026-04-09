# scripts/monitor_system.py
"""
Прост монитор за CPU, RAM и диск по време на обучение
Пуска се в отделен CMD прозорец
"""

import psutil
import time
import os

print("=== Системен монитор (Ctrl+C за спиране) ===")
print("Интервал: 3 секунди")
print("-" * 50)

try:
    while True:
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"CPU: {cpu:5.1f}% | "
              f"RAM: {ram.percent:5.1f}% used ({ram.used / (1024**3):.1f}/{ram.total / (1024**3):.1f} GB) | "
              f"Disk: {disk.percent:5.1f}% used")
        
        time.sleep(3)
except KeyboardInterrupt:
    print("\nМониторът спря.")
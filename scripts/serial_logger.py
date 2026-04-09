#!/usr/bin/env python3
"""
EdgeSense Nano: COM Port Data Logger
Слуша COM порт и записва данните директно в data/pool_record.csv
"""
import serial
import csv
import os
import sys

# Настройки на порта (промени COM7, ако платката ти е на друг порт)
COM_PORT = 'COM7'
BAUD_RATE = 115200

# Създаваме папката data, ако липсва
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(data_dir, exist_ok=True)
csv_file = os.path.join(data_dir, 'pool_record.csv')

try:
    # Задължително трябва да имаш инсталиран pyserial: pip install pyserial
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"✅ Успешно свързване към {COM_PORT} на {BAUD_RATE} baud.")
except Exception as e:
    print(f"❌ Грешка при свързване с {COM_PORT}: {e}")
    print("Увери се, че платката е включена и серийният монитор на Arduino/IDE е ЗАТВОРЕН!")
    sys.exit(1)

print(f"🔴 Започва запис в {csv_file}")
print("Движи сензорите/лодката! Натисни CTRL+C, за да спреш записа.")

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Записваме заглавния ред (Header), който AZURO очаква
    writer.writerow(['error', 'gyro_y', 'thrust'])
    
    records = 0
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                # Очакваме данни във формат: "0.12, 1.45, 0.50"
                if ',' in line:
                    data_points = [x.strip() for x in line.split(',')]
                    if len(data_points) == 3:
                        writer.writerow(data_points)
                        records += 1
                        if records % 20 == 0:
                            print(f"Записани {records} реда...")
    except KeyboardInterrupt:
        print(f"\n⏹️ Записът е спрян от потребителя. Общо редове: {records}")
        print(f"✅ Файлът е готов: {csv_file}")
    finally:
        ser.close()
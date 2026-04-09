# scripts/train_anomaly.py
"""
Обучение с symbolic distillation
Използва EdgeSenseAnomalyFull и symbolic_from_nn
ОПРАВЕНО: Работи от всяка директория!
"""

import sys
import os

# Добавяме src към пътя
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import sympy as sp
from edgesense_core import EdgeSenseAnomalyFull
from symbolic_from_nn import symbolic_from_nn, sympy_to_c
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ========== ОПРАВЕНА КОНФИГУРАЦИЯ ==========
# Автоматично намира правилния път до данните
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "wake_word_data.npz")

CONFIG = {
    "X_path": DATA_PATH,  # ← Сега пътят е абсолютен!
    "symbolic": True,
    "epochs": 100,
    "save_onnx": True,
}

def find_data_file(filename):
    """Търси файла на различни места"""
    possible_paths = [
        filename,  # текуща директория
        os.path.join(BASE_DIR, "data", filename),  # data/ директория
        os.path.join(BASE_DIR, "scripts", filename),  # scripts/ директория
        os.path.join(os.getcwd(), filename),  # където и да си
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Намерен файл: {path}")
            return path
    
    return None

def train_pipeline(config):
    # Създаване на директории
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("📦 Loading processed data...")
    print(f"   Търсене на: {config['X_path']}")
    
    # Проверка дали файлът съществува
    if not os.path.exists(config["X_path"]):
        print(f"❌ Файлът не съществува: {config['X_path']}")
        
        # Опитай да намериш файла автоматично
        filename = os.path.basename(config["X_path"])
        found_path = find_data_file(filename)
        
        if found_path:
            config["X_path"] = found_path
            print(f"✅ Намерен е файл: {found_path}")
        else:
            # Покажи всички .npz файлове в проекта
            print("\n🔍 Търсене на .npz файлове в проекта:")
            for root, dirs, files in os.walk(BASE_DIR):
                for file in files:
                    if file.endswith('.npz'):
                        print(f"   📄 {os.path.join(root, file)}")
            
            print("\n❌ Моля, провери къде е файлът и актуализирай пътя.")
            return
    
    # Зареждане на данни
    try:
        data = np.load(config["X_path"], allow_pickle=True)
    except Exception as e:
        print(f"❌ Грешка при зареждане: {e}")
        return
    
    # Проверка какви ключове има във файла
    print(f"📊 Налични ключове: {list(data.files)}")
    
    # Обработка на различни формати
    if 'X' in data.files and 'y' in data.files:
        X = data['X']
        y = data['y']
        feature_names = [f"x{i}" for i in range(X.shape[1])]
        print("✅ Използвам X и y")
        
    elif 'X_train' in data.files and 'y_train' in data.files:
        X = data['X_train']
        y = data['y_train']
        feature_names = data['features'] if 'features' in data.files else [f"x{i}" for i in range(X.shape[1])]
        print("✅ Използвам X_train и y_train")
        
    else:
        print("❌ Непознат формат на данните!")
        return
    
    print(f"Shape: {X.shape}, Features: {len(feature_names)}")
    print(f"   Positive: {sum(y)}")
    print(f"   Negative: {len(y) - sum(y)}")
    
    # Разделяне на тренировъчни и тестови (ако няма X_test)
    if 'X_test' not in data.files:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("✅ Автоматично разделяне на данните")
    else:
        X_train = X
        y_train = y
        X_test = data['X_test']
        y_test = data['y_test']
    
    print("🧠 Training neural model...")
    model = EdgeSenseAnomalyFull(X_train.shape[1])
    model.train_model(X_train, y_train, epochs=config.get("epochs", 100))
    
    print("📤 Exporting ONNX...")
    model.export_onnx(X_train.shape[1], save_path=os.path.join(BASE_DIR, "outputs", "wake_word_model.onnx"))
    
    # Тест на модела
    print("\n📊 Тестване на модела...")
    y_pred = model.predict(X_test)
    
    # За бинарна класификация (wake word)
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = np.mean(y_pred_binary == y_test) * 100
    
    mse_test = mean_squared_error(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)
    
    print(f"   ✅ Точност: {accuracy:.2f}%")
    print(f"   ✅ Test MSE: {mse_test:.4f}")
    print(f"   ✅ Test MAE: {mae_test:.4f}")
    
    if config["symbolic"]:
        print("\n🔬 Running symbolic distillation...")
        try:
            eq_str = symbolic_from_nn(model, X_train, feature_names)
            
            # Запазване на уравнението
            eq_path = os.path.join(BASE_DIR, "outputs", "discovered_equation.txt")
            with open(eq_path, "w") as f:
                f.write(eq_str)
            print(f"✨ Equation: {eq_str}")
            
            # C код
            c_code = sympy_to_c(eq_str, feature_names)
            c_path = os.path.join(BASE_DIR, "outputs", "discovered_model.c")
            with open(c_path, "w") as f:
                f.write(c_code)
            print(f"💾 C код запазен в: {c_path}")
            
        except Exception as e:
            print(f"⚠️ Грешка при symbolic distillation: {e}")
            print("   Продължавам без символично уравнение...")
    
    # Запази модела като .pth
    import torch
    model_path = os.path.join(BASE_DIR, "outputs", "wake_word_model.pth")
    torch.save(model.nn_model.state_dict(), model_path)
    print(f"💾 PyTorch модел запазен в: {model_path}")
    
    print("\n🎉 Pipeline complete!")
    print(f"📁 Всички файлове са в: {os.path.join(BASE_DIR, 'outputs')}")

if __name__ == "__main__":
    train_pipeline(CONFIG)
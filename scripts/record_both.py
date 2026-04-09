#!/usr/bin/env python3
"""
record_both.py - Запис на 20 positive + 20 negative примера
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from datetime import datetime
import time

SAMPLE_RATE = 16000
DURATION = 1.0  # секунди

def record_samples(folder, prompt, count=20, examples=None):
    """Записва count на брой примера"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    
    # Виж колко вече има
    existing = list(folder.glob("*.wav"))
    if existing:
        print(f"\n📂 В папката вече има {len(existing)} файла")
        response = input("   Да ги запазим ли? (да/не): ")
        if response.lower() != 'да':
            for f in existing:
                f.unlink()
            print("   🗑️ Старите файлове са изтрити")
    
    print(f"\n📂 Ще записваме в: {folder}")
    print(f"🎤 Ще запишем {count} примера")
    
    if examples:
        print(f"💡 Идеи: {', '.join(examples)}")
    
    for i in range(count):
        print(f"\n--- {i+1}/{count} ---")
        if examples and i < len(examples):
            print(f"💡 Пример: '{examples[i]}'")
        
        input("⏺️ Натисни Enter и говори...")
        
        print("🎤 Записвам", end="", flush=True)
        audio = sd.rec(int(DURATION * SAMPLE_RATE), 
                      samplerate=SAMPLE_RATE,
                      channels=1, dtype='float32')
        
        # Анимация докато записва
        for _ in range(10):
            time.sleep(0.1)
            print(".", end="", flush=True)
        
        sd.wait()
        audio = audio.flatten()
        print(" OK")
        
        # Провери силата на звука
        volume = np.abs(audio).mean()
        if volume < 0.01:
            print("⚠️  Много тихо! Говори по-силно следващия път")
        elif volume > 0.5:
            print("⚠️  Много силно! Говори по-тихо следващия път")
        else:
            print("✅  Добра сила на звука")
        
        # Запази файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = folder / f"sample_{timestamp}.wav"
        sf.write(filename, audio, SAMPLE_RATE)
        print(f"   ✅ Записано: {filename.name}")
    
    return count

def main():
    print("="*70)
    print("🎤 EdgeSenseNano - Запис на тренировъчни данни (20+20)")
    print("="*70)
    
    print("\n📋 Ще запишем:")
    print("   1️⃣ POSITIVE - думата 'стоп' (20 пъти)")
    print("   2️⃣ NEGATIVE - всичко друго (20 пъти)")
    print("\n⚠️  Важно: Говори естествено, с нормална сила")
    
    # POSITIVE ПРИМЕРИ (20 пъти "стоп")
    print("\n" + "="*70)
    print("🔴 ЧАСТ 1: POSITIVE ПРИМЕРИ (20× 'стоп')")
    print("="*70)
    
    pos_examples = ["стоп"] * 20  # 20 пъти една и съща дума
    
    pos_count = record_samples(
        "data/tts_data/tts_positive", 
        "Кажи 'стоп'", 
        20,
        pos_examples
    )
    
    # NEGATIVE ПРИМЕРИ (20 различни неща)
    print("\n" + "="*70)
    print("🔵 ЧАСТ 2: NEGATIVE ПРИМЕРИ (20× друго)")
    print("="*70)
    
    neg_examples = [
        "едно", "две", "три", "четири", "пет",
        "старт", "пауза", "продължи", "край", "стоп"  # Да, "стоп" и в negative? НЕ!
        # ВАЖНО: "стоп" НЕ трябва да е в negative!
    ]
    
    # По-добри negative примери:
    neg_examples = [
        "здравей", "довиждане", "моля", "благодаря", "извинявай",
        "едно", "две", "три", "четири", "пет",
        "котка", "куче", "кола", "къща", "дърво",
        "днес", "утре", "вчера", "сега", "никога",
        "песен", "филм", "книга", "вода", "огън",
        "бързо", "бавно", "високо", "ниско", "далеч"
    ][:20]  # Вземи първите 20
    
    neg_count = record_samples(
        "data/tts_data/tts_negative", 
        "Кажи НЕЩО ДРУГО (не 'стоп')", 
        20,
        neg_examples
    )
    
    # Резюме
    print("\n" + "="*70)
    print("✅ ЗАВЪРШЕНО!")
    print("="*70)
    print(f"   📊 Positive: {pos_count} примера")
    print(f"   📊 Negative: {neg_count} примера")
    print(f"   📁 Данните са в: data/tts_data/")
    
    # Покажи какво има във всяка папка
    print("\n📂 Съдържание:")
    pos_files = list(Path("data/tts_data/tts_positive").glob("*.wav"))
    neg_files = list(Path("data/tts_data/tts_negative").glob("*.wav"))
    print(f"   Positive: {len(pos_files)} файла")
    print(f"   Negative: {len(neg_files)} файла")
    
    if len(pos_files) >= 20 and len(neg_files) >= 20:
        print("\n✅ ИМАШ ДОСТАТЪЧНО ДАННИ!")
        print("\n🚀 Следваща стъпка:")
        print("   python scripts/prepare_wake_word_data.py")
    else:
        print("\n⚠️  Нямаш достатъчно данни!")
        print("   Трябват ти поне 20 positive и 20 negative")
    
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Записът е прекратен")
    except Exception as e:
        print(f"\n❌ Грешка: {e}")
        import traceback
        traceback.print_exc()
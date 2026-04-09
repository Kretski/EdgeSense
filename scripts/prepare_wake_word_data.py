# prepare_wake_word_data.py
"""
Използва съществуващите данни от TinyAutoFUSv3
"""
import os
import numpy as np
import librosa
from pathlib import Path

# Конфигурация (същата като в профила)
SAMPLE_RATE = 16000
N_MELS = 40
TIME_STEPS = 32
REQUIRED_SAMPLES = 16000

def load_audio_files(folder_path, label):
    """Зарежда всички аудио файлове от папка"""
    features = []
    labels = []
    
    # Провери дали папката съществува
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Папката не съществува: {folder_path}")
        return np.array([]), np.array([])
    
    # Намери всички wav файлове
    wav_files = list(folder.glob("*.wav"))
    print(f"   Намерени {len(wav_files)} файла в {folder_path}")
    
    for audio_file in wav_files:
        try:
            # Зареди аудио
            audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
            
            # Направи същата обработка като в оригиналния код
            if len(audio) < REQUIRED_SAMPLES:
                audio = np.pad(audio, (0, REQUIRED_SAMPLES - len(audio)))
            else:
                audio = audio[:REQUIRED_SAMPLES]
            
            # Mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=audio, sr=SAMPLE_RATE,
                n_mels=N_MELS, hop_length=160, n_fft=512
            )
            mel = librosa.power_to_db(mel)
            
            # Нормализиране на времевата ос
            if mel.shape[1] < TIME_STEPS:
                pad = np.zeros((N_MELS, TIME_STEPS - mel.shape[1]))
                mel = np.hstack([mel, pad])
            else:
                mel = mel[:, :TIME_STEPS]
            
            # Транспониране и flatten за EdgeSenseNano
            mel = mel.T.flatten().astype(np.float32)
            
            features.append(mel)
            labels.append(label)
            
        except Exception as e:
            print(f"Грешка при {audio_file}: {e}")
    
    return np.array(features), np.array(labels)

def main():
    print("="*60)
    print("🔊 EdgeSenseNano - Wake Word Data Preparation")
    print("="*60)
    
    # ПЪТИЩА - ОПРАВЕНИ!
    base_dir = Path("data/tts_data")
    pos_dir = base_dir / "tts_positive"
    neg_dir = base_dir / "tts_negative"
    
    print(f"\n📂 Търсене в:")
    print(f"   Positive: {pos_dir.absolute()}")
    print(f"   Negative: {neg_dir.absolute()}")
    
    # Зареди данни
    print("\n📂 Зареждане на positive примери...")
    positive_features, positive_labels = load_audio_files(pos_dir, 1)
    
    print("\n📂 Зареждане на negative примери...")
    negative_features, negative_labels = load_audio_files(neg_dir, 0)
    
    # Провери дали има данни
    if len(positive_features) == 0 and len(negative_features) == 0:
        print("\n❌ НЯМА НАМЕРЕНИ ДАННИ!")
        print("\n📋 Възможни причини:")
        print("   1. Няма записани аудио файлове")
        print("   2. Файловете са в грешна директория")
        print("   3. Файловете не са .wav формат")
        print("\n🔧 Решение:")
        print("   Първо изпълни: python scripts/record_both.py")
        return
    
    # Обедини
    if len(positive_features) > 0 and len(negative_features) > 0:
        X = np.concatenate([positive_features, negative_features])
        y = np.concatenate([positive_labels, negative_labels])
    elif len(positive_features) > 0:
        X = positive_features
        y = positive_labels
    else:
        X = negative_features
        y = negative_labels
    
    print(f"\n📊 РЕЗУЛТАТ:")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Positive: {len(positive_features)}")
    print(f"   Negative: {len(negative_features)}")
    
    # Запази за EdgeSenseNano
    output_path = "data/wake_word_data.npz"
    np.savez(output_path, X=X, y=y)
    
    print(f"\n💾 Данните са запазени в: {output_path}")
    print(f"   Размер: {os.path.getsize(output_path)/1024:.1f} KB")
    
    # Провери дали имаме достатъчно данни
    if len(positive_features) >= 20 and len(negative_features) >= 20:
        print("\n✅ ИМАШ ДОСТАТЪЧНО ДАННИ!")
        print("\n🚀 Следваща стъпка:")
        print("   python scripts/train_anomaly_simple.py --data data/wake_word_data.npz --epochs 100 --save-model")
    else:
        print(f"\n⚠️ Нямаш достатъчно данни! Нужни са 20 positive и 20 negative.")
        print(f"   Имаш: {len(positive_features)} positive, {len(negative_features)} negative")

if __name__ == "__main__":
    main()
import numpy as np
import pickle
import os
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

data = np.load('Data.npz', allow_pickle=True)

all_labels = []
for key in data.files:
    if 'y' in key.lower():
        labels = data[key]
        all_labels.extend(labels)
        print(f"Ключ {key}: {len(labels)} меток")

all_labels = np.array(all_labels)
print(f"Всего меток: {len(all_labels)}")


def enhanced_grouping(planet_name):
    name = str(planet_name).lower()

    if 'kepler' in name:
        if '62' in name:
            return 'Kepler-62'
        elif '186' in name:
            return 'Kepler-186'
        elif '22' in name:
            return 'Kepler-22'
        elif '174' in name:
            return 'Kepler-174'
        elif '155' in name:
            return 'Kepler-155'
        elif '296' in name:
            return 'Kepler-296'
        elif '283' in name:
            return 'Kepler-283'
        else:
            return 'Kepler-Other'
    elif 'gliese' in name:
        if '163' in name:
            return 'Gliese-163'
        elif '12' in name:
            return 'Gliese-12'
        else:
            return 'Gliese-Other'
    elif 'k2' in name:
        if '72' in name:
            return 'K2-72'
        elif '155' in name:
            return 'K2-155'
        elif '288' in name:
            return 'K2-288'
        elif '332' in name:
            return 'K2-332'
        else:
            return 'K2-Other'
    elif 'hd' in name:
        return 'HD'
    elif 'hip' in name:
        return 'HIP'
    elif '55 cancri' in name:
        return '55-Cancri'
    else:
        return 'Unknown'


grouped_labels = np.array([enhanced_grouping(label) for label in all_labels])

unique, counts = np.unique(grouped_labels, return_counts=True)
print("\nРаспределение по группам:")
for g, c in sorted(zip(unique, counts), key=lambda x: x[1], reverse=True):
    print(f"{g:15}: {c:4} ({c / len(grouped_labels) * 100:.1f}%)")

train_x = data['train_x']
valid_x = data['valid_x']

print(f"Train: {train_x.shape}")
print(f"Valid: {valid_x.shape}")


def extract_enhanced_features(audio_batch):
    features = []

    for audio in audio_batch:
        audio = audio.flatten()

        mean = np.mean(audio)
        std = np.std(audio)
        max_val = np.max(np.abs(audio))
        min_val = np.min(audio)
        rms = np.sqrt(np.mean(audio ** 2))

        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        crest_factor = max_val / (rms + 1e-10)
        amplitude_ratio = max_val / (np.abs(min_val) + 1e-10)

        n_segments = 8
        segment_length = len(audio) // n_segments
        segment_energies = []
        segment_means = []

        for i in range(n_segments):
            start = i * segment_length
            end = start + segment_length
            segment = audio[start:end]
            segment_energies.append(np.sum(segment ** 2) / len(segment))
            segment_means.append(np.mean(segment))

        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1 / 16000)

        spectral_centroid = np.sum(freqs[:500] * fft[:500]) / (np.sum(fft[:500]) + 1e-10)

        spectral_bandwidth = np.sqrt(
            np.sum(((freqs[:500] - spectral_centroid) ** 2) * fft[:500]) /
            (np.sum(fft[:500]) + 1e-10)
        )

        cumsum = np.cumsum(fft)
        rolloff_point = 0.95 * cumsum[-1]
        spectral_rolloff = freqs[np.searchsorted(cumsum, rolloff_point)]

        rolloff_85_point = 0.85 * cumsum[-1]
        spectral_rolloff_85 = freqs[np.searchsorted(cumsum, rolloff_85_point)]

        low_freq_energy = np.sum(fft[freqs < 1000])
        high_freq_energy = np.sum(fft[freqs >= 1000]) + 1e-10
        freq_ratio = low_freq_energy / high_freq_energy

        mel_bands = 13
        mel_filters = np.linspace(0, len(fft) - 1, mel_bands + 2).astype(int)
        mfcc_like = []

        for j in range(mel_bands):
            start = mel_filters[j]
            center = mel_filters[j + 1]
            end = mel_filters[j + 2]

            filter_bank = np.zeros(len(fft))
            filter_bank[start:center] = np.linspace(0, 1, center - start)
            filter_bank[center:end] = np.linspace(1, 0, end - center)

            mel_energy = np.sum(fft * filter_bank)
            mfcc_like.append(np.log(mel_energy + 1e-10))

        corr = np.correlate(audio - mean, audio - mean, mode='same')
        corr = corr[len(corr) // 2:]
        corr = corr / (corr[0] + 1e-10)
        autocorr_peaks = np.max(corr[1:10]) if len(corr) > 10 else 0

        feature_vector = [
            mean, std, max_val, min_val, rms,
            zero_crossings, crest_factor, amplitude_ratio,
            spectral_centroid, spectral_bandwidth,
            spectral_rolloff, spectral_rolloff_85,
            freq_ratio, autocorr_peaks
        ]

        feature_vector.extend(segment_energies)
        feature_vector.extend(segment_means)
        feature_vector.extend(mfcc_like)

        features.append(feature_vector)

    return np.array(features)


train_features = extract_enhanced_features(train_x)
print(f"Train признаки: {train_features.shape}")

valid_features = extract_enhanced_features(valid_x)
print(f"Valid признаки: {valid_features.shape}")

n_train = len(train_x)
train_groups = grouped_labels[:n_train]
valid_groups = grouped_labels[n_train:n_train + len(valid_x)]

train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
valid_features = np.nan_to_num(valid_features, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
valid_features_scaled = scaler.transform(valid_features)

label_encoder = LabelEncoder()
label_encoder.fit(train_groups)
y_train = label_encoder.transform(train_groups)
y_valid = label_encoder.transform(valid_groups)

num_classes = len(label_encoder.classes_)
print(f"\nКоличество классов: {num_classes}")
print(f"Классы: {label_encoder.classes_}")


def create_enhanced_model(input_dim, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


input_dim = train_features_scaled.shape[1]
model = create_enhanced_model(input_dim, num_classes)
model.summary()

os.makedirs('enhanced_model', exist_ok=True)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1,
        mode='max'
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1,
        mode='min'
    ),
    keras.callbacks.ModelCheckpoint(
        'enhanced_model/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    ),
    keras.callbacks.CSVLogger('enhanced_model/training_log.csv')
]

history1 = model.fit(
    train_features_scaled, y_train,
    validation_data=(valid_features_scaled, y_valid),
    epochs=40,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\nЭтап 2: Дообучение")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_features_scaled, y_train,
    validation_data=(valid_features_scaled, y_valid),
    epochs=20,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

history = {}
for key in history1.history:
    history[key] = history1.history[key] + history2.history[key]

val_loss, val_accuracy = model.evaluate(valid_features_scaled, y_valid, verbose=0)
print(f"\nФинальная точность на валидации: {val_accuracy:.4f}")

y_pred = model.predict(valid_features_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_pred_proba = np.max(y_pred, axis=1)

print("\nОтчет по классам:")
print(classification_report(y_valid, y_pred_classes, target_names=label_encoder.classes_))

plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_valid, y_pred_classes)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title('Матрица ошибок')
plt.xlabel('Предсказано')
plt.ylabel('Истина')
plt.tight_layout()
plt.savefig('enhanced_model/confusion_matrix.png')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(history['accuracy'], label='Train')
axes[0, 0].plot(history['val_accuracy'], label='Validation')
axes[0, 0].axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Точность модели')
axes[0, 0].set_xlabel('Эпоха')
axes[0, 0].set_ylabel('Точность')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history['loss'], label='Train')
axes[0, 1].plot(history['val_loss'], label='Validation')
axes[0, 1].axvline(x=len(history1.history['loss']), color='r', linestyle='--', alpha=0.5)
axes[0, 1].set_title('Потери модели')
axes[0, 1].set_xlabel('Эпоха')
axes[0, 1].set_ylabel('Потери')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].hist(y_pred_proba, bins=20, edgecolor='black')
axes[1, 0].set_title('Распределение уверенности предсказаний')
axes[1, 0].set_xlabel('Уверенность')
axes[1, 0].set_ylabel('Количество')
axes[1, 0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5)

class_accuracies = []
for i in range(num_classes):
    mask = (y_valid == i)
    if np.sum(mask) > 0:
        acc = np.mean(y_pred_classes[mask] == i)
        class_accuracies.append(acc)
    else:
        class_accuracies.append(0)

axes[1, 1].bar(range(len(class_accuracies[:20])), class_accuracies[:20])
axes[1, 1].set_title('Точность по первым 20 классам')
axes[1, 1].set_xlabel('Класс')
axes[1, 1].set_ylabel('Точность')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('enhanced_model/training_analysis.png', dpi=150)
plt.show()

print("\nСОХРАНЕНИЕ МОДЕЛИ")
print("-" * 50)

model.save('enhanced_model/final_model.keras')
model.save_weights('enhanced_model/final_weights.weights.h5')

with open('enhanced_model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('enhanced_model/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

metadata = {
    'accuracy': float(val_accuracy),
    'classes': label_encoder.classes_.tolist(),
    'input_dim': input_dim,
    'model_type': 'enhanced_classifier',
    'num_features': input_dim,
    'feature_names': [f'feature_{i}' for i in range(input_dim)]
}
with open('enhanced_model/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("Модель сохранена в папке 'enhanced_model'")

print("\nТЕСТИРОВАНИЕ НА ПРИМЕРАХ")
print("-" * 50)

np.random.seed(42)
indices = np.random.choice(len(valid_features_scaled), 10, replace=False)

correct = 0
print("\nРезультаты на тестовых примерах:")
print("-" * 60)
print(f"{'#':3} {'Истина':15} {'Предсказано':15} {'Уверенность':12} {'Результат'}")
print("-" * 60)

for i, idx in enumerate(indices):
    features = valid_features_scaled[idx].reshape(1, -1)
    pred = model.predict(features, verbose=0)[0]
    pred_class = label_encoder.classes_[np.argmax(pred)]
    true_class = valid_groups[idx]
    confidence = np.max(pred)

    is_correct = (pred_class == true_class)
    if is_correct:
        correct += 1

    mark = "OK" if is_correct else "ERR"
    print(f"{i + 1:3} {true_class:15} {pred_class:15} {confidence:.4f}       {mark}")

print("-" * 60)
print(f"\nТочность на тестовых примерах: {correct / 10:.0%}")

print("\nФУНКЦИЯ ДЛЯ ПРЕДСКАЗАНИЯ")
print("-" * 50)


def predict_enhanced(audio_array, model, scaler, label_encoder):
    features = extract_enhanced_features([audio_array])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled, verbose=0)[0]
    top_5_idx = np.argsort(pred)[-5:][::-1]

    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
    print("=" * 50)

    results = []
    for i, idx in enumerate(top_5_idx):
        planet = label_encoder.classes_[idx]
        confidence = pred[idx]
        results.append((planet, confidence))

        bar_length = int(confidence * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"{i + 1}. {planet:20} [{bar}] {confidence:.2%}")

    best_planet = label_encoder.classes_[np.argmax(pred)]
    best_confidence = np.max(pred)

    print("=" * 50)
    if best_confidence > 0.7:
        verdict = "ВЫСОКАЯ УВЕРЕННОСТЬ"
    elif best_confidence > 0.5:
        verdict = "СРЕДНЯЯ УВЕРЕННОСТЬ"
    else:
        verdict = "НИЗКАЯ УВЕРЕННОСТЬ"

    print(f"Лучший результат: {best_planet} ({best_confidence:.2%}) - {verdict}")

    return best_planet, best_confidence, results


with open('enhanced_model/predict_function.py', 'w', encoding='utf-8') as f:
    f.write("""
import numpy as np
import pickle
from tensorflow import keras

def extract_enhanced_features(audio_batch):
    features = []
    for audio in audio_batch:
        audio = audio.flatten()
        mean = np.mean(audio)
        std = np.std(audio)
        max_val = np.max(np.abs(audio))
        min_val = np.min(audio)
        rms = np.sqrt(np.mean(audio**2))
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        crest_factor = max_val / (rms + 1e-10)
        amplitude_ratio = max_val / (np.abs(min_val) + 1e-10)

        n_segments = 8
        segment_length = len(audio) // n_segments
        segment_energies = []
        segment_means = []
        for i in range(n_segments):
            start = i * segment_length
            end = start + segment_length
            segment = audio[start:end]
            segment_energies.append(np.sum(segment**2) / len(segment))
            segment_means.append(np.mean(segment))

        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1/16000)

        spectral_centroid = np.sum(freqs[:500] * fft[:500]) / (np.sum(fft[:500]) + 1e-10)
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs[:500] - spectral_centroid)**2) * fft[:500]) /
            (np.sum(fft[:500]) + 1e-10)
        )

        cumsum = np.cumsum(fft)
        rolloff_point = 0.95 * cumsum[-1]
        spectral_rolloff = freqs[np.searchsorted(cumsum, rolloff_point)]

        mel_bands = 13
        mel_filters = np.linspace(0, len(fft)-1, mel_bands+2).astype(int)
        mfcc_like = []
        for j in range(mel_bands):
            start = mel_filters[j]
            center = mel_filters[j+1]
            end = mel_filters[j+2]

            filter_bank = np.zeros(len(fft))
            filter_bank[start:center] = np.linspace(0, 1, center-start)
            filter_bank[center:end] = np.linspace(1, 0, end-center)

            mel_energy = np.sum(fft * filter_bank)
            mfcc_like.append(np.log(mel_energy + 1e-10))

        feature_vector = [
            mean, std, max_val, min_val, rms,
            zero_crossings, crest_factor, amplitude_ratio,
            spectral_centroid, spectral_bandwidth, spectral_rolloff
        ]
        feature_vector.extend(segment_energies)
        feature_vector.extend(segment_means)
        feature_vector.extend(mfcc_like)

        features.append(feature_vector)

    return np.array(features)

def predict_enhanced(audio_array, model, scaler, label_encoder):
    features = extract_enhanced_features([audio_array])
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled, verbose=0)[0]
    top_5_idx = np.argsort(pred)[-5:][::-1]

    print("\\n" + "="*50)
    print("РЕЗУЛЬТАТЫ КЛАССИФИКАЦИИ")
    print("="*50)

    results = []
    for i, idx in enumerate(top_5_idx):
        planet = label_encoder.classes_[idx]
        confidence = pred[idx]
        results.append((planet, confidence))

        bar_length = int(confidence * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"{i+1}. {planet:20} [{bar}] {confidence:.2%}")

    best_planet = label_encoder.classes_[np.argmax(pred)]
    best_confidence = np.max(pred)

    print("="*50)
    if best_confidence > 0.7:
        verdict = "ВЫСОКАЯ УВЕРЕННОСТЬ"
    elif best_confidence > 0.5:
        verdict = "СРЕДНЯЯ УВЕРЕННОСТЬ"
    else:
        verdict = "НИЗКАЯ УВЕРЕННОСТЬ"

    print(f"Лучший результат: {best_planet} ({best_confidence:.2%}) - {verdict}")

    return best_planet, best_confidence, results

if __name__ == "__main__":
    model = keras.models.load_model('final_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)

    print("Модель загружена")
""")


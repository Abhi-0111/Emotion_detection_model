import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

# --- 1. GPU CHECK ---
print("Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Detected: {gpus[0]}")
    # Prevent TF from taking all 12GB VRAM at once
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("❌ No GPU found. Running on CPU.")

# --- 2. CONFIGURATION ---
# Replace with the path where you unzipped RAVDESS
DATA_PATH = 'data/RAVDESS/' 
EMOTIONS = {
    '01':'neutral', '02':'calm', '03':'happy', '04':'sad', 
    '05':'angry', '06':'fearful', '07':'disgust', '08':'surprised'
}

def extract_features(file_path, max_pad_len=174):
    # Duration=3 to keep inputs consistent
    audio, sr = librosa.load(file_path, res_type='kaiser_fast', duration=3, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # Pad or truncate to max_pad_len
    if mfccs.shape[1] < max_pad_len:
        mfccs = np.pad(mfccs, pad_width=((0,0), (0, max_pad_len - mfccs.shape[1])))
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs.T

# --- 3. DATA LOADING ---
def load_ravdess_data(path):
    x, y = [], []
    print("Extracting features (this may take a few minutes)...")
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                # RAVDESS format: 03-01-XX... where XX is emotion
                emotion_num = file.split('-')[2]
                emotion = EMOTIONS[emotion_num]
                
                feature = extract_features(os.path.join(root, file))
                x.append(feature)
                y.append(emotion)
    return np.array(x), np.array(y)

# Execute Loading
X, y_raw = load_ravdess_data(DATA_PATH)

# Encode labels
le = LabelEncoder()
y = to_categorical(le.fit_transform(y_raw))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. MODEL ARCHITECTURE (CNN-LSTM) ---
model = models.Sequential([
    layers.Input(shape=(X.shape[1], X.shape[2])), # (174, 40)
    
    # CNN Branch
    layers.Conv1D(128, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=4),
    layers.Dropout(0.3),
    
    # LSTM Branch
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),
    
    # Dense Classifier
    layers.Dense(64, activation='relu'),
    layers.Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 5. TRAINING ---
print("\nStarting Training on RTX 4050...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('speech_emotion_model.h5')
print("Model saved as speech_emotion_model.h5")
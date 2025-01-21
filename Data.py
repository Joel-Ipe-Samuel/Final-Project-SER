import os
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from Modules import extract_features

# Function to apply Gaussian noise
def add_gaussian_noise(signal, noise_factor=0.005):
    noise = np.random.randn(len(signal))
    return signal + noise_factor * noise

def pitch_shift(signal, sr, n_steps=2):
    return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)

def time_stretch(signal, rate=1.1):
    return librosa.effects.time_stretch(signal, rate=rate)

# Function to normalize the features
def normalize_features(features, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        return features, scaler
    else:
        return scaler.transform(features)

# Load and preprocess data
def load_data(data_paths, augment_data=False):
    features, labels = [], []
    accepted_file_count = 0

    for data_path in data_paths:
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".wav"):
                    file_path = os.path.join(root, file)
                    
                    # Extract label based on dataset-specific conditions
                    if 'TESS' in data_path:
                        emotion_code = file.split("_")[-1].split(".")[0]
                        tess_emotion_dict = {
                            'sad': 'sad',
                            'happy': 'happy',
                            'angry': 'angry',
                            'ps': 'surprise',
                            'fear': 'fear',
                            'disgust': 'disgust',
                            'neutral': 'neutral'
                        }
                        label = tess_emotion_dict.get(emotion_code, None)
                    elif 'RAVDESS' in data_path:
                        emotion_code = file.split("-")[2]  # Extract the emotion code (3rd part)
                        ravdess_emotion_dict = {
                            '01': 'neutral',
                            '02': 'neutral',
                            '03': 'happy',
                            '04': 'sad',
                            '05': 'angry',
                            '06': 'fear',
                            '07': 'disgust',
                            '08': 'surprise'
                        }
                        label = ravdess_emotion_dict.get(emotion_code, None)
                    elif 'CREMA-D' in data_path:
                        emotion_code = file.split("_")[2]  # Extract the emotion code (3rd part)
                        crema_emotion_dict = {
                            'ANG': 'angry',
                            'DIS': 'disgust',
                            'FEA': 'fear',
                            'HAP': 'happy',
                            'NEU': 'neutral',
                            'SAD': 'sad'
                        }
                        label = crema_emotion_dict.get(emotion_code, None)
                    elif 'SAVEE' in data_path:
                        emotion_code = file[3:5]
                        savee_emotion_dict = {
                            'sa': 'sad',
                            'ha': 'happy',
                            'an': 'angry',
                            'su': 'surprise',
                            'fe': 'fear',
                            'di': 'disgust',
                            'ne': 'neutral',
                            'h': 'happy',
                            'n': 'neutral',
                            'f': 'fear',
                            'd': 'disgust',
                            'a': 'angry'
                        }
                        label = savee_emotion_dict.get(emotion_code, None)
                    else:
                        continue
                    
                    if label is None:
                        continue

                    try:
                        signal, sr = librosa.load(file_path, sr=None)

                        # Data augmentation
                        if augment_data:
                            augmented_signals = [
                                signal,
                                add_gaussian_noise(signal),
                                pitch_shift(signal, sr),
                                time_stretch(signal)
                            ]
                        else:
                            augmented_signals = [signal]

                        # Extract features for each augmented version
                        for augmented_signal in augmented_signals:
                            extracted_features, _ = extract_features(augmented_signal, sr)
                            features.append(extracted_features)
                            labels.append(label)
                            accepted_file_count += 1
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    print(f"Total accepted files: {accepted_file_count}")

    if len(features) == 0:
        raise ValueError("No features extracted. Check dataset paths or audio file processing.")

    features, scaler = normalize_features(np.array(features))
    return np.array(features), np.array(labels), scaler

# Load data
data_paths = [r'Datasets\TESS',
              r'Datasets\CREMA-D',
              r'Datasets\SAVEE',
              r'Datasets\RAVDESS']

X, y, scaler = load_data(data_paths, augment_data=True)

if len(X) == 0 or len(y) == 0:
    raise ValueError("No audio data found. Please check the dataset paths and ensure they are correctly formatted.")

# Encode the labels
encoder = LabelEncoder()
encoder.fit(['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise'])
y_encoded = encoder.transform(y)

# Balance the dataset using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y_encoded)

# Save processed data, encoder, and scaler
save_dir = r'Datasets\Processed Data'
os.makedirs(save_dir, exist_ok=True)
np.savez(os.path.join(save_dir, 'data.npz'), X=X_balanced, y=y_balanced)

with open(os.path.join(save_dir, 'encoder.pkl'), 'wb') as file:
    pickle.dump(encoder, file)
with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as file:
    pickle.dump(scaler, file)

print("Data preprocessing and augmentation complete.")
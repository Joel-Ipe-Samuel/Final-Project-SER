import os
import torch
import pickle
import librosa
from Modules import EnhancedSERModel, extract_features
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load environment variables
load_dotenv()

# Audio settings and device setup
SAMPLE_RATE = 16000  # 16 kHz for compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load saved SER model
model_path = os.path.join("Saved Models", "Best Overall Model.pth")
input_size = 128 + 40 + 12 + 7  # Adjust based on feature extraction in data.py
num_classes = 7  # Example: Adjust to your dataset
model = EnhancedSERModel(input_size, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load encoder
encoder_path = os.path.join("Datasets", "Processed Data", "encoder.pkl")
with open(encoder_path, 'rb') as file:
    encoder = pickle.load(file)

# Function to perform emotion prediction
def predict_emotion(audio_path):
    signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    features, _ = extract_features(signal, sr)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
    emotion = encoder.inverse_transform(predicted.cpu().numpy())[0]
    filenameemo="Emotions.txt"
    with open(filenameemo, "a") as file:
        file.write(f"SER: {emotion}\n")
        file.write(f"FER: Sad\n")


import os
import torch
import pickle
import numpy as np
import sounddevice as sd
import librosa
import openai
from scipy.io.wavfile import write
from Modules import EnhancedSERModel, extract_features
from transcribe import load_audio, process_audio_from_file
from pynput import keyboard  # For handling key events
global is_recording, exit_program
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("joemama")

# Audio recording settings
SAMPLE_RATE = 16000  # 16 kHz for compatibility

# Device setup
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

# Global variables for recording state
is_recording = False
audio_data_buffer = []
exit_program = False

# Function to handle recording logic
def record_audio_toggle(sample_rate):
    global is_recording, audio_data_buffer
    print("Press 'b' to start/stop recording.")
    
    def callback(indata, frames, time, status):
        if is_recording:
            audio_data_buffer.append(indata.copy())
        else:
            raise sd.CallbackStop()

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', callback=callback):
        while is_recording:
            pass  # Wait while recording
    
    audio_data = np.concatenate(audio_data_buffer, axis=0)
    audio_data_buffer = []  # Clear buffer after saving
    return np.squeeze(audio_data)

# Function to save audio as WAV file
def save_audio(audio_data, file_path, sample_rate):
    write(file_path, sample_rate, (audio_data * 32767).astype(np.int16))

# Function to perform emotion prediction
def predict_emotion(audio_path):
    signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    features, _ = extract_features(signal, sr)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
    emotion = encoder.inverse_transform(predicted.cpu().numpy())[0]
    return emotion

# Function to convert speech to text using OpenAI Whisper API
def speech_to_text(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        response = openai.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
        )
    return response['text']

# Main script
if __name__ == "__main__":
    print("Press 'b' to start/stop recording and 'q' to quit.")

    def on_press(key):
        global is_recording, exit_program
        try:
            if key.char == 'b':
                is_recording = not is_recording
                if is_recording:
                    print("Recording started...")
                else:
                    print("Recording stopped.")
            elif key.char == 'q':
                exit_program = True
                print("Exiting program...")
                return False  # Stop listener
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while not exit_program:
        if is_recording:
            audio_data = record_audio_toggle(SAMPLE_RATE)

            # Save recorded audio
            audio_file_path = "Recording.wav"
            save_audio(audio_data, audio_file_path, SAMPLE_RATE)

            # Predict emotion
            predicted_emotion = predict_emotion(audio_file_path)
            print(f"Predicted Emotion: {predicted_emotion}")

            # Convert speech to text
            transcription = process_audio_from_file(audio_file_path)
            # Prompt user to redo recording
            redo = input("Do you want to redo the recording? (y/n): ").strip().lower()
            if redo == 'y':
                print("\nRedoing the recording...\nPress 'b' to start/stop recording and 'q' to quit.")
            else:
                print("Proceeding...")
                break

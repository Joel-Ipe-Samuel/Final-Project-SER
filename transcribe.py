import whisper
import numpy as np
import torch
import jiwer  # Library for calculating Word Error Rate (WER)
import librosa
from TER import terprocess

# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the model (load to the appropriate device)
model = whisper.load_model("medium").to(device)

# Ground truth transcription (example)
# Ideally, you will have a dataset with reference transcriptions.
reference_text = "This is an example of what I want the transcription to be."

# Function to load audio from a file and convert it to the format Whisper expects (16kHz mono)
def load_audio(file_path, target_sample_rate=16000):
    # Load the audio file using librosa
    audio, sr = librosa.load(file_path, sr=target_sample_rate, mono=True)
    return audio

# Function to process audio from a file and calculate WER
def process_audio_from_file(file_path):
    print(f"Processing file: {file_path}")

    # Load audio from file
    audio = load_audio(file_path)

    # Transcribe the entire audio
    result = model.transcribe(audio)
    recognized_text = result['text']
    print(f"Recognized: {recognized_text}")

    # Calculate WER against the reference transcription
    wer = jiwer.wer(reference_text, recognized_text)
    print(f"Word Error Rate (WER): {wer:.4f}")

    terprocess(recognized_text)


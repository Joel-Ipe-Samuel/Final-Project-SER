import whisper
import numpy as np
import torch
import jiwer  # Library for calculating Word Error Rate (WER)
import librosa


# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the model (load to the appropriate device)
model = whisper.load_model("medium").to(device)  #potentially need large model for longer sequeces of inputs 

# Function to load audio from a file and convert it to the format Whisper expects (16kHz mono)
def load_audio(file_path, target_sample_rate=16000):
    # Load the audio file using librosa
    audio, sr = librosa.load(file_path, sr=target_sample_rate, mono=True)
    return audio

# Function to process audio from a file and calculate WER
def process_audio_from_file(file_path):
    
    # Load audio from file
    audio = load_audio(file_path)

    # Transcribe the entire audio
    result = model.transcribe(audio)
    recognized_text = result['text']
    
    file_path=r'User Response.txt'
    # Save the model's response to a file
    with open(file_path, "a") as file:
        file.write(recognized_text)


    
    


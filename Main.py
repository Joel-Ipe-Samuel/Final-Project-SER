from flask import Flask, jsonify
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from SER import predict_emotion
from Transcribe import process_audio_from_file
from TER import terprocess
from Llama_Model import chat_with_model, conversation_history, generate_summary
from TTS_Model import text_to_speech
from Final_Emotion import read_emotions_from_file, determine_common_emotion
from FER import start_camera, stop_camera
import warnings
import threading
import time

warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)

# Global variables
SAMPLE_RATE = 16000
is_recording = False
audio_data_buffer = []
recording_thread = None

def recording_callback(indata, frames, time, status):
    if is_recording:
        audio_data_buffer.append(indata.copy())

def recording_process():
    global is_recording, audio_data_buffer
    
    # Set up the InputStream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=recording_callback
    )
    
    # Start the stream
    stream.start()
    
    # Wait while recording is True
    while is_recording:
        time.sleep(0.1)
    
    # Stop the stream when recording is set to False
    stream.stop()
    stream.close()
    
    if audio_data_buffer:
        # Process the recorded audio
        audio_data = np.concatenate(audio_data_buffer, axis=0)
        audio_file_path = "Recording.wav"
        save_audio(audio_data, audio_file_path, SAMPLE_RATE)
        
        # Clear the buffer
        audio_data_buffer.clear()
        
        # Process the recording
        process_recording(audio_file_path)

def save_audio(audio_data, file_path, sample_rate):
    write(file_path, sample_rate, (audio_data * 32767).astype(np.int16))

def process_recording(audio_file_path):
    # Define file paths upfront
    file_paths = [
        'Emotions.txt',
        'User Response.txt',
        'Model Response.txt'
    ]
    
    try:
        # Predict emotion using the SER model
        predict_emotion(audio_file_path)
        
        # Convert speech to text
        process_audio_from_file(audio_file_path)
        
        # Access user input text
        user_file_path = "User Response.txt"
        with open(user_file_path, "r") as file:
            user_text = file.read().strip()
        
        if not user_text:
            print("\nError: The file is empty.")
            return {"status": "error", "message": "No speech detected in recording"}
        
        # TER Model (text emotion recognition)
        terprocess(user_text)
        emotions = read_emotions_from_file()
        
        # Check if we have all the required emotions
        if "SER" in emotions and "FER" in emotions and "TER" in emotions:
            # Determine the common emotion
            common_emotion = determine_common_emotion(emotions)
        else:
            print("\nError: Not all emotion models are available in the file.")
            common_emotion = "neutral"  # Default fallback
        
        # Model response using chat
        chat_with_model(user_text, common_emotion)
        
        # Text-to-speech conversion
        text_to_speech()
        
        # Get the model's response before clearing files
        with open("Model Response.txt", "r") as file:
            model_response = file.read().strip()
        
        return {"status": "success", "response": model_response}
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return {"status": "error", "message": f"Processing error: {str(e)}"}
    
    finally:
        # Clear all files regardless of success or failure
        for file_path in file_paths:
            try:
                with open(file_path, "w") as file:
                    pass
                print(f"Cleared {file_path}")
            except Exception as e:
                print(f"Error clearing {file_path}: {str(e)}")
    
@app.route('/api/startRecording', methods=['GET'])
def start_recording():
    global is_recording, recording_thread
    
    if not is_recording:
        is_recording = True
        audio_data_buffer.clear()
        
        # Start the camera for facial emotion recognition
        start_camera()
        
        # Start recording in a separate thread
        recording_thread = threading.Thread(target=recording_process)
        recording_thread.start()
        
        return jsonify({"status": "success", "message": "Recording started"})
    
    return jsonify({"status": "error", "message": "Already recording"})

@app.route('/api/stopRecording', methods=['GET'])
def stop_recording():
    global is_recording, recording_thread
    
    if is_recording:
        is_recording = False
        
        # Stop the camera
        try:
            stop_camera()
        except Exception as e:
            print(f"Error stopping camera: {str(e)}")
        
        # Wait for the recording thread to finish
        if recording_thread:
            recording_thread.join()
        
        # Get the latest model response
        #model_response = "I didn't catch that. Could you please try again?"
        try:
            with open("Model Response.txt", "r") as file:
                content = file.read().strip()
                if content:
                    model_response = content
        except Exception as e:
            print(f"Error reading model response: {str(e)}")
        
        return jsonify({
            "status": "success", 
            "message": "Recording stopped",
            "response": model_response
        })
    
    return jsonify({"status": "error", "message": "Not recording"})

@app.route('/api/generateSummary', methods=['GET'])
def get_summary():
    if conversation_history:
        summary = generate_summary(conversation_history)
        return jsonify({"status": "success", "summary": summary})
    
    return jsonify({"status": "error", "message": "No conversation history to summarize"})

# Enable CORS to allow requests from your Next.js app
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

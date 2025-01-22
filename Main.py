import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from SER import predict_emotion 
from Transcribe import process_audio_from_file
from TER import terprocess
from Llama_Model import chat
from TTS_Model import text_to_speech
from pynput import keyboard
from Final_Emotion import read_emotions_from_file, determine_common_emotion
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Global variables
SAMPLE_RATE = 16000
is_recording = False
audio_data_buffer = []
exit_program = False

# Function to record audio
def record_audio_toggle(sample_rate):
    global is_recording, audio_data_buffer
    print("\nPress 'b' to start/stop recording.")
    
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

# Main script
if __name__ == "__main__":
    print("\nPress 'b' to start/stop recording and 'q' to quit.")

    def on_press(key):
        global is_recording, exit_program
        try:
            if key.char == 'b':
                is_recording = not is_recording
                if is_recording:
                    print("Recording started...")
                    #start_camera()
                else:
                    print("Recording stopped.")
                    #stop_camera()
            elif key.char == 'q':
                exit_program = True
                print("\nExiting program...")
                exit
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while not exit_program:
        if is_recording:
            # Record audio
            audio_data = record_audio_toggle(SAMPLE_RATE)

            # Save recorded audio
            audio_file_path = "Recording.wav"
            save_audio(audio_data, audio_file_path, SAMPLE_RATE)

            # Predict emotion using the SER model
            predict_emotion(audio_file_path)

            # Convert speech to text (if needed)
            process_audio_from_file(audio_file_path)

            # Access user input text
            user_file_path = "User Response.txt"
            with open(user_file_path, "r") as file:
                user_text = file.read().strip()

            if not user_text:
                print("\nError: The file is empty.")
            else:
                # TER Model (text emotion recognition)
                terprocess(user_text)
                emotions = read_emotions_from_file()

                # Check if we have all the required emotions
                if "SER" in emotions and "FER" in emotions and "TER" in emotions:
                # Determine the common emotion
                    common_emotion = determine_common_emotion(emotions)
                    print(common_emotion)
                else:
                    print("\nError: Not all emotion models are available in the file.")

                # Model response using chat
                chat(user_text, common_emotion)

                # Text-to-speech conversion
                text_to_speech()
            
            # Clear the file content after processing
            file_path1=r'Emotions.txt'
            file_path2=r'User Response.txt'
            file_path3=r'Model Response.txt'
            
            with open(file_path1, "w") as file:
                pass
                        
            with open(file_path2, "w") as file:
                pass
            
            with open(file_path3, "w") as file:
                pass
            
            # Prompt user to redo recording
            redo = input("\nDo you want to redo the recording? (y/n): ").strip().lower()
            if redo == 'y':
                print("\nRedoing the recording...\nPress 'b' to start/stop recording and 'q' to quit.")
            else:
                print("Proceeding...")
                break

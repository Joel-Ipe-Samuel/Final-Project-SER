import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
from collections import Counter
from threading import Thread, Lock
from cv2 import CAP_DSHOW

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor and model, then move the model to the selected device
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection", use_fast=True)
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection").to(device)

# Global variables
cap = None
emotions_collected = []
camera_active = False
lock = Lock()


def start_camera():
    """Start the camera and begin processing frames."""
    global cap, camera_active, emotions_collected
    cap = cv2.VideoCapture(0, CAP_DSHOW)
    camera_active = True
    emotions_collected = []
    thread = Thread(target=process_emotions)
    thread.start()


def stop_camera():
    """Stop the camera."""
    global cap, camera_active
    camera_active = False
    if cap:
        cap.release()
        cv2.destroyAllWindows()

    # Write the most common emotion to file after the camera stops
    write_emotion_to_file()


def process_emotions():
    """Continuously process frames and store emotions while the camera is active."""
    global emotions_collected, camera_active
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (224, 224))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE and Gaussian Blur
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray_frame = clahe.apply(gray_frame)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Convert to RGB and process
        resized_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(resized_frame)
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Extract and store emotion
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

        # Append the emotion in a thread-safe manner
        with lock:
            emotions_collected.append(predicted_label)


def write_emotion_to_file():
    """Write the most common emotion to the 'Emotions.txt' file."""
    global emotions_collected
    with lock:
        if emotions_collected:
            most_common_emotion = Counter(emotions_collected).most_common(1)[0][0]
            with open("Emotions.txt", "a") as file:
                file.write(f"FER: {most_common_emotion}\n")

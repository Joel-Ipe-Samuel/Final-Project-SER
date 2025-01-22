import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the processor and model, then move the model to the selected device
processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection", use_fast=True)
model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection").to(device)

# Optional: Display model summary
print(model)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    frame=cv2.resize(frame,(224,224))
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to the image for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray_frame = clahe.apply(gray_frame)
    
    # Apply Gaussian Blur to reduce noise
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Convert the processed frame to 3D by adding a channel dimension
    resized_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    
    # Convert the resized frame to a PIL image
    image = Image.fromarray(resized_frame)

    # Process the image
    inputs = processor(images=image, return_tensors="pt").to(device)  # Move inputs to device

    # Run the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predictions
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

    # Get the class label
    labels = model.config.id2label
    predicted_label = labels[predicted_class_idx]

    # Display the resulting frame with prediction
    cv2.putText(frame, f"Emotion: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Real-time Emotion Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Face-detection 

#open vs code and make a file name face_dataset.py the copy the code
import cv2
import os

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# User ID for saving images
user_id = input("Enter user ID: ")
name = input("Enter user name: ")
print("Capturing images... Look at the camera.")

# Ensure the dataset directory exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Capture and save 30 face samples
count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        count += 1
        # Save the captured image into the dataset folder
        cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
        
        # Draw rectangle around the face and display the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow('Face Capture', frame)
    
    # Break if enough images are taken or 'q' is pressed
    if count >= 30 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Images for {name} with ID {user_id} captured successfully.")

#open vs code and make 2nd file name face_training.py the copy the code

import cv2
import numpy as np
import os
from PIL import Image

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Path to the dataset folder
dataset_path = 'dataset'

# Function to get images and labels
def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    
    for image_path in image_paths:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img_np = np.array(img, 'uint8')
        id = int(os.path.split(image_path)[-1].split(".")[1])  # Extract ID
        faces = face_cascade.detectMultiScale(img_np)
        
        for (x, y, w, h) in faces:
            face_samples.append(img_np[y:y+h, x:x+w])
            ids.append(id)
    
    return face_samples, ids

# Get face samples and corresponding IDs
faces, ids = get_images_and_labels(dataset_path)
recognizer.train(faces, np.array(ids))

# Save the model to trainer.yml
recognizer.save('trainer.yml')
print("Training complete and model saved as 'trainer.yml'.")



#open vs code and make file name face_recognition.py the copy the code
import cv2
import numpy as np

# Load the trained model and Haar Cascade for face detection
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ID-to-Name Mapping (replace with your names and IDs)
id_to_name = {
    1: "ASHUTOSH",
    2: "Samar",
    # Add more IDs and names as needed
}

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Recognize the face and get the ID and confidence
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        # Display name and confidence if recognition is successful
        if confidence < 50:  # Adjust confidence threshold as needed
            name = id_to_name.get(id, "Unknown")  # Get the name from the dictionary
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"
        
        # Display name and confidence
        cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, confidence_text, (x+5, y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


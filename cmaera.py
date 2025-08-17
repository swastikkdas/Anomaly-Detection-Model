import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Constants
IMAGE_SIZE = (64, 64)
CATEGORIES = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
              'Fighting', 'RoadAccidents', 'Robbery', 'Shooting',
              'Shoplifting', 'Stealing', 'Vandalism', 'Normal']

# Load trained model
model = load_model("Anomaly_detection_model.h5")


# Function to process video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame
        frame_resized = cv2.resize(frame, IMAGE_SIZE)
        frame_array = img_to_array(frame_resized) / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)

        # Predict crime
        prediction = model.predict(frame_array)
        predicted_label = CATEGORIES[np.argmax(prediction)]
        predictions.append(predicted_label)

        # Display frame with prediction
        cv2.putText(frame, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Crime Detection', frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Final prediction
    final_prediction = max(set(predictions), key=predictions.count)
    print(f"Final Video Prediction: {final_prediction}")


# Test with a sample video
video_path = "videoplayback (1).mp4"  # Change this to your test video path
process_video(video_path)

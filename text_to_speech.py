#!/usr/bin/env python3
import subprocess
import collections
import numpy as np
import tensorflow as tf
import cv2
from hand_segmentation import HandSegmentation

class PredictionStabilizer:
    def __init__(self, stability_frames=10):
        self.stability_frames = stability_frames
        self.predictions = collections.deque(maxlen=stability_frames)

    def update(self, probabilities):
        predicted_class = np.argmax(probabilities)
        self.predictions.append(predicted_class)

        if len(self.predictions) == self.stability_frames:
            most_common_class, count = collections.Counter(self.predictions).most_common(1)[0]
            if count / self.stability_frames >= 0.9:
                self.predictions.clear()
                return most_common_class
        return None


def text_to_speech(text):
    try:
        subprocess.run(['espeak', text])
    except FileNotFoundError:
        print("Error: eSpeak is not installed or not found in PATH.")


# Initialize Camera and Models
cap = cv2.VideoCapture(0)
segmenter = HandSegmentation()
model = tf.keras.models.load_model("models")
stabilizer = PredictionStabilizer(stability_frames=10)

# Adjust this mapping to your trained model classes
class_names = ['A', 'B', 'Bye', 'C', 'D', 'E', 'F', 'G', 'H', 'Hello', 'I', 'J', 'K', 'L', 'M',
               'N', 'No', 'O', 'P', 'Perfect', 'Q', 'R', 'S', 'T', 'Thank You', 'U', 'V', 'W',
               'X', 'Y', 'Yes', 'Z', 'del', 'nothing', 'space']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    segmented_hand, _, _ = segmenter.segment_hand(frame)
    if segmented_hand is not None:
        input_image = cv2.resize(segmented_hand, (224, 224))
        input_image = np.expand_dims(input_image, axis=0)
        input_image = tf.keras.applications.mobilenet_v2.preprocess_input(input_image)

        probabilities = model.predict(input_image, verbose=0)[0]

        stable_class = stabilizer.update(probabilities)
        if stable_class is not None:
            recognized_sign = class_names[stable_class]
            print(f"Stable Prediction: {recognized_sign}")

            # Text-to-speech output
            text_to_speech(recognized_sign)

            # Show recognized sign on the camera feed
            cv2.putText(frame, recognized_sign, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Real-time Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

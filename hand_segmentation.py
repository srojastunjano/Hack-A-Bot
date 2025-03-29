import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf

class HandSegmentation:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def segment_hand(self, image):
        """
        Segment hand from the image using MediaPipe
        Returns the segmented hand image and the original image with hand landmarks
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # Get the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Create a mask for the hand
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            h, w = image.shape[:2]
            
            # Get hand landmarks coordinates
            points = []
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                points.append([x, y])
            
            # Convert points to numpy array
            points = np.array(points, dtype=np.int32)
            
            # Create convex hull
            hull = cv2.convexHull(points)
            
            # Draw the hull on the mask
            cv2.fillConvexPoly(mask, hull, 255)
            
            # Apply morphological operations to improve the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Apply the mask to the original image
            segmented_hand = cv2.bitwise_and(image, image, mask=mask)
            
            # Draw landmarks on the original image
            annotated_image = image.copy()
            self.mp_draw.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )
            
            return segmented_hand, annotated_image, mask
            
        return None, None, None
    
    def preprocess_for_model(self, segmented_hand, target_size=(224, 224)):
        """
        Preprocess the segmented hand image for the model
        """
        if segmented_hand is None:
            return None
            
        # Resize the image
        resized = cv2.resize(segmented_hand, target_size)
        
        # Convert to RGB if needed
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[2] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
        # Preprocess for MobileNetV2
        preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
        
        # Add batch dimension
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        return preprocessed

def process_raspberry_pi_camera():
    """
    Process video feed from Raspberry Pi camera with hand segmentation
    """
    # Initialize camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera
    
    # Initialize hand segmentation
    hand_segmenter = HandSegmentation()
    
    # Load the trained model
    model = tf.keras.models.load_model("models")
    
    # Class mapping (from test.py)
    class_indices = {
        'A': 0, 'B': 1, 'Bye': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7,
        'H': 8, 'Hello': 9, 'I': 10, 'J': 11, 'K': 12, 'L': 13, 'M': 14,
        'N': 15, 'No': 16, 'O': 17, 'P': 18, 'Perfect': 19, 'Q': 20,
        'R': 21, 'S': 22, 'T': 23, 'Thank You': 24, 'U': 25, 'V': 26,
        'W': 27, 'X': 28, 'Y': 29, 'Yes': 30, 'Z': 31, 'del': 32,
        'nothing': 33, 'space': 34
    }
    class_names = list(class_indices.keys())
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Segment hand
        segmented_hand, annotated_image, mask = hand_segmenter.segment_hand(frame)
        
        if segmented_hand is not None:
            # Preprocess for model
            preprocessed = hand_segmenter.preprocess_for_model(segmented_hand)
            
            # Make prediction
            predictions = model.predict(preprocessed)
            predicted_class_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_index]
            
            # Display prediction
            prediction_text = f"{class_names[predicted_class_index]} ({confidence:.2f})"
            cv2.putText(annotated_image, prediction_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show the processed frames
            cv2.imshow('Segmented Hand', segmented_hand)
            cv2.imshow('Annotated Image', annotated_image)
            cv2.imshow('Hand Mask', mask)
        
        # Show original frame
        cv2.imshow('Original', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_raspberry_pi_camera() 
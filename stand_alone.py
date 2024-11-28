import cv2
import sys
import signal
import socketio
import time
import numpy as np
import tensorflow as tf
from mediapipe import solutions as mp_solutions
from flask_socketio import SocketIO
from app import socketio   # Assuming the SocketIO instance is defined in app.py

# Load the trained model
model = tf.keras.models.load_model('resnet_model.h5')

# Class labels for predictions
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'hello', 
'thanks', 'yes', 'no', 'please', 'sorry'] 

# Initialize MediaPipe Hands
mp_hands = mp_solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp_solutions.drawing_utils

# Global variable to control the loop
stop_prediction = False

def preprocess_frame(frame):
    img_size = (224, 224)
    frame_resized = cv2.resize(frame, img_size)
    frame_normalized = frame_resized / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded

def run_real_time_prediction():
    global stop_prediction
    cap = cv2.VideoCapture(0)

    while not stop_prediction:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            height, width, _ = frame.shape
            hand_landmarks = results.multi_hand_landmarks[0]
            x_coords = [lm.x * width for lm in hand_landmarks.landmark]
            y_coords = [lm.y * height for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            hand_region = frame[y_min:y_max, x_min:x_max]
            if hand_region.size > 0:
                preprocessed_frame = preprocess_frame(hand_region)
                predictions = model.predict(preprocessed_frame)
                predicted_class = class_labels[np.argmax(predictions)]
                confidence = np.max(predictions)

                # Send the prediction to Flask UI
                send_prediction(predicted_class, confidence)

                # Display the prediction on the frame
                cv2.putText(
                    frame, 
                    f"Prediction: {predicted_class} ({confidence:.2f})", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2, 
                    cv2.LINE_AA
                )
        else:
            cv2.putText(
                frame, 
                "No hands detected", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2, 
                cv2.LINE_AA
            )

        cv2.imshow("Real-Time Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting Real-Time Prediction.")
            break

    cap.release()
    cv2.destroyAllWindows()

def send_prediction(prediction, confidence):
    data = {'prediction': prediction, 'confidence': confidence}
    print(f"Sending prediction: {prediction}, Confidence: {confidence}")  # Debugging line
    socketio.emit('prediction_data', data)

def signal_handler(sig, frame):
    global stop_prediction
    print('Exiting Real-Time Prediction...')
    stop_prediction = True  # Set the flag to stop the while loop


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    run_real_time_prediction()

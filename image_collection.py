import os
import cv2
import shutil

# List of labels you want to redo, including moving signs
redo_labels = ['J', 'O', 'Z', 'thanks', 'yes', 'no', 'sorry']

# Directory to store the collected data
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# List of all possible labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
          'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'hello', 'thanks', 
          'yes', 'no', 'please', 'sorry']

# Number of images to collect per label
dataset_size = 100

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create directories for each label if they don't already exist
for label in labels:
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

# Function to capture data for each label
def capture_data_for_label(label):
    print(f'Collecting data for label: {label}')

    done = False
    while not done:
        # Capture frame from webcam
        ret, frame = cap.read()
        cv2.putText(frame, f'Collecting data for {label}. Press "Q" to start.', (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # Wait for user input to begin capturing data (press 'Q' to start)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            done = True

    # Capture the dataset for the current label
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        cv2.putText(frame, f'Collecting data for {label}. Image {counter+1}/{dataset_size}', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        # Save the captured frame
        image_path = os.path.join(DATA_DIR, label, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)
        
        counter += 1

    print(f'Data collection for label {label} completed.')

# Function to redo collection for specific labels
def redo_collection():
    for label in redo_labels:
        print(f"Redoing data collection for label: {label}")
        
        # Delete existing data for this label
        label_dir = os.path.join(DATA_DIR, label)
        if os.path.exists(label_dir):
            shutil.rmtree(label_dir)  # Remove old data for the label
            os.makedirs(label_dir)    # Recreate the directory
        
        # Collect new data for the label
        capture_data_for_label(label)

# Redo collection for specific labels (moving signs and others)
redo_collection()

# Release webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

print('Data collection process finished.')

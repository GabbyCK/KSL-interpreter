import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Constants
IMAGE_SIZE = (64, 64)  # Resizing to 64x64 for CNN model
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
          'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'hello', 'thanks', 
          'yes', 'no', 'please', 'sorry']  # Labels as strings

def prepare_data(data_dir):
    data = []
    labels = []

    label_map = {label: idx for idx, label in enumerate(LABELS)}  # Map labels to integer indices

    for label in LABELS:
        folder_path = os.path.join(data_dir, label)

        if not os.path.exists(folder_path):
            print(f"Skipping missing directory: {folder_path}")
            continue

        count = 0
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only image files
                try:
                    # Load the image and resize it to the target size
                    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    data.append(img_array)
                    labels.append(label_map[label])  # Store the integer label
                    count += 1
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
        print(f"Loaded {count} images for label: {label}")

    # Normalize data (pixel values to range [0, 1])
    data = np.array(data) / 255.0
    labels = np.array(labels)

    # Ensure there is data before proceeding
    if data.size == 0 or labels.size == 0:
        print("No data found. Please check your dataset directory.")
        return None, None, None, None

    # Check data shapes and labels
    print(f"Shape of first image: {data[0].shape if data.size > 0 else 'N/A'}")
    print(f"Shape of first label: {labels[0] if labels.size > 0 else 'N/A'}")

    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Convert labels to one-hot encoding after splitting the data
    y_train = to_categorical(y_train, num_classes=32)
    y_test = to_categorical(y_test, num_classes=32)

    y_train = y_train.reshape(y_train.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)


    # Check the one-hot encoded labels shapes
    print(f"Shape of one-hot encoded y_train: {y_train.shape}")
    print(f"Shape of one-hot encoded y_test: {y_test.shape}")

    print(f"y_train shape after preparation: {y_train.shape}")
    print(f"y_test shape after preparation: {y_test.shape}")

    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")


    return X_train, X_test, y_train, y_test


# Usage
data_dir = './data'
X_train, X_test, y_train, y_test = prepare_data(data_dir)

if X_train is not None:
    print("Data preparation successful!")
else:
    print("Data preparation failed.")

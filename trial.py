import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

# Constants
IMAGE_SIZE = (224, 224)  # Resizing to 224x224 for ResNet50 compatibility
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
          'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'hello', 'thanks', 
          'yes', 'no', 'please', 'sorry']  # Labels as strings

# Prepare data function
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

    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Convert labels to one-hot encoding after splitting the data
    y_train = to_categorical(y_train, num_classes=len(LABELS))
    y_test = to_categorical(y_test, num_classes=len(LABELS))

    return X_train, X_test, y_train, y_test


# Define your ResNet50 model
def create_resnet_model(input_shape):
    # Load the pre-trained ResNet50 model without the top layers (classification layers)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + (3,))

    # Freeze the layers of the base model
    base_model.trainable = False

    # Create a new model on top of ResNet50
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Reduces 4D output to 2D
        layers.Dense(128, activation='relu'),
        layers.Dense(len(LABELS), activation='softmax')  # Adjusted for multi-class classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Prepare data
data_dir = './data'
X_train, X_test, y_train, y_test = prepare_data(data_dir)

# Check if data is successfully loaded
if X_train is not None:
    print("Data preparation successful!")

    # Create the ResNet model
    model = create_resnet_model(X_train.shape[1:])

    # Train the model
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

    # Save the model after training
    model.save('resnet_model.h5')  
    print("Model saved successfully!")
else:
    print("Data preparation failed.")

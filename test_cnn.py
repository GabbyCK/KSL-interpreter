import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model (replace with your model path)
model = tf.keras.models.load_model('model.h5')

# Define the label mapping
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
          'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'hello', 'thanks', 
          'yes', 'no', 'please', 'sorry']  # Assuming 32 labels

# Path to the image you want to test
image_path = './data/H/12.jpg'  # Modify if your image has a different name/extension

# Load the image
img = image.load_img(image_path, target_size=(64, 64))  # Resize to match your training image size
img_array = image.img_to_array(img)  # Convert image to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize image

# Predict the label
prediction = model.predict(img_array)

# Get the predicted label (index of the highest probability)
predicted_index = np.argmax(prediction, axis=1)

# Get the predicted label from the LABELS list
predicted_label = LABELS[predicted_index[0]]

print(f"The model predicted: {predicted_label}")

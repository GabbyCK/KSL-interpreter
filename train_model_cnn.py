import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from prepare_data_cnn import prepare_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

# Path to your dataset
data_dir = './data'

# Prepare data (get training and testing data)
X_train, X_test, y_train, y_test = prepare_data(data_dir)

# Debugging shapes before any processing
print(f"Initial X_train shape: {X_train.shape}")
print(f"Initial X_test shape: {X_test.shape}")
print(f"Initial y_train shape: {y_train.shape}")
print(f"Initial y_test shape: {y_test.shape}")

# Ensure y_train and y_test are one-hot encoded and have correct shapes
num_classes = 32  # Total number of classes in your dataset
y_train = to_categorical(y_train, num_classes=num_classes)  # Output shape: (2560, 32)
y_test = to_categorical(y_test, num_classes=num_classes)  

# Debugging shapes after one-hot encoding
print(f"X_train shape: {X_train.shape}")  # Expected: (2560, 64, 64, 3)
print(f"X_test shape: {X_test.shape}")    # Expected: (640, 64, 64, 3)
print(f"y_train shape: {y_train.shape}")  # Expected: (2560, 32)
print(f"y_test shape: {y_test.shape}")    # Expected: (640, 32)


# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data generator on training images
datagen.fit(X_train)

# Define custom data generator for augmented images and labels
def data_generator_with_labels(datagen, X, y, batch_size):
    gen = datagen.flow(X, y, batch_size=batch_size)
    while True:
        X_batch, y_batch = next(gen)
        yield X_batch, y_batch

# Custom training data generator
train_generator = data_generator_with_labels(datagen, X_train, y_train, batch_size=32)

# Load the pre-trained ResNet50 model without the top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Freeze some layers of the pre-trained model
for layer in base_model.layers[:-12]:  # Freeze all layers except the last 12
    layer.trainable = True

# Add custom layers for our task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(num_classes, activation='softmax')(x)  # Output layer for 32 classes

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Reduce learning rate on plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Debugging the generator's batch output
X_batch, y_batch = next(iter(train_generator))
print(f"Batch shape: {X_batch.shape}, {y_batch.shape}")

# Train the model
model.fit(
    train_generator,
    epochs=20,
    validation_data=(X_test / 255.0, y_test),  # Normalize test data
    steps_per_epoch=int(np.ceil(len(X_train) / 32)),
    verbose=1,
    callbacks=[reduce_lr, early_stopping]
)


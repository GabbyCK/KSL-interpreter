import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

def create_resnet_model(input_shape=(224, 224, 3), num_classes=33):
    # Load ResNet50 pre-trained model without the top layers (classification layers)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
for layer in base_model.layers[-20:]:
    layer.trainable = True

    # Add custom layers on top of the base ResNet model
    model = models.Sequential([
        base_model,  # ResNet50 base model
        layers.GlobalAveragePooling2D(),  # Pooling layer
        layers.Dense(256, activation='relu'),  # Fully connected layer
        layers.Dropout(0.5),  # Dropout for regularization
        layers.Dense(num_classes, activation='softmax')  # Output layer (33 classes)
    ])
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Generate dummy data for demonstration
# Assuming you have 1000 samples
num_samples = 1000
num_features = 57
num_classes = 7

# Create random data
X = np.random.rand(num_samples, num_features)  # 1000 samples, 57 features
y = np.random.randint(0, num_classes, num_samples)  # Random integer labels for 7 classes
y = tf.keras.utils.to_categorical(y, num_classes)  # One-hot encoding

# Define the CNN model using Keras Functional API
def create_cnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # Reshape input to add a channel dimension
    x = layers.Reshape((input_shape[0], 1, 1))(inputs)  # Reshape to (57, 1, 1)
    print(x)
    
    # Convolutional layers
    x = layers.Conv2D(32, kernel_size=(3, 1), activation='relu')(x)  # 32 filters, kernel size (3, 1)
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)  # Max pooling
    
    x = layers.Conv2D(64, kernel_size=(3, 1), activation='relu')(x)  # 64 filters
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)  # Max pooling
    
    x = layers.Conv2D(128, kernel_size=(3, 1), activation='relu')(x)  # 128 filters
    x = layers.MaxPooling2D(pool_size=(2, 1))(x)  # Max pooling
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Dropout for regularization
    x = layers.Dense(64, activation='relu')(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Create the model
input_shape = (num_features,)  # 57 features
model = create_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
tf.keras.utils.plot_model(model, "/home/mpaul/projects/mpaul/mai/src_3/cnn.png", show_shapes=True)

model.summary()

# Train the model
# Assuming you have a train-test split
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X, y)
print(f"Loss: {loss}, Accuracy: {accuracy}")

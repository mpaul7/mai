import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic data for demonstration
def generate_synthetic_data(num_samples=1000, time_steps=10, num_features=2):
    X = np.random.rand(num_samples, time_steps, num_features)
    y = np.random.randint(0, 2, size=(num_samples, 1))  # Binary classification
    return X, y

# Generate training and testing data
X_train, y_train = generate_synthetic_data(num_samples=800)
X_test, y_test = generate_synthetic_data(num_samples=200)

# Define the LSTM model
def create_lstm_model(input_shape, dropout_rate=0.2):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(32),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Create the model
input_shape = (X_train.shape[1], X_train.shape[2])
model = create_lstm_model(input_shape)

# Set up MLflow tracking
mlflow.set_experiment("LSTM-MLflow-Demo5")

with mlflow.start_run(run_name='run1'):
    # Log model parameters
    mlflow.log_param("dropout_rate", 0.2)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("epochs", 20)
    mlflow.log_param("batch_size", 32)

    # Define early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Log metrics
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", accuracy)

    # Prepare input example and infer signature
    input_example = X_test[:1]  # Example input for the model
    print(input_example)
    signature = infer_signature(X_test, model.predict(X_test))
    print(signature)

    # Log the model with input example and signature
    mlflow.keras.log_model(
        model,
        artifact_path="lstm_model",
        input_example=input_example,
        signature=signature
    )

    print("Model logged to MLflow")

# Start MLflow UI (optional, run this in your terminal)
# mlflow ui

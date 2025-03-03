import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt  # Keras Tuner for hyperparameter tuning
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

# Define the model builder function for Keras Tuner
def build_model(hp):
    model = Sequential()
    # First LSTM layer with tunable units
    model.add(LSTM(units=hp.Int('units_lstm1', min_value=32, max_value=128, step=32),
                   return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))

    # Second LSTM layer with tunable units
    model.add(LSTM(units=hp.Int('units_lstm2', min_value=16, max_value=64, step=16)))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    # Compile the model with tunable learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    
    return model

# Initialize Keras Tuner
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='tuner_dir',
    project_name='lstm_hyperparameter_tuning'
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Search for the best hyperparameters
tuner.search(X_train, y_train, validation_split=0.2, epochs=20, callbacks=[early_stopping])

# Retrieve the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best Hyperparameters:")
print(f" - Units (LSTM1): {best_hps.get('units_lstm1')}")
print(f" - Units (LSTM2): {best_hps.get('units_lstm2')}")
print(f" - Dropout Rate: {best_hps.get('dropout_rate')}")
print(f" - Learning Rate: {best_hps.get('learning_rate')}")

# Set a default batch size or make it tunable
batch_size = 32  # Default batch size
if 'batch_size' in best_hps.values:
    batch_size = best_hps.get('batch_size')

# Build and train the best model
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=batch_size,
    callbacks=[early_stopping]
)

# Evaluate the model
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Log the results and model in MLflow
mlflow.set_experiment("LSTM-Hyperparameter-Tuning-Demo")

with mlflow.start_run():
    # Log the best hyperparameters
    mlflow.log_param("units_lstm1", best_hps.get('units_lstm1'))
    mlflow.log_param("units_lstm2", best_hps.get('units_lstm2'))
    mlflow.log_param("dropout_rate", best_hps.get('dropout_rate'))
    mlflow.log_param("learning_rate", best_hps.get('learning_rate'))
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", 20)

    # Log metrics
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", accuracy)

    # Prepare input example and infer signature
    input_example = X_test[:1]  # Example input for the model
    signature = infer_signature(X_test, best_model.predict(X_test))

    # Log the model with input example and signature
    mlflow.keras.log_model(
        best_model,
        artifact_path="best_lstm_model",
        input_example=input_example,
        signature=signature
    )

    print("Best model logged to MLflow")

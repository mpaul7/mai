import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
import mlflow
import itertools
import os
import numpy as np
from datetime import datetime

class LSTMExperiment:
    def __init__(self, train_dataset, val_dataset, input_shape, num_classes):
        """
        Initialize experiment parameters
        
        Args:
            train_dataset: TF dataset for training
            val_dataset: TF dataset for validation
            input_shape: Shape of input data (timesteps, features)
            num_classes: Number of output classes
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Base configuration
        self.base_config = {
            'lstm_units': params['lstm_units'],
            'dense_units': 32,
            'batch_size': 32,
            'epochs': 50
        }
        
        # Experiment configurations
        self.experiments = {
            'baseline': {'name': 'baseline'},
            
            'l1_reg': {
                'name': 'l1_regularization',
                'values': [0.01, 0.001, 0.0001]
            },
            
            'l2_reg': {
                'name': 'l2_regularization',
                'values': [0.01, 0.001, 0.0001]
            },
            
            'dropout': {
                'name': 'dropout',
                'values': [0.1, 0.3, 0.5]
            },
            
            'learning_rate': {
                'name': 'learning_rate',
                'values': [0.1, 0.01, 0.001, 0.0001]
            },
            
            'early_stopping': {
                'name': 'early_stopping',
                'values': [5, 10, 15]  # patience values
            }
        }

    def create_model(self, experiment_type=None, param_value=None):
        """Create LSTM model with specified regularization"""
        inputs = layers.Input(shape=self.input_shape)
        
        # LSTM layer configuration
        lstm_config = {}
        if experiment_type == 'l1_reg':
            lstm_config['kernel_regularizer'] = l1(param_value)
        elif experiment_type == 'l2_reg':
            lstm_config['kernel_regularizer'] = l2(param_value)
            
        x = layers.LSTM(self.base_config['lstm_units'], **lstm_config)(inputs)
        
        # Add dropout if specified
        if experiment_type == 'dropout':
            x = layers.Dropout(param_value)(x)
            
        x = layers.Dense(self.base_config['dense_units'], activation='relu')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        return Model(inputs=inputs, outputs=outputs)

    def get_callbacks(self, experiment_type=None, param_value=None):
        """Configure callbacks based on experiment type"""
        callbacks = []
        
        # Add ModelCheckpoint callback
        checkpoint_path = f"checkpoints/model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_loss',
                save_best_only=True
            )
        )
        
        # Add EarlyStopping if specified
        if experiment_type == 'early_stopping':
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=param_value,
                    restore_best_weights=True
                )
            )
        
        return callbacks

    def run_experiment(self, experiment_type, param_value=None):
        """Run a single experiment"""
        # Create model
        model = self.create_model(experiment_type, param_value)
        
        # Configure learning rate
        lr = param_value if experiment_type == 'learning_rate' else 0.001
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Get callbacks
        callbacks = self.get_callbacks(experiment_type, param_value)
        
        # Train model
        history = model.fit(
            self.train_dataset,
            epochs=self.base_config['epochs'],
            validation_data=self.val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        return history, model

    def run_all_experiments(self):
        """Run all experiments and log results with MLflow"""
        mlflow.set_experiment("lstm_overfitting_experiments")
        
        for exp_type, exp_config in self.experiments.items():
            if exp_type == 'baseline':
                # Run baseline experiment
                with mlflow.start_run(run_name=f"baseline"):
                    history, model = self.run_experiment(None)
                    self._log_metrics(history, exp_type, None)
            else:
                # Run experiments with different parameter values
                for value in exp_config['values']:
                    with mlflow.start_run(run_name=f"{exp_type}_{value}"):
                        history, model = self.run_experiment(exp_type, value)
                        self._log_metrics(history, exp_type, value)

    def _log_metrics(self, history, experiment_type, param_value):
        """Log metrics to MLflow"""
        # Log parameters
        mlflow.log_param("experiment_type", experiment_type)
        if param_value is not None:
            mlflow.log_param("param_value", param_value)
        
        # Log metrics for each epoch
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metrics({
                "train_loss": history.history['loss'][epoch],
                "train_accuracy": history.history['accuracy'][epoch],
                "val_loss": history.history['val_loss'][epoch],
                "val_accuracy": history.history['val_accuracy'][epoch]
            }, step=epoch)
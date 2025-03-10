
import warnings
from nntplib import decode_header

warnings.filterwarnings('ignore')

import os
import glob
import time
import click
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from keras.layers import Bidirectional
import tensorflow as tf
import tensorflow.data as tfd
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import CategoricalCrossentropy

import utils_dl

PROJECT_HOME = '/home/mpaul/projects/mpaul/mai'

class DLModels:
    def train_model(self, model_type, train_file, config, test_file):
        """Trains a DL ,model based on historical feature view data..

        Parameters:
                train_file: train data filename.
                config: model configuration filename (JSON)
                trained_model: output trained model filename.
        """

        with open(config) as f:
            params = json.load(f)
            
        _model = ''
        """Create model"""
        MODEL_JSON = f'{model_type}_model.json'
        MODEL_PNG = f'{model_type}_model.png'
        MODEL_H5 = f'{model_type}__model.h5'
        MODEL_CSV = f'{model_type}__mode.csv'
        output_units = len(params['labels'])
        if model_type == 'lstm':
            _model = utils_dl.create_dl_model_lstm(params, output_units)
        elif model_type == 'mlp':
            _model = utils_dl.create_dl_model_mlp(params, output_units)
        elif model_type == 'cnn':
            _model = utils_dl.create_dl_model_cnn(params, output_units)

        """Show model architecture"""
        # tf.keras.utils.plot_model(_model, f"{PROJECT_HOME}/models/{MODEL_PNG}", show_shapes=True)
        _model.summary()
        
        """Save model architecture in Json config file"""
        model_json = _model.to_json()
        with open(f"{PROJECT_HOME}/models/{MODEL_JSON}", "w") as f:
            json.dump(model_json, f)

        """ Load model architecture from Json config file"""
        
        with open(f"{PROJECT_HOME}/models/{MODEL_JSON}") as f:
            model_arch = json.load(f)
        model = tf.keras.models.model_from_json(model_arch)
        
        """Show loaded model architecture"""
        tf.keras.utils.plot_model(model, f"{PROJECT_HOME}/models/{MODEL_PNG}", show_shapes=True)
        model.summary()
        
        """Compile Model"""
        metrics = ['accuracy']
        losses = CategoricalCrossentropy()
        learning_rate = params['learning_rate']
        adam = Adam(learning_rate=learning_rate)
        
        model.compile(loss=losses, optimizer=adam, metrics=metrics)
        
        """Train Model"""
        
        # Create dataset
        train_dataset, validation_dataset = utils_dl.create_train_test_dataset_tf(data_file=train_file, params=params, train=True, evaluation=False)
        train_dataset = train_dataset.batch(params['train_batch_size'])
        validation_dataset = validation_dataset.batch(params['test_batch_size'])
        
        # Set training parameters
        options = {
                    'epochs': params['epochs'],
                    'steps_per_epoch': params['steps_per_epoch'],
                    'validation_data': validation_dataset,
                    'callbacks': []
                }
         
        log_dir = f"{PROJECT_HOME}/{params['log_dir']}"
        if log_dir:
            options['callbacks'].append(TensorBoard(log_dir=log_dir, histogram_freq=1))
            
        if params['early_stopping']:
            options['callbacks'].append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7))
        
        """ Train Model """
        model.fit(train_dataset, **options)
        
        """ Model Evaluation"""
        score = model.evaluate(validation_dataset , verbose=0)
        print(f'score-1 -> {score[1]}')
        print(f'score-0 -> {score[0]}')
        
        """Save Trained model"""
        model.save(f"{PROJECT_HOME}/models/{MODEL_H5}", save_format='h5')
        
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """ Model Evaluation"""
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """Load Trained Model Model"""
        loaded_model = load_model(f"{PROJECT_HOME}/models/{MODEL_H5}")
        
        """Create test dataset"""
        test_dataset = utils_dl.create_train_test_dataset_tf(data_file=test_file, params=params, train=False, evaluation=True)
        test_dataset = test_dataset.batch(128)
        
        """predict test data using loaded model"""
        y_test = np.concatenate([y for _, y in test_dataset], axis=0).argmax(axis=1)
        predictions = loaded_model.predict(test_dataset)
        predictions = predictions.argmax(axis=1)
        
        """Confusion Matrix"""
        labels = params['labels']
        labels = [labels[i] for i in np.unique(predictions).tolist()]
        _confusion_matrix_flow_count = confusion_matrix(y_test, predictions)
        matrix = utils_dl.getClassificationReport(_confusion_matrix=_confusion_matrix_flow_count, traffic_classes=labels)
        # matrix.loc['average'] = matrix[['recall', 'precision', 'weighted_precision', 'f1_score']].mean(axis=0).round(decimals=2)
        # report_average = matrix.iloc[[-1], [-4, -3, -2, -1]]
        # print(f'\n{report_average}')

        nl = '\n'
        # click.echo(f"{nl}Classification Summary Report{nl}{'=' * 29}{nl}{report_average}{nl}")
        matrix.to_csv(f"{PROJECT_HOME}/results/{_model}_cm.csv")
        click.echo(f"{nl}Confusion Matrix Flow Count Based{nl}{'=' * 33}{nl}{matrix}{nl}")

        
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """ Model Prediction"""
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
        dataset = pd.read_parquet(train_file)
        """Load Trained Model Model"""
        loaded_model = load_model(f"{PROJECT_HOME}/models/{MODEL_H5}")
        
        """Create test dataset"""
        prediction_dataset = utils_dl.create_prediction_dataset_tf(dataset=dataset, params=params)
        
        """predict test data using loaded model"""
        predictions = loaded_model.predict(prediction_dataset.batch(64))
        predictions = predictions.argmax(axis=1)
        
        """Retrofit predicted labels with original test dataset"""
        labels = params['labels']
        labels = [labels[i] for i in np.unique(predictions).tolist()]
        label_dict = dict([(i, x) for i, x in enumerate(labels)])

        predictions = np.vectorize(label_dict.get)(predictions).tolist()
        dataset['label'] = predictions
        dataset.to_csv(f"{PROJECT_HOME}/results/{MODEL_CSV}")  
    
    
    def test_model(self, trained_model_file, config, test_file, output_file):
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """ Model Evaluation"""
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # python3 src/cli.py dl test /home/mpaul/projects/enta_dl/enta_workspace/models/lstm_model_crypto.h5 /home/mpaul/projects/mpaul/mai/configs/dl/crypto_lstm_70files_config.json  /home/mpaul/projects/mpaul/mai/data/bitcoin-capture5.parquet cross_data.csv
        with open(config) as f:
            params = json.load(f)


        """Load Trained Model Model"""
        loaded_model = load_model(trained_model_file)
        
        """Create test dataset"""
        test_dataset = utils_dl.create_train_test_dataset_tf(data_file=test_file, params=params, train=False, evaluation=True)
        test_dataset = test_dataset.batch(128)
        
        """predict test data using loaded model"""
        y_test = np.concatenate([y for _, y in test_dataset], axis=0).argmax(axis=1)
        predictions = loaded_model.predict(test_dataset)
        predictions = predictions.argmax(axis=1)
        
        """Confusion Matrix"""
        labels = params['labels']
        labels = [labels[i] for i in np.unique(predictions).tolist()]
        _confusion_matrix_flow_count = confusion_matrix(y_test, predictions)
        matrix = utils_dl.getClassificationReport(_confusion_matrix=_confusion_matrix_flow_count, traffic_classes=labels)
        # matrix.loc['average'] = matrix[['recall', 'precision', 'weighted_precision', 'f1_score']].mean(axis=0).round(decimals=2)
        # report_average = matrix.iloc[[-1], [-4, -3, -2, -1]]
        # print(f'\n{report_average}')

        nl = '\n'
        # click.echo(f"{nl}Classification Summary Report{nl}{'=' * 29}{nl}{report_average}{nl}")
        matrix.to_csv(output_file)
        click.echo(f"{nl}Confusion Matrix Flow Count Based{nl}{'=' * 33}{nl}{matrix}{nl}")
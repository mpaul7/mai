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
from tensorflow.keras.layers import Dense, Flatten, Reshape
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
        
        
        
        # ================================
        models = params['models']
        # print(f'models -> {models}')
        # model_type = ''
        # input_branches = {'inputs': [], 'layer': []}
        # for model_type in models:
        #     model_type = model_type.strip()
        #     if model_type == 'lstm':
        #         input_branches['inputs'], input_branches['layer'] = utils_dl.create_dl_model_lstm(params, output_units)
        #         print(f'input_branches -> {input_branches}')
                
        #     elif model_type == 'mlp':
        #         input_branches['inputs'], input_branches['layer'] = utils_dl.create_dl_model_mlp(params, output_units)
        #         print(f'\n\ninput_branches -> {input_branches}')
        #     elif model_type == 'cnn':
        #         input_branches['inputs'], input_branches['layer'] = utils_dl.create_dl_model_cnn(params, output_units)
        #         print(f'\n\ninput_branches -> {input_branches}')
                
                
        # if len(params["models"]) > 1:
        #     x = layers.concatenate(input_branches.get('layer'))
        # else:
        #     x = input_branches['layer'][0]

        # x = layers.Dropout(params["encoder_dense_dropout_rate"])(x)

        # for i in range(params["num_encoder_dense"]):
        #     x = layers.Dense(units=params["encoder_dense_units_list"][i])(x)
        #     # x = layers.BatchNormalization()(x)
        #     x = layers.LeakyReLU()(x)

        # for i in range(params["dense_layers"]):
        #     x = layers.Dense(units=params["dense_layers_list"][i], kernel_initializer=params['initializer'])(x)
        #     # x = layers.BatchNormalization()(x)
        #     x = layers.LeakyReLU()(x)
        #     x = layers.Dropout(params["dense_layer_dropout_rate"])(x)

        # outputs = layers.Dense(7, activation='softmax', name='softmax')(x)
        # """Show model architecture"""
        # # tf.keras.utils.plot_model(_model, f"{PROJECT_HOME}/models/{MODEL_PNG}", show_shapes=True)
        # model = models.Model(inputs=input_branches.get('inputs'), outputs=outputs)
        
        # ================================
        # from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Flatten, Concatenate, Softmax
        
        for model_type in params["model_types"]:
            model_type = model_type.strip()
            if model_type == 'lstm':
                lstm_train, lstm_val = utils_dl.create_train_test_dataset_tf(model_type=model_type, data_file=train_file, params=params, train=True, evaluation=False)
                lstm_train = lstm_train.batch(params['train_batch_size'])
                lstm_val = lstm_val.batch(params['test_batch_size'])
            elif model_type == 'mlp':
                mlp_train, mlp_val = utils_dl.create_train_test_dataset_tf(model_type=model_type, data_file=train_file, params=params, train=True, evaluation=False)
                mlp_train = mlp_train.batch(params['train_batch_size'])
                mlp_val = mlp_val.batch(params['test_batch_size'])
            elif model_type == 'cnn':
                cnn_train, cnn_val = utils_dl.create_train_test_dataset_tf(model_type=model_type, data_file=train_file, params=params, train=True, evaluation=False)
                cnn_train = cnn_train.batch(params['train_batch_size'])
                cnn_val = cnn_val.batch(params['test_batch_size'])
        
        input_data = {
            "MLP_Input": mlp_train,
            "LSTM_Input": lstm_train,
            "CNN_Input": cnn_train
        }
        
        # from tensorflow.keras.models import Model

        # Define input shapes
        mlp_input_shape = (len(params['features']),)  # Adjust len(statistical_features)
        lstm_input_shape = (30, 3)  # Define sequence length
        cnn_input_shape = (30, 3)  # Same as LSTM for packets

        # MLP Model
        mlp_input = Input(shape=(1,), name="MLP_Input")
        mlp_hidden = Dense(64, activation='relu')(mlp_input)
        mlp_output = Dense(32, activation='relu')(mlp_hidden)
        mlp_output = Reshape((32,))(mlp_output) 

        # LSTM Model
        lstm_input = Input(shape=(30, 3), name="LSTM_Input")
        lstm_hidden = LSTM(64, return_sequences=False)(lstm_input)

        # CNN Model
        cnn_input = Input(shape=(30, 3), name="CNN_Input")
        cnn_hidden = layers.Conv1D(64, kernel_size=3, activation='relu')(cnn_input)
        cnn_flatten = Flatten()(cnn_hidden)

        # Concatenate Layers
        concatenated = layers.Concatenate()([mlp_output, lstm_hidden, cnn_flatten])

        # Final Dense Layer
        dense_hidden = Dense(64, activation='relu')(concatenated)

        # Output Layer
        output = Dense(7, activation='softmax', name="Output")(dense_hidden)

        # Build the Model
        model = Model(
            inputs={
                "MLP_Input": mlp_input,
                "LSTM_Input": lstm_input,
                "CNN_Input": cnn_input
            },
            outputs=output
        )

        # Compile the Model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Summary
        model.summary()

        # ================================
        # model.summary()
        """Save model architecture in Json config file"""
        # model_json = _model.to_json()
        # with open(f"{PROJECT_HOME}/models/{MODEL_JSON}", "w") as f:
        #     json.dump(model_json, f)

        """ Load model architecture from Json config file"""
        
        # with open(f"{PROJECT_HOME}/models/{MODEL_JSON}") as f:
        #     model_arch = json.load(f)
        # model = tf.keras.models.model_from_json(model_arch)
        
        """Show loaded model architecture"""
        # tf.keras.utils.plot_model(model, f"{PROJECT_HOME}/models/{MODEL_PNG}", show_shapes=True)
        # model.summary()
        
        """Compile Model"""
        metrics = ['accuracy']
        losses = CategoricalCrossentropy()
        learning_rate = params['learning_rate']
        adam = Adam(learning_rate=learning_rate)
        
        model.compile(loss=losses, optimizer=adam, metrics=metrics)
        
        """Train Model"""
        
        # Create separate datasets for each input type
        lstm_train = None
        lstm_val = None
        mlp_train = None
        mlp_val = None
        cnn_train = None
        cnn_val = None

        for model_type in params["model_types"]:
            model_type = model_type.strip()     
            if model_type == 'lstm':
                lstm_train, lstm_val = utils_dl.create_train_test_dataset_tf(model_type=model_type, data_file=train_file, params=params, train=True, evaluation=False)
                lstm_train = lstm_train.batch(params['train_batch_size'])
                lstm_val = lstm_val.batch(params['test_batch_size'])
            elif model_type == 'mlp':
                mlp_train, mlp_val = utils_dl.create_train_test_dataset_tf(model_type=model_type, data_file=train_file, params=params, train=True, evaluation=False)
                mlp_train = mlp_train.batch(params['train_batch_size'])
                mlp_val = mlp_val.batch(params['test_batch_size'])
            elif model_type == 'cnn':
                cnn_train, cnn_val = utils_dl.create_train_test_dataset_tf(model_type=model_type, data_file=train_file, params=params, train=True, evaluation=False)
                cnn_train = cnn_train.batch(params['train_batch_size'])
                cnn_val = cnn_val.batch(params['test_batch_size'])

        # First, let's add some debug prints
        # for model_type in params["model_types"]:
        #     model_type = model_type.strip()     
        #     if model_type == 'lstm':
        #         lstm_train, lstm_val = utils_dl.create_train_test_dataset_tf(model_type=model_type, data_file=train_file, params=params, train=True, evaluation=False)
        #         # Debug print
        #         for data in lstm_train.take(1):
        #             features, labels = data
        #             print(f"LSTM train data shape: {features}, label shape: {labels}")
        #     elif model_type == 'mlp':
        #         mlp_train, mlp_val = utils_dl.create_train_test_dataset_tf(model_type=model_type, data_file=train_file, params=params, train=True, evaluation=False)
        #         # Debug print
        #         for data in mlp_train.take(1):
        #             features, labels = data
        #             print(f"MLP train data shape: {features}, label shape: {labels}")
        #     elif model_type == 'cnn':
        #         cnn_train, cnn_val = utils_dl.create_train_test_dataset_tf(model_type=model_type, data_file=train_file, params=params, train=True, evaluation=False)
        #         # Debug print
        #         for data in cnn_train.take(1):
        #             features, labels = data
        #             print(f"CNN train data shape: {features}, label shape: {labels}")

        # Try a different approach to create the combined dataset
        def prepare_input(x, y):
            # Extract only the required features for each model type
            return {
                "MLP_Input": x["mlp_features"],  # Adjust key names based on your utils_dl.create_train_test_dataset_tf output
                "LSTM_Input": x["lstm_features"],
                "CNN_Input": x["cnn_features"]
            }, y

        train_dataset = tf.data.Dataset.zip((
            mlp_train,
            lstm_train,
            cnn_train
        )).map(lambda mlp, lstm, cnn: ({
            "MLP_Input": mlp[0],
            "LSTM_Input": lstm[0],
            "CNN_Input": cnn[0]
        }, mlp[1]))  # Using mlp labels since they should all have the same labels

        val_dataset = tf.data.Dataset.zip((
            mlp_val,
            lstm_val,
            cnn_val
        )).map(lambda mlp, lstm, cnn: ({
            "MLP_Input": mlp[0],
            "LSTM_Input": lstm[0],
            "CNN_Input": cnn[0]
        }, mlp[1]))

        # Apply batching
        train_dataset = train_dataset.batch(params['train_batch_size'])
        val_dataset = val_dataset.batch(params['test_batch_size'])

        # Set training parameters
        options = {
                    'epochs': params['epochs'],
                    'steps_per_epoch': params['steps_per_epoch'],
                    'validation_data': val_dataset,
                    'callbacks': []
                }
         
        log_dir = f"{PROJECT_HOME}/{params['log_dir']}"
        if log_dir:
            options['callbacks'].append(TensorBoard(log_dir=log_dir, histogram_freq=1))
            
        if params['early_stopping']:
            options['callbacks'].append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7))
        
        """ Train Model """
        model.fit(
            train_dataset,
            epochs=params['epochs'],
            steps_per_epoch=params['steps_per_epoch'],
            validation_data=val_dataset,
            callbacks=options['callbacks']
        )
        
        """ Model Evaluation"""
        score = model.evaluate(val_dataset , verbose=0)
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
        test_dataset = utils_dl.create_train_test_dataset_tf(model_type=model_type, data_file=test_file, params=params, train=False, evaluation=True)
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

        
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        # """ Model Prediction"""
        # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        
        # dataset = pd.read_parquet(train_file)
        # """Load Trained Model Model"""
        # loaded_model = load_model(f"{PROJECT_HOME}/models/{MODEL_H5}")
        
        # """Create test dataset"""
        # prediction_dataset = utils_dl.create_prediction_dataset_tf(model_type=model_type, dataset=dataset, params=params)
        
        # """predict test data using loaded model"""
        # predictions = loaded_model.predict(prediction_dataset.batch(64))
        # predictions = predictions.argmax(axis=1)
        
        # """Retrofit predicted labels with original test dataset"""
        # labels = params['labels']
        # labels = [labels[i] for i in np.unique(predictions).tolist()]
        # label_dict = dict([(i, x) for i, x in enumerate(labels)])

        # predictions = np.vectorize(label_dict.get)(predictions).tolist()
        # dataset['label'] = predictions
        # dataset.to_csv(f"{PROJECT_HOME}/results/{MODEL_CSV}")  
    
    
    # def test_model(self, trained_model_file, config, test_file, output_file):
    #     """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #     """ Model Evaluation"""
    #     """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #     # python3 src/cli.py dl test /home/mpaul/projects/enta_dl/enta_workspace/models/lstm_model_crypto.h5 /home/mpaul/projects/mpaul/mai/configs/dl/crypto_lstm_70files_config.json  /home/mpaul/projects/mpaul/mai/data/bitcoin-capture5.parquet cross_data.csv
    #     with open(config) as f:
    #         params = json.load(f)


    #     """Load Trained Model Model"""
    #     loaded_model = load_model(trained_model_file)
        
    #     """Create test dataset"""
    #     test_dataset = utils_dl.create_train_test_dataset_tf(data_file=test_file, params=params, train=False, evaluation=True)
    #     test_dataset = test_dataset.batch(128)
        
    #     """predict test data using loaded model"""
    #     y_test = np.concatenate([y for _, y in test_dataset], axis=0).argmax(axis=1)
    #     predictions = loaded_model.predict(test_dataset)
    #     predictions = predictions.argmax(axis=1)
        
    #     """Confusion Matrix"""
    #     labels = params['labels']
    #     labels = [labels[i] for i in np.unique(predictions).tolist()]
    #     _confusion_matrix_flow_count = confusion_matrix(y_test, predictions)
    #     matrix = utils_dl.getClassificationReport(_confusion_matrix=_confusion_matrix_flow_count, traffic_classes=labels)
    #     # matrix.loc['average'] = matrix[['recall', 'precision', 'weighted_precision', 'f1_score']].mean(axis=0).round(decimals=2)
    #     # report_average = matrix.iloc[[-1], [-4, -3, -2, -1]]
    #     # print(f'\n{report_average}')

    #     nl = '\n'
    #     # click.echo(f"{nl}Classification Summary Report{nl}{'=' * 29}{nl}{report_average}{nl}")
    #     matrix.to_csv(output_file)
    #     click.echo(f"{nl}Confusion Matrix Flow Count Based{nl}{'=' * 33}{nl}{matrix}{nl}")
from lib2to3.pgen2.tokenize import printtoken

import click
import numpy as np
import pandas as pd
import tensorflow.data as tfd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf


"""Create Keras DL model - CNN"""
def create_dl_model_cnn(params, output_units):
    """Create input layers for packet sequence data """
    inputs = {name: layers.Input(shape=(params['sequence_length'],), dtype=tf.float32, name=name) for name in params['seq_packet_feature_cnn']}

    """Stack input layers"""
    pktseq_x = tf.stack(list(inputs.values()), axis=2)
    # pktseq_x = layers.Reshape(target_shape=(params['sequence_length'], 1))(list(inputs.values())[-1])

    pktseq_x = layers.Conv1D(200, kernel_size=7, strides=1, padding='same', input_shape=(None, 3))(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(200, kernel_size=5, strides=1, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(200, kernel_size=5, strides=1, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(200, kernel_size=5, strides=1, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(300, kernel_size=5, strides=1, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(300, kernel_size=5, strides=1, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(300, kernel_size=4, strides=2, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)

    pktseq_x = layers.GlobalAveragePooling1D()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(0.5)(pktseq_x)

    """Output layer"""
    # outputs = layers.Dense(output_units, activation='softmax', name='softmax')(pktseq_x)
    # model = models.Model(inputs=[inputs], outputs=outputs)

    return inputs, pktseq_x

"""Create Keras DL model - LSTM"""
def create_dl_model_lstm(params, output_units):
    
    """Create input layers for packet sequence data """
    inputs = {name: layers.Input(shape=(params['sequence_length'],), dtype=tf.float32, name=name)for name in params['seq_packet_feature']}
    # print(inputs)
    
    """Stack input layers"""
    pktseq_x1 = tf.stack(list(inputs.values()), axis=2)
    pktseq_x2 = layers.Reshape(target_shape=(params['sequence_length'], 1))(list(inputs.values())[-1])
    # pktseq_x = layers.Concatenate(axis=-1)([pktseq_x1, pktseq_x2])
    
    """LSTM units layer"""
    lstm = layers.LSTM(units=128, input_shape=(params['sequence_length'], 3), recurrent_dropout=params['dropout_rate'])(pktseq_x1)
    
    
    """Create chain of Dense layers"""
    x = layers.Dropout(0.1)(lstm)
    x = layers.Dense(units=128)(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(units=32)(x)
    
    """Output layer"""
    # outputs = layers.Dense(output_units, activation='softmax', name='softmax')(x)
    # model = models.Model(inputs=[inputs], outputs=outputs)
    
    return inputs, x

"""Create Keras DL model - MLP"""
def create_dl_model_mlp(params, output_units):
    
    """Create input layers for packet sequence data """
    inputs = {name: layers.Input(shape=(1,), dtype=tf.float32, name=name)for name in params['features']}

    """Stack input layers"""
    x = layers.Concatenate(axis=-1)(list(inputs.values()))

    # flow_x = layers.Reshape(target_shape=(len(params['features']),))(flow_x)  
    
    """LSTM units layer"""
    # lstm = layers.LSTM(units=128, recurrent_dropout=params['dropout_rate'])(inputs_stack)
    
    """Create chain of Dense layers"""
    # x = layers.Dense(units=128)(flow_x)
    # x = layers.Dropout(0.1)(x)
    # x = layers.Dense(units=32)(x)
    
    x = layers.Dense(units=32)(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(units=32)(x)
    # x = layers.LeakyReLU()(x)
    # x = layers.Dense(units=32)(x)
    # x = layers.LeakyReLU()(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.Dense(units=8, kernel_regularizer='l2')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.LeakyReLU()(x)
    
    """Output layer"""
    # outputs = layers.Dense(units=output_units, activation='softmax', name='softmax')(x)
    # model = models.Model(inputs=[inputs], outputs=outputs)
    
    return inputs, x

def create_train_test_dataset_tf(model_type=None, data_file=None, params=None, train=None, evaluation=None):
    df = pd.read_parquet(data_file)
    model_type = params['model_types']
    features = []
    if 'mlp' in model_type:
        features.extend(params['features'])
    if 'lstm' in model_type:
        features.extend(params['seq_packet_feature'])
    if 'cnn' in model_type:
        features.extend(params['seq_packet_feature_cnn'])
        df['splt_direction_cnn'] = df['splt_direction']
        df['splt_ps_cnn'] = df['splt_ps'] 
        df['splt_piat_cnn'] = df['splt_piat']
    # Add target column to features
    features.extend([params['target_column']])

    X = df.loc[:, features]
    _y = df.loc[:, [params['target_column']]]
    y = pd.get_dummies(_y)
    
    """ Create tf dataset """  
    def create_dataset(X, y, features):
        feat_dict = {}
        if 'mlp' in model_type:
            X_flow = {name: np.stack(value) for name, value in X.loc[:, params['features']].items()}
            feat_dict['flow_features'] = X_flow
        if 'lstm' in model_type:
            X_pktseq = {name: np.stack(value) for name, value in X.loc[:, params['seq_packet_feature']].items()}
            feat_dict['pktseq_features'] = X_pktseq
        if 'cnn' in model_type:
            X_pktseq = {name: np.stack(value) for name, value in X.loc[:, params['seq_packet_feature_cnn']].items()}
            feat_dict['pktseq_features_cnn'] = X_pktseq

        
        ds_X = tf.data.Dataset.from_tensor_slices(feat_dict, name='X')
        ds_y = tf.data.Dataset.from_tensor_slices(y)
        tf_dataset = tf.data.Dataset.zip((ds_X, ds_y))
        return tf_dataset

    def _create_balanced_tf_dataset(X, y, features, params):
        # Creates a DataFrame with a single column containing the predicted class labels
        # by finding the index of the maximum value in each row of one-hot encoded y
        df = pd.DataFrame({params['target_column']: y.idxmax(axis=1)})
        partials = []
        for _, group in df.groupby(params['target_column']):
            partials.append(create_dataset(X.loc[group.index], y.loc[group.index], features).repeat())
        return tfd.Dataset.sample_from_datasets(partials)

    if train:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['validation_percent'])
        training_dataset = _create_balanced_tf_dataset(X_train, y_train, features, params)
        # training_dataset = create_dataset(X_train, y_train, features)
        validation_dataset = create_dataset(X_test, y_test, features)
        return training_dataset, validation_dataset
    elif evaluation: 
        test_dataset = create_dataset(X, y, features)
        return test_dataset
        
def create_prediction_dataset_tf(model_type=None, dataset=None, params=None):
    # dataset = pd.read_parquet(data_file)
    if model_type == 'lstm':
        features = params['seq_packet_feature']
    elif model_type == 'mlp':
        features = params['features']
    elif model_type == 'cnn':
        features = params['seq_packet_feature']
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    X = dataset[features]
    
    """ Create tf dataset """  
    def create_dataset(X, features):
        feat_dict = {}
        X_pktseq = {name: np.stack(value) for name, value in X.loc[:, features].items()}
        feat_dict['features'] = X_pktseq
        ds_X = tf.data.Dataset.from_tensor_slices(feat_dict, name='X')
        tf_dataset = tf.data.Dataset.zip((ds_X))
        return tf_dataset

    prediction_dataset = create_dataset(X, features)
    return prediction_dataset

def create_data(file_name, label_encoder, params):
    dataset = pd.read_parquet(file_name)
    features = params['seq_packet_feature'] + params['encoded_label']
    # dataset = pd.read_orc(file_name)
    dataset = dataset.loc[dataset[params['target_column']].isin(params['labels'])]
    dataset['encoded_label'] = label_encoder.fit_transform(dataset[params['target_column']])
    df = dataset[features]
    return df

def display_report(cm=None, type=None):
    _cm = cm.copy()
    _cm.loc['average'] = _cm[['recall', 'precision', 'weighted_precision', 'f1_score']].mean(axis=0).round(decimals=2)
    report_average = _cm.iloc[[-1], [-4, -3,-2, -1]]
    nl = '\n'
    click.echo(f"{nl}Evaluation Classification Summary Report{nl}{'=' * 41}{nl}{report_average}{nl}")
    click.echo(f"{nl}Evaluation Confusion Matrix{nl}{'='*28}{nl}{cm}{nl}")

def getClassificationReport(_confusion_matrix=None, traffic_classes=None, byte_count=False):
    total_flows_bytes = np.sum(_confusion_matrix, axis=1)
    recall = np.diag(_confusion_matrix) / np.sum(_confusion_matrix, axis=1)
    precision = np.diag(_confusion_matrix) / np.sum(_confusion_matrix, axis=0)
    _weighted_precision = [[a * b for a, b in zip(*l)] for l in zip(np.array(_confusion_matrix).T.tolist(),
                                                                    [[x / y for y in total_flows_bytes] for x in
                                                                    total_flows_bytes])]
    weighted_precision = np.diag(np.array(_confusion_matrix).T.tolist()) / [sum(l) for l in _weighted_precision]
    f1_Score = [2 * (a * b) / (a + b) for a, b in zip(weighted_precision, recall)]
    df_conf_matrix = pd.DataFrame(_confusion_matrix, columns=traffic_classes, index=traffic_classes)

    if byte_count:
        df_conf_matrix['total_bytes'] = total_flows_bytes
    else:
        df_conf_matrix['total_flows'] = total_flows_bytes
    df_conf_matrix['recall'] = np.round(recall * 100, 2)
    df_conf_matrix['precision'] = np.round(precision * 100, 2)

    df_conf_matrix['weighted_precision'] = np.round(weighted_precision * 100, 2)
    df_conf_matrix["f1_score"] = np.round(np.array(f1_Score) * 100, 2)
    return df_conf_matrix
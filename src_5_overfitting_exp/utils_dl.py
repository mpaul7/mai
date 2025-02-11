from lib2to3.pgen2.tokenize import printtoken

import click
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.data as tfd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers


KERAS_INITIALIZER = {
    'default': tf.keras.initializers.GlorotUniform,
    'he_uniform': tf.keras.initializers.HeUniform,
    'he_normal': tf.keras.initializers.HeNormal
}

def create_dl_model_cnn_xx(params):

    
    """Create input layers for packet sequence data """
    print("using cnn_v2")
    # inputs = {name: layers.Input(shape=(params['sequence_length'],), dtype=tf.float32, name=name) for name in params['seq_packet_feature']}
    # inputs = {name: layers.Input(shape=(3, 1), dtype=tf.float32, name=name) for name in params['cnn_statistical']}
    # print(inputs)
    # layers.Input(shape=input_shape)
    # print(inputs)
    """Stack input layers"""
    # pktseq_x = tf.stack(list(inputs.values()), axis=2)
    
    
    inputs = {name: layers.Input(shape=(3,), dtype=tf.float32, name=name)for name in params['cnn_statistical']}
    # print(inputs)
    
    """Stack input layers"""
    pktseq_x = tf.stack(list(inputs.values()), axis=2)
    # pktseq_x = layers.Concatenate(axis=-1)(list(inputs.values()))
    # pktseq_x = layers.Reshape(target_shape=(params['sequence_length'], 1))(list(inputs.values())[-1])
    # pktseq_x = layers.Reshape((3, 1))(list(inputs.values())[-1])  # Reshape to (3, 1)
    # x = layers.Reshape((3, 1))(inputs)  # Reshape to (57, 1)
    pktseq_x = layers.Conv1D(100, kernel_size=7, strides=1, padding='same', input_shape=(3,1))(pktseq_x)
    # pktseq_x = layers.Conv1D(200, kernel_size=7, strides=1, padding='same', input_shape=(None, 1))(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(100, kernel_size=5, strides=1, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(100, kernel_size=5, strides=1, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(50, kernel_size=2, strides=1, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(50, kernel_size=1, strides=1, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    # pktseq_x = layers.Conv1D(100, kernel_size=2, strides=1, padding='valid')(pktseq_x)
    # pktseq_x = layers.ReLU()(pktseq_x)
    # pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    # pktseq_x = layers.Conv1D(100, kernel_size=1, strides=2, padding='valid')(pktseq_x)
    # pktseq_x = layers.ReLU()(pktseq_x)

    # pktseq_x = layers.GlobalAveragePooling1D()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(0.5)(pktseq_x)

    """Output layer"""
    # outputs = layers.Dense(output_units, activation='softmax', name='softmax')(pktseq_x)
    # model = models.Model(inputs=[inputs], outputs=outputs)

    return inputs, pktseq_x

"""Create Keras DL model - CNN"""
def create_dl_model_cnn_v2(params):

    """Create input layers for packet sequence data """
    inputs = {name: layers.Input(shape=(params['stat_length'],), dtype=tf.float32, name=name) for name in params['stat_packet_feature']}
    # inputs = {name: layers.Input(shape=(150,), dtype=tf.float32, name=name) for name in params['seq_packet_feature']}
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

    pktseq_x = layers.Conv1D(200, kernel_size=4, strides=1, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(200, kernel_size=3, strides=1, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    # pktseq_x = layers.Conv1D(200, kernel_size=3, strides=1, padding='valid')(pktseq_x)
    # pktseq_x = layers.ReLU()(pktseq_x)
    # pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    # pktseq_x = layers.Conv1D(200, kernel_size=2, strides=2, padding='valid')(pktseq_x)
    # pktseq_x = layers.ReLU()(pktseq_x)

    pktseq_x = layers.GlobalAveragePooling1D()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(0.5)(pktseq_x)

    """Output layer"""
    # outputs = layers.Dense(output_units, activation='softmax', name='softmax')(pktseq_x)
    # model = models.Model(inputs=[inputs], outputs=outputs)

    return inputs, pktseq_x

def create_dl_model_cnn(params):

    if params['regularizer'] == 'l1':
            regularizer = tf.keras.regularizers.L1(params['regularizer_value'])
    elif params['regularizer'] == 'l2':
            regularizer = tf.keras.regularizers.L2(params['regularizer_value'])
    else:
            regularizer = None

    """Create input layers for packet sequence data """
    # if params['cnn_feature_type'] == 'sequence':
    #     inputs = {name: layers.Input(shape=(params['sequence_length'],), dtype=tf.float32, name=name) for name in params['seq_packet_feature']}
    # elif params['cnn_feature_type'] == 'statistical':
    #     inputs = {name: layers.Input(shape=(params['cnn_stat_feature_length'],), dtype=tf.float32, name=name) for name in params['cnn_stat_feature']}
    # elif params['cnn_feature_type'] == 'packet_bytes':
    #     inputs = {name: layers.Input(shape=(params['cnn_byte_feature_length'],), dtype=tf.float32, name=name) for name in params['cnn_byte_feature']}
    inputs = {name: layers.Input(shape=(params['cnn_stat_feature_length'],), dtype=tf.float32, name=name) for name in params['cnn_stat_feature']}
    # inputs = {name: layers.Input(shape=(150,), dtype=tf.float32, name=name) for name in params['seq_packet_feature']}
    """Stack input layers"""
    pktseq_x = tf.stack(list(inputs.values()), axis=2)
    # pktseq_x = layers.Reshape(target_shape=(params['sequence_length'], 1))(list(inputs.values())[-1])

    pktseq_x = layers.Conv1D(200, kernel_size=7, strides=1, kernel_regularizer=regularizer,  padding='same', input_shape=(None, 3))(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)

    pktseq_x = layers.Conv1D(200, kernel_size=5, strides=1, kernel_regularizer=regularizer, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)

    pktseq_x = layers.Conv1D(200, kernel_size=5, strides=1, kernel_regularizer=regularizer, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(200, kernel_size=4, strides=1, kernel_regularizer=regularizer, padding='same')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)

    pktseq_x = layers.Conv1D(300, kernel_size=3, strides=1, kernel_regularizer=regularizer, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)

    pktseq_x = layers.Conv1D(300, kernel_size=3, strides=1, kernel_regularizer=regularizer, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)

    pktseq_x = layers.Conv1D(300, kernel_size=2, strides=2, kernel_regularizer=regularizer, padding='valid')(pktseq_x)
    pktseq_x = layers.ReLU()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(params['dropout_rate'])(pktseq_x)

    pktseq_x = layers.GlobalAveragePooling1D()(pktseq_x)
    pktseq_x = layers.BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9, center=True, scale=True)(pktseq_x)
    pktseq_x = layers.Dropout(0.5)(pktseq_x)

    """Output layer"""
    # outputs = layers.Dense(output_units, activation='softmax', name='softmax')(pktseq_x)
    # model = models.Model(inputs=[inputs], outputs=outputs)

    return inputs, pktseq_x


"""Create Keras DL model - LSTM"""
def create_dl_model_lstm(params):
    
    """Create input layers for packet sequence data """
    inputs = {name: layers.Input(shape=(params['sequence_length'],), dtype=tf.float32, name=name)for name in params['seq_packet_feature']}
    
    """Stack input layers"""
    pktseq_x1 = tf.stack(list(inputs.values()), axis=2)
    pktseq_x2 = layers.Reshape(target_shape=(params['sequence_length'], 1))(list(inputs.values())[-1])
    # pktseq_x = layers.Concatenate(axis=-1)([pktseq_x1, pktseq_x2])
    
    """LSTM units layer"""
    if params['lstm']['num_lstm'] == 1:
        if params['regularizer'] == 'l1':
            regularizer = tf.keras.regularizers.L1(params['regularizer_value'])
        elif params['regularizer'] == 'l2':
            regularizer = tf.keras.regularizers.L2(params['regularizer_value'])
        else:
            regularizer = None
        lstm = layers.LSTM(units=params['lstm']['lstm_units'], 
                            input_shape=(params['sequence_length'], 3), 
                            recurrent_dropout=params['dropout_rate'],   
                            kernel_regularizer=regularizer,
                            # recurrent_regularizer=regularizer,
                            name=f'lstm_1'
                        )(pktseq_x1)
    if params['lstm']['num_lstm'] == 2:
        if params['regularizer'] == 'l1':
            regularizer = tf.keras.regularizers.L1(params['regularizer_value'])
        elif params['regularizer'] == 'l2':
            regularizer = tf.keras.regularizers.L2(params['regularizer_value'])
        else:
            regularizer = None
        lstm = layers.LSTM(params['lstm']['lstm_units'], 
                                input_shape=(params['sequence_length'], 3),
                                return_sequences=True,
                                recurrent_dropout=params['dropout_rate'],
                                kernel_regularizer=regularizer,
                                # recurrent_regularizer=regularizer,
                                name=f'lstm_1'
                                )(pktseq_x1)
        lstm = layers.LSTM(params['lstm']['lstm_units'],    
                                input_shape=(params['sequence_length'], 3),
                                go_backwards=True, 
                                recurrent_dropout=params['dropout_rate'],
                                kernel_regularizer=regularizer,
                                # recurrent_regularizer=regularizer,
                                name=f'lstm_2'
                                )(lstm)
    """Create chain of Dense layers"""
    for i in range(params['lstm']['num_lstm_dense']): # 2
        lstm = layers.Dense(units=params['lstm']['lstm_dense_units_list'][i], 
                            kernel_initializer=KERAS_INITIALIZER[params['initializer']],
                            name=f'lstm_dense_{i}'
                            )(lstm)
        lstm = layers.LeakyReLU(name=f'lstm_leaky_relu_{i}')(lstm)
        lstm = layers.Dropout(params['dropout_rate'], name=f'lstm_dropout_{i}')(lstm)
    
    """Output layer"""
    # outputs = layers.Dense(output_units, activation='softmax', name='softmax')(x)
    # model = models.Model(inputs=[inputs], outputs=outputs)
    
    return inputs, lstm

"""Create Keras DL model - MLP"""
def create_dl_model_mlp(params):
    
    if params['regularizer'] == 'l1':
            regularizer = tf.keras.regularizers.L1(params['regularizer_value'])
    elif params['regularizer'] == 'l2':
            regularizer = tf.keras.regularizers.L2(params['regularizer_value'])
    else:
            regularizer = None
    
    """Create input layers for packet sequence data """
    inputs = {name: layers.Input(shape=(1,), dtype=tf.float32, name=name)for name in params['features']}

    """Stack input layers"""
    x = layers.Concatenate(axis=-1)(list(inputs.values()))
    
    """Create chain of Dense layers"""
    for i in range(params['mlp']['num_dense']):
        x = layers.Dense(units=params['mlp']['units_list'][i], kernel_regularizer=regularizer, kernel_initializer=KERAS_INITIALIZER[params['initializer']], name=f'dense_{i}')(x)
        x = layers.BatchNormalization(name=f'batch_norm_{i}')(x)
        x = layers.LeakyReLU(name=f'leaky_relu_{i}')(x)
        x = layers.Dropout(params['dropout_rate'], name=f'dropout_{i}')(x)
    
    """Output layer"""
    # outputs = layers.Dense(units=output_units, activation='softmax', name='softmax')(x)
    # model = models.Model(inputs=[inputs], outputs=outputs)
    
    return inputs, x

def create_train_test_dataset_tf(data_file=None, params=None, train=None, evaluation=None):
    df = pd.read_parquet(data_file)
    model_type = params['model_types']
    features = []
    if 'mlp' in model_type:
        # print("mlp")
        features.extend(params['features'])
    if 'lstm' in model_type:
        # print("lstm")
        features.extend(params['seq_packet_feature'])
    if 'cnn' in model_type:
        features.extend(params['cnn_stat_feature'])
        # if params['cnn_feature_type'] == "sequence":
        #     features.extend(params['seq_packet_feature'])
        # elif params['cnn_feature_type'] == "statistical":
        #     features.extend(params['cnn_stat_feature'])
        # elif params['cnn_feature_type'] == "packet_bytes":
        #     features.extend(params['cnn_byte_feature'])
        
    features.extend([params['target_column']])


    # Subset the dataframe based on the specified features and target column
    df_subset = df[features]
    
    # Subset the dataframe based on the values of the 'target_column'
    target_column_values = params['labels']
    df_subset = df_subset[df_subset[params['target_column']].isin(target_column_values)]

    X = df_subset.drop(columns=[params['target_column']])

    _y = df_subset.loc[:, [params['target_column']]]
    y = pd.get_dummies(_y)
    
    """ Create tf dataset """  
    
    def create_dataset(X, y, features):
        feat_dict = {}
        X_pktseq = {}
        if 'mlp' in model_type:
            X_flow = {name: np.stack(value) for name, value in X.loc[:, params['features']].items()}
            feat_dict['flow_features'] = X_flow
        if 'lstm' in model_type:
            X_pktseq = {name: np.stack(value) for name, value in X.loc[:, params['seq_packet_feature']].items()}
            feat_dict['pktseq_features'] = X_pktseq
        if 'cnn' in model_type:
            # print("cnn")
            # if params['cnn_feature_type'] == "sequence":
            #     print("sequence")
            #     X_pktseq = {name: np.stack(value) for name, value in X.loc[:, params['seq_packet_feature']].items()}
            #     feat_dict['cnn_features'] = X_pktseq
            # elif params['cnn_feature_type'] == "statistical":
            #     X_pktseq = {name: np.stack(value) for name, value in X.loc[:, params['cnn_stat_feature']].items()}
            #     feat_dict['cnn_features'] = X_pktseq
            # elif params['cnn_feature_type'] == "packet_bytes":
            #     X_pktseq = {name: np.stack(value) for name, value in X.loc[:, params['cnn_byte_feature']].items()}
            #     feat_dict['cnn_features'] = X_pktseq
                
            X_pktseq = {name: np.stack(value) for name, value in X.loc[:, params['cnn_stat_feature']].items()}
            feat_dict['pktstat_features'] = X_pktseq
        
        # if 'cnn' in model_type:
        #     # when the structure only contains cnn as input branch, i.e. the feature array only contains one feature
        #     # use the raw byte feature straight forward without forming a dictionary
        #     ds_X = tf.data.Dataset.from_tensor_slices(X_pktseq, name='X')
        # else:
        #     ds_X = tf.data.Dataset.from_tensor_slices(feat_dict, name='X')

        # if len(model_type) == 1 and model_type[0] == 'cnn':
        #     print("1")
        #     ds_X = tf.data.Dataset.from_tensor_slices(X_pktseq, name='X')
        # elif len(model_type) == 1 and model_type[0] != 'cnn':
        #     print("2")
        #     ds_X = tf.data.Dataset.from_tensor_slices(feat_dict, name='X')
        # elif len(model_type) > 1:
        #     print("3")
        #     ds_X = tf.data.Dataset.from_tensor_slices(feat_dict, name='X')
        # else:
        #     raise ValueError("Invalid model_type configuration.")
        # ds_X = tf.data.Dataset.from_tensor_slices(X_pktseq, name='X')
        ds_X = tf.data.Dataset.from_tensor_slices(feat_dict, name='X')
        ds_y = tf.data.Dataset.from_tensor_slices(y)
        tf_dataset = tf.data.Dataset.zip((ds_X, ds_y))
        return tf_dataset
        
        # ds_X = tf.data.Dataset.from_tensor_slices(X_pktseq, name='X')
        
    
    def create_dataset_original(X, y, features):

        feat_dict = {}

        if 'mlp' in model_type:
            X_flow = {name: np.stack(value) for name, value in X.loc[:, params['features']].items()}
            feat_dict['flow_features'] = X_flow

        if 'lstm' in model_type:
            if len(params['seq_packet_feature']) > 1:
                X_pktseq = {name: np.stack(value) for name, value in X.loc[:, params['seq_packet_feature']].items()}
                feat_dict['pktseq_features'] = X_pktseq

        if 'cnn' in model_type:
            X_pktseq = {name: np.stack(value) for name, value in X.loc[:, params['seq_packet_feature']].items()}
            feat_dict['pktseq_features'] = X_pktseq
        
        if 'cnn' in model_type:
            # when the structure only contains cnn as input branch, i.e. the feature array only contains one feature
            # use the raw byte feature straight forward without forming a dictionary
            ds_X = tf.data.Dataset.from_tensor_slices(X_pktseq, name='X')
        else:
            ds_X = tf.data.Dataset.from_tensor_slices(feat_dict, name='X')

        print(feat_dict, 6666)
        ds_X = tf.data.Dataset.from_tensor_slices(feat_dict, name='X')
        ds_y = tf.data.Dataset.from_tensor_slices(y)
        tf_dataset = tf.data.Dataset.zip((ds_X, ds_y))
        return tf_dataset

    def _create_balanced_tf_dataset(X, y, features, params):
        # Creates a DataFrame with a single column containing the predicted class labels
        # by finding the index of the maximum value in each row of one-hot encoded y
        df = pd.DataFrame({params['target_column']: y.idxmax(axis=1)})
        partials = []
        # print(df.groupby(params['target_column']).size())
        for _, group in df.groupby(params['target_column']):

            partials.append(create_dataset(X.loc[group.index], y.loc[group.index], features).repeat())
        # print(partials, 77777)
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
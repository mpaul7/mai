import numpy as np
import pandas as pd
import tensorflow as tf
import json
import time

from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow.data as tfd
import keras_tuner as kt
from keras_tuner import Objective, BayesianOptimization

from .common import DefaultAbstractModel, AbstractModel, load_model, get_classification_report
from .openset_threeinputs import ReplaceZerosLayer
from tw.cli.params import KERAS_INITIALIZER
import mlflow


class ThreeFeatureTypeModelFactoryMixin:
    def _create_keras_model(self, preprocessor_flow, preprocessor_pkt, params, dataset):
        initializer = KERAS_INITIALIZER.get(params.initializer)()
        input_branches = {'inputs': [],
                          'layer': []}

        if 'mlp' in params.structure:
            # Statistical inputs to multi-input model
            flow_inputs = {name: layers.Input(shape=(1,), dtype=tf.float32, name=name) for name in params.features}

            flow_x = layers.Concatenate(axis=-1)(list(flow_inputs.values()))
            flow_x = layers.Reshape(target_shape=(len(params.features),))(flow_x)

            flow_x = preprocessor_flow(flow_x)

            for i in range(params.num_dense):
                flow_x = layers.Dense(units=params.units_list[i], kernel_initializer=initializer)(flow_x)
                flow_x = layers.BatchNormalization()(flow_x)
                flow_x = layers.LeakyReLU()(flow_x)
                flow_x = layers.Dropout(params.dropout_rate)(flow_x)

            input_branches['inputs'].append(flow_inputs)
            input_branches['layer'].append(flow_x)

        if 'lstm' in params.structure:
            # Packet sequence inputs to multi-input model
            pktseq_inputs = {name: layers.Input(shape=(params.sequence_length,), dtype=tf.float32, name=name) for name in params.packet_feature_name}
            # normalize input of first two packet features
            pktseq_x1 = layers.Concatenate(axis=-1)(list(pktseq_inputs.values())[:2])
            pktseq_x1 = layers.Reshape(target_shape=(params.sequence_length, len(params.packet_feature_name[:2])))(
                pktseq_x1)
            if params.masking == True:
                n_values = tf.math.count_nonzero(pktseq_x1[:, :, 0])  # will return a scalar
                pktseq_x1 = tf.keras.layers.Masking(mask_value=0.0,
                                                    input_shape=(
                                                        params.sequence_length, len(params.packet_feature_name[:2])),
                                                    name="masking_1")(pktseq_x1)  # masking before norm
            pktseq_x1 = preprocessor_pkt(pktseq_x1)

            # Reshape the packet direction feature
            pktseq_x2 = layers.Reshape(target_shape=(params.sequence_length, 1))(list(pktseq_inputs.values())[-1])
            if params.masking == True:
                # Replace zeros from the nth element using -1, n is calculated using count_nonzero from the seq_iat feature
                pktseq_x2 = ReplaceZerosLayer()(pktseq_x2, n_values)
                # Apply masking to the input tensor, skip the zeros from the nth timestep without ignoring forward direction
                pktseq_x2 = tf.keras.layers.Masking(mask_value=-1.0, name="masking_2")(pktseq_x2)

            # concat normalized input with packet direction flags
            pktseq_x = layers.Concatenate(axis=-1)([pktseq_x1, pktseq_x2])

            # TODO: when num_lstm == 0, create a CNN layer instead
            if params.num_lstm == 0:
                pass
            if params.num_lstm == 1:
                pktseq_x = layers.LSTM(params.lstm_units, input_shape=(params.sequence_length, 3),
                                       recurrent_dropout=params.rdropout_rate)(pktseq_x)
            if params.num_lstm == 2:
                pktseq_x = layers.LSTM(params.lstm_units, input_shape=(params.sequence_length, 3),
                                       return_sequences=True,
                                       recurrent_dropout=params.rdropout_rate)(pktseq_x)
                pktseq_x = layers.LSTM(params.lstm_units, input_shape=(params.sequence_length, 3),
                                       go_backwards=True, recurrent_dropout=params.rdropout_rate)(pktseq_x)
            if params.num_lstm == 3:
                pktseq_x = layers.LSTM(params.lstm_units, input_shape=(params.sequence_length, 3),
                                       return_sequences=True,
                                       recurrent_dropout=params.rdropout_rate)(pktseq_x)
                pktseq_x = layers.LSTM(params.lstm_units, input_shape=(params.sequence_length, 3),
                                       return_sequences=True,
                                       go_backwards=True, recurrent_dropout=params.rdropout_rate)(pktseq_x)
                pktseq_x = layers.LSTM(params.lstm_units, input_shape=(params.sequence_length, 3),
                                       recurrent_dropout=params.rdropout_rate)(pktseq_x)

            for i in range(params.num_lstm_dense):
                pktseq_x = layers.Dense(units=params.lstm_dense_units_list[i], kernel_initializer=initializer)(pktseq_x)
                pktseq_x = layers.BatchNormalization()(pktseq_x)
                pktseq_x = layers.LeakyReLU()(pktseq_x)
                pktseq_x = layers.Dropout(params.lstm_dense_dropout_rate)(pktseq_x)

            input_branches['inputs'].append(pktseq_inputs)
            input_branches['layer'].append(pktseq_x)

        if 'cnn' in params.structure:
            # Raw byte inputs to multi-input model
            raw_byte_inputs = layers.Input(shape=(params.len_bytes,), dtype=tf.float32,
                                           name=params.rawbytes_feature_name[0])
            raw_x = layers.Reshape(target_shape=(params.len_bytes, 1))(raw_byte_inputs)

            raw_x = layers.Lambda(lambda x: x / 256)(raw_x)

            ## Initial cnn layers
            raw_x = layers.Conv1D(filters=params.num_filter, kernel_size=5, strides=1,
                                  padding='same', activation='relu', kernel_initializer=initializer,
                                  input_shape=(params.len_bytes, 1))(raw_x)
            raw_x = layers.Conv1D(filters=params.num_filter, kernel_size=5, strides=1,
                                  padding='same', activation='relu', kernel_initializer=initializer)(raw_x)
            raw_x = layers.MaxPooling1D(pool_size=2, strides=2)(raw_x)
            skip_x = raw_x = layers.Dropout(params.cnn_dropout_rate)(raw_x)

            ## Additional cnn layers
            # n_filters = params.num_filter/4
            for i in range(params.num_cnn):
                n_filters = params.num_filter * (i + 1) * 2
                raw_x = layers.Conv1D(filters=n_filters, kernel_size=5, strides=1, padding='same', activation='relu',
                                      kernel_initializer=initializer)(raw_x)
                raw_x = layers.Conv1D(filters=n_filters, kernel_size=5, strides=1, padding='same', activation='relu',
                                      kernel_initializer=initializer)(raw_x)

                raw_x = layers.Concatenate()([raw_x, skip_x])

                raw_x = layers.MaxPooling1D(pool_size=2, strides=2)(raw_x)
                skip_x = raw_x = layers.Dropout(params.cnn_dropout_rate)(raw_x)
                # n_filters = n_filters / 4

            raw_x = layers.Flatten()(raw_x)

            ## Dense layers after cnn
            if params.num_cnn_dense == 0:
                pass
            else:
                for i in range(params.num_cnn_dense):
                    raw_x = layers.Dense(units=params.cnn_dense_units_list[i], kernel_initializer=initializer)(raw_x)
                    raw_x = layers.LeakyReLU()(raw_x)
                    raw_x = layers.Dropout(params.cnn_dense_dropout_rate)(raw_x)

            input_branches['inputs'].append(raw_byte_inputs)
            input_branches['layer'].append(raw_x)

        if len(params.structure) > 1:
            # Concat input heads
            x = layers.concatenate(tuple(input_branches.get('layer')))
        else:
            x = input_branches.get('layer')[0]

        for i in range(params.num_encoder_dense):
            x = layers.Dense(units=params.encoder_dense_units_list[i], kernel_initializer=initializer)(x)
            # x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(params.encoder_dense_dropout_rate)(x)

        encoding = layers.Dense(params.openset_encoding_size, activation=None, kernel_initializer=initializer,
                                name='encoding')(x)
        x = layers.LeakyReLU()(encoding)

        for i in range(params.num_final_dense):
            x = layers.Dense(units=params.final_dense_units_list[i], kernel_initializer=initializer)(x)
            # x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(params.final_dense_dropout_rate)(x)

        outputs = layers.Dense(params.n_labels, activation='softmax', name='softmax')(x)

        return models.Model(inputs=input_branches.get('inputs'),
                            outputs=outputs)

    def _create_preprocess_layer_flow(self, features, dataset):
        @tf.function
        def featuremap(x, y):
            return {k: v for k, v in x['flow_features'].items() if k in features}

        print(dataset)
        dataset = dataset.map(featuremap)
        data = np.vstack([np.array(list(x.values())) for x in dataset])

        preprocess = layers.Normalization()
        preprocess.adapt(data)

        return preprocess

    def _create_preprocess_layer_pkt(self, features, dataset):
        @tf.function
        def featuremap(x, y):
            return {k: v for k, v in x['packet_features'].items() if k in features}

        dataset = dataset.map(featuremap)
        data = np.vstack([np.array(list(x.values())) for x in dataset])

        preprocess = layers.Normalization(axis=None)
        preprocess.adapt(data)

        return preprocess

    def train(self, dataset, validation, logdir, params):
        if not hasattr(self, 'model'):
            if 'mlp' in params.structure:
                preprocessor_flow = self._create_preprocess_layer_flow(params.features, dataset)
            else:
                preprocessor_flow = None
            if 'lstm' in params.structure:
                preprocessor_pkt = self._create_preprocess_layer_pkt(params.packet_feature_name[:2], dataset)
            else:
                preprocessor_pkt = None

            self.model = self._create_keras_model(preprocessor_flow, preprocessor_pkt, params, dataset)

        self.model.summary()

        dataset = dataset.repeat(params.repeat).shuffle(10000)
        dataset = self._balance_classes(dataset, params.n_labels)
        dataset = dataset.batch(64)

        validation = validation.batch(64)

        return self._train(self.model, dataset, validation, logdir, params)


class ThreeFeatureTypeDatasetFactoryMixin:

    def _create_dataset(self, datafile, params, training=False, filter_by_label=False):
        df = pd.read_orc(datafile)
        params.sequence_length = df.loc[0, params.packet_feature_name[0]].shape[
            0]  # retrieve sequence length from dataset

        if filter_by_label:
            # Remove unncessary labels and feature columns from dataset
            labels = params.labels
            df = df.loc[df[params.label_column].isin(labels)]
            y = df.loc[:, params.label_column]

        features = params.packet_feature_name + params.features + params.rawbytes_feature_name
        X = df.loc[:, features]

        if filter_by_label:
            validation_dataset = None
            if params.validation_percent is not None:
                X, X_test, y, y_test = train_test_split(X, y, test_size=params.validation_percent)
                validation_dataset = self._create_tf_dataset(X=X_test, params=params, y=y_test)

            return self._create_tf_dataset(X=X, params=params, y=y), validation_dataset
        else:
            # If filter_by_label is false, it is for prediction and no need to provide ground truth
            return self._create_tf_dataset(X=X, params=params)

    def _create_tf_dataset(self, X, params, y = None):
        fea_dict = {}
        y_onehot = None
        label_column = params.label_column
        labels = params.labels

        if y is not None:
            # Filter out unnecessary labels
            indices = y.isin(params.labels)
            y = y[indices]
            X = X.loc[indices, :]

            y_onehot = pd.get_dummies(y)
            labels = np.sort(np.unique(labels))

        print("The dataset size is {}".format(X.shape[0]))

        if 'mlp' in params.structure:
            X_flow = {name: np.stack(value) for name, value in X.loc[:, params.features].items()}
            fea_dict['flow_features'] = X_flow
        if 'lstm' in params.structure:
            X_pktseq = {name: np.stack(value) for name, value in X.loc[:, params.packet_feature_name].items()}
            fea_dict['packet_features'] = X_pktseq
        if 'cnn' in params.structure:
            X_raw_bytes = {name: np.stack(value) for name, value in X.loc[:, params.rawbytes_feature_name].items()}
            fea_dict['raw_byte_features'] = X_raw_bytes

        if params.structure == ['cnn']:
            # when the structure only contains cnn as input branch, i.e. the feature array only contains one feature
            # use the raw byte feature straight forward without forming a dictionary
            ds_X = tf.data.Dataset.from_tensor_slices(X_raw_bytes, name='X')
            print("[cnn]")
        else:
            ds_X = tf.data.Dataset.from_tensor_slices(fea_dict, name='X')

        # If ground truth is not provided, return the dataset with feature matrix only
        if y is not None:
            ds_y = tf.data.Dataset.from_tensor_slices(y_onehot)

            return tfd.Dataset.zip((ds_X, ds_y))
        else:
            return tfd.Dataset.zip((ds_X))


class ThreeFeatureTypeTrainerMixin:
    """Implements training loop for models."""
    def compile(self, model, params):
        metrics = ['accuracy']
        losses = CategoricalCrossentropy()

        if params.decay_steps > 0:
            learning_rate = ExponentialDecay(initial_learning_rate=params.learning_rate,
                                             decay_steps=params.decay_steps,
                                             decay_rate=params.decay_rate,
                                             staircase=True)
        else:
            learning_rate = params.learning_rate

        adam = Adam(learning_rate=learning_rate)

        model.compile(loss=losses, optimizer=adam, metrics=metrics)

    def _train(self, model, dataset, validation, logdir, params):
        self.compile(model, params)

        options = {
            'epochs': params.epochs,
            'steps_per_epoch': params.steps_per_epoch,
            'validation_data': validation,
        }

        callbacks = []
        if logdir:
            callbacks.append(TensorBoard(log_dir=logdir, histogram_freq=1))

        if params.early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7))

        if callbacks:
            options['callbacks'] = callbacks

        return model.fit(dataset, **options)


class ThreeFeatureTypeHyperModel(ThreeFeatureTypeModelFactoryMixin, ThreeFeatureTypeTrainerMixin, kt.HyperModel):
    MAX_LAYERS = 3

    def __init__(self, params, dataset):
        self.params = params
        self.dataset = dataset
        self.count = 0

    def build(self, hp):
        params = self.params
        dataset = self.dataset

        if 'mlp' in params.structure:
            preprocessor_flow = self._create_preprocess_layer_flow(params.features, dataset)
        else:
            preprocessor_flow = None
        if 'lstm' in params.structure:
            preprocessor_pkt = self._create_preprocess_layer_pkt(params.packet_feature_name[:2], dataset)
        else:
            preprocessor_pkt = None

        hps = {
            "openset_encoding_size": hp.Int("openset_encoding_size", 32, 128, 32),
            "num_encoder_dense": hp.Int("num_encoder_dense", 1, self.MAX_LAYERS),
            # "num_final_dense": hp.Int("num_final_dense", 1, self.MAX_LAYERS),
            # "encoder_dense_dropout_rate": hp.Float("encoder_dense_dropout_rate", min_value=0, max_value=0.6, step=0.1)
            # "openset_encoding_size": 128,
            # "num_encoder_dense": 3,
            "num_final_dense": 1,
            "encoder_dense_dropout_rate": 0.1
        }

        if 'mlp' in params.structure:
            hps["num_dense"] = hp.Int("num_dense", 1, self.MAX_LAYERS)
            hps["units_list"] = [hp.Int(f"input_dense_{i}", 64, 256, 64) for i in range(hps['num_dense'])]
            # hps["dropout_rate"] = hp.Float("dropout_rate", min_value=0, max_value=0.6, step=0.1)
            # hps["num_dense"] = 3
            # hps["units_list"] = [64, 64, 128]
            hps["dropout_rate"] = 0.0

        if 'lstm' in params.structure:
            # hps["masking"] = hp.Choice('masking', [True, False])
            hps["num_lstm"] = hp.Choice('num_lstm', [1, 2, 3])
            # hps["num_lstm_dense"] = hp.Int("num_lstm_dense", 1, self.MAX_LAYERS)
            hps["lstm_units"] = hp.Int("lstm_units", 64, 512, 64)
            # hps["lstm_dense_units_list"] = [hp.Int(f"lstm_dense_{i}", 64, 256, 64) for i in
            #                                 range(hps['num_lstm_dense'])]
            # hps["lstm_dense_dropout_rate"] = hp.Float("lstm_dense_dropout_rate", min_value=0, max_value=0.6, step=0.1)
            hps["masking"] = False
            # hps["num_lstm"] = 2
            hps["num_lstm_dense"] = 2
            # hps["lstm_units"] = 128
            hps["lstm_dense_units_list"] = [64, 128]
            hps["lstm_dense_dropout_rate"] = 0.0
        if 'cnn' in params.structure:
            # hps["num_cnn"] = hp.Int("num_cnn", 3, 6)
            # hps["num_filter"] = hp.Choice('num_filter', [2, 4, 8, 16])
            # hps["cnn_dropout_rate"] = hp.Float("cnn_dropout_rate", min_value=0, max_value=0.6, step=0.1)
            hps["num_cnn_dense"] = hp.Int("num_cnn_dense", 0, 2)
            hps["cnn_dense_units_list"] = [hp.Int(f"cnn_dense_{i}", 64, 256, 64) for i in
                                           range(hps['num_cnn_dense'])]
            hps["cnn_dense_dropout_rate"] = hp.Float("cnn_dense_dropout_rate", min_value=0, max_value=0.2, step=0.1)
            hps["num_cnn"] = 3
            hps["num_filter"] = 8
            hps["cnn_dropout_rate"] = 0.1
            # hps["num_cnn_dense"] = 1
            # hps["cnn_dense_units_list"] = [64]
            # hps["cnn_dense_dropout_rate"] = 0.5

        # hps["encoder_dense_units_list"] = [hp.Int(f"encoder_dense_{i}", 64, 256, 64) for i in
        #                                    range(hps['num_encoder_dense'])]
        # hps["final_dense_units_list"] = [hp.Int(f"final_dense_{i}", 64, 256, 64) for i in
        #                                  range(hps['num_final_dense'])]
        hps["learning_rate"] = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
        # hps["decay_step"] = hp.Int('decay_step', min_value=1000, max_value=6000, step=1000)
        hps["encoder_dense_units_list"] = [96, 128, 128]
        hps["final_dense_units_list"] = [192]
        # hps["learning_rate"] = 0.0001
        hps["decay_step"] = 0

        params.update(hps)
        model = self._create_keras_model(preprocessor_flow, preprocessor_pkt, params, dataset)

        self.compile(model, params)

        return model

    def fit(self, hp, model, *args, **kwargs):
        custom_run_name = "search_"+str(self.count)
        with mlflow.start_run(run_name=custom_run_name):
            mlflow.log_params(hp.values)
            mlflow.tensorflow.autolog()
            self.count += 1
            return model.fit(*args, **kwargs)


# Custom Keras Callback to log models with MLflow
class MLflowCallback(tf.keras.callbacks.Callback):
    def on_train_end(self, epoch, logs=None):
        # Log the model with MLflow
        mlflow.tensorflow.log_model(self.model, "model", keras_model_kwargs={"save_format": "h5"})
        super().on_epoch_end(epoch, logs)


class ThreeFeatureTypeHyperparameterTunerMixin:
    MAX_LAYERS = 3
    NUM_TRIALS = 50

    def tune(self, dataset, validation, logdir, params):
        mlflow.set_experiment(experiment_name=logdir)

        hypermodel = ThreeFeatureTypeHyperModel(params, dataset)
        # hypermodel = ThreeFeatureTypeHyperModel()
        tuner = BayesianOptimization(
            hypermodel=hypermodel,
            objective=Objective("val_accuracy", direction="max"),
            max_trials=self.NUM_TRIALS,
            overwrite=True,
            directory=logdir,
            project_name="twdl_threeinputs-checkpoints",
        )

        dataset = dataset.repeat(params.repeat).shuffle(10000)
        dataset = self._balance_classes(dataset, params.n_labels)
        dataset = dataset.batch(64)

        validation = validation.batch(64)

        options = {
            'epochs': 10,
            'steps_per_epoch': 100,
            'validation_data': validation,
        }
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)

        if logdir:
            options['callbacks'] = [TensorBoard(log_dir=logdir, histogram_freq=1), early_stopping, MLflowCallback()]

        tuner.search(dataset, **options)

        best_results = tuner.get_best_hyperparameters(1)
        print(best_results[0].values)

        best_models = tuner.get_best_models(num_models=5)
        for i in range(5):
            best_models[i].save(logdir + "/models/best_model_" + str(i) + ".h5")


class DefaultThreeFeatureTypeModel(ThreeFeatureTypeModelFactoryMixin,
                                   ThreeFeatureTypeDatasetFactoryMixin,
                                   ThreeFeatureTypeTrainerMixin,
                                   ThreeFeatureTypeHyperparameterTunerMixin,
                                   DefaultAbstractModel):
    """Combines default model and mixins into a reusable model."""

    def load(self, filename, params, trainable=False):
        """Load trained model parameters from the identified file."""

        self.model = load_model(filename, trainable)
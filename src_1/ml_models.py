import warnings

warnings.filterwarnings('ignore')

import json
import time

from pathlib import Path
from pickle import load, dump


from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from utils import *


class MLModels:
    
    def train_model_anomaly(self, train_file, config, trained_model):
        """Trains a ML ,model based on historical feature view data..

        Parameters:
                train_file: train data filename.
                config: model configuration filename (JSON)
                trained_model: output trained model filename.
        """
        nl = '\n'
        start_time = time.time()
        with open(config) as f:
            config = json.load(f)

        label = config['label_col']
        print(label)
        cv = config['cross_validations']
        parameters = config['parameters']

        workflows, xgb = get_pipeline(config)
        data = pd.read_parquet(path=train_file)
        target_labels = config['target_labels']
        target_label = config['label_col']
        data = data[data[target_label].isin(target_labels)]
        data.sort_values(by=[label], inplace=True)
        target_features = config['target_features']

        if xgb:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(data[label])
        else:
            y = data[label]
        X = data[target_features]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)
        pipe = GridSearchCV(workflows, parameters, cv=cv)
        model = pipe.fit(X)

        elapsed_time = time.time() - start_time

        # predictions = model.predict(X_train)

        # if xgb:
        #     predictions = label_encoder.inverse_transform(predictions)
        #     y_train = label_encoder.inverse_transform(y_train)

        # clf_report = classification_report(y_train, predictions)
        # click.echo(f'\nClassification Report (Training Dataset)\n')
        # click.echo(clf_report)

        # predictions = pipe.predict(X_test)
        # if xgb:
        #     predictions = label_encoder.inverse_transform(predictions)
        #     y_test = label_encoder.inverse_transform(y_test)

        # clf_report = classification_report(y_test, predictions)
        # click.echo(f'\nClassification Report (Validation Dataset)\n')
        # click.echo(clf_report)

        click.echo(f"{nl}Training Time(s): {str(round(elapsed_time, 2))} seconds{nl}")
        click.echo(f"{nl}Cross Validation Splits: {cv}{nl}")
        click.echo(f"{nl}Total Input Features: {model.best_estimator_.n_features_in_}{nl}")
        click.echo(f"{nl}Feature Names {nl}{'=' * 14}{nl}{model.feature_names_in_}{nl}")
        click.echo(f"{nl}Best Parameters{nl}{'=' * 15}{nl}{model.best_params_}{nl}")
        click.echo(f"{nl}Best Estimator{nl}{'=' * 15}{nl}{model.best_estimator_}{nl}")

        # col_index = model.best_estimator_[-2].get_support()
        # selected_columns = X_train.columns[col_index]
        # click.echo(
        #     f"{nl}Total Selected Features: {len(selected_columns)} features selected out of {model.best_estimator_.n_features_in_}{nl}")
        # click.echo(f"{nl}Selected Feature Names {nl}{'=' * 22}{nl}{selected_columns}{nl}")

        dump(pipe, open(trained_model, 'wb'))
        
    def train_model(self, train_file, config, trained_model):
        """Trains a ML ,model based on historical feature view data..

        Parameters:
                train_file: train data filename.
                config: model configuration filename (JSON)
                trained_model: output trained model filename.
        """
        nl = '\n'
        start_time = time.time()
        with open(config) as f:
            config = json.load(f)

        label = config['label_col']
        print(label)
        cv = config['cross_validations']
        parameters = config['parameters']

        workflows, xgb = get_pipeline(config)
        data = pd.read_parquet(path=train_file)
        target_labels = config['target_labels']
        target_label = config['label_col']
        data = data[data[target_label].isin(target_labels)]
        data.sort_values(by=[label], inplace=True)
        target_features = config['target_features']

        if xgb:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(data[label])
        else:
            y = data[label]
        X = data[target_features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)
        pipe = GridSearchCV(workflows, parameters, cv=cv)
        model = pipe.fit(X_train, y_train)

        elapsed_time = time.time() - start_time

        predictions = model.predict(X_train)

        if xgb:
            predictions = label_encoder.inverse_transform(predictions)
            y_train = label_encoder.inverse_transform(y_train)

        clf_report = classification_report(y_train, predictions)
        click.echo(f'\nClassification Report (Training Dataset)\n')
        click.echo(clf_report)

        predictions = pipe.predict(X_test)
        if xgb:
            predictions = label_encoder.inverse_transform(predictions)
            y_test = label_encoder.inverse_transform(y_test)

        clf_report = classification_report(y_test, predictions)
        click.echo(f'\nClassification Report (Validation Dataset)\n')
        click.echo(clf_report)

        click.echo(f"{nl}Training Time(s): {str(round(elapsed_time, 2))} seconds{nl}")
        click.echo(f"{nl}Cross Validation Splits: {cv}{nl}")
        click.echo(f"{nl}Total Input Features: {model.best_estimator_.n_features_in_}{nl}")
        click.echo(f"{nl}Feature Names {nl}{'=' * 14}{nl}{model.feature_names_in_}{nl}")
        click.echo(f"{nl}Best Parameters{nl}{'=' * 15}{nl}{model.best_params_}{nl}")
        click.echo(f"{nl}Best Estimator{nl}{'=' * 15}{nl}{model.best_estimator_}{nl}")

        col_index = model.best_estimator_[-2].get_support()
        selected_columns = X_train.columns[col_index]
        click.echo(
            f"{nl}Total Selected Features: {len(selected_columns)} features selected out of {model.best_estimator_.n_features_in_}{nl}")
        click.echo(f"{nl}Selected Feature Names {nl}{'=' * 22}{nl}{selected_columns}{nl}")

        dump(pipe, open(trained_model, 'wb'))

    def test_model(self, trained_model_file, config_file, test_data_file, output_file):
        """Evaluates a trained ml model

        Parameters:
            trained_model_file: trained model filename.
            config: model configuration filename (JSON)
            test_data_file: test data filename.
            output_file: output classification report filename.
        """
        nl = '\n'
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        _model = open(trained_model_file, 'rb')
        model = load(_model)

        with open(config_file) as f:
            config = json.load(f)

        label = config['label_col']
        target_features = model.feature_names_in_
        test_data = pd.read_parquet(path=test_data_file)
        target_labels = config['target_labels']
        target_label = config['label_col']
        # data = data[data[target_label].isin(target_labels)]
        test_data = test_data[test_data[target_label].isin(target_labels)]
        test_data.sort_values(by=[label], inplace=True)

        def is_xgboost():
            xgb = False
            if isinstance(model.best_estimator_.steps[-1][1], XGBClassifier):
                xgb = True
            return xgb

        X = test_data[target_features]
        y = test_data[label]
        predictions = model.predict(X)

        if is_xgboost():
            label_encoder = LabelEncoder()
            label_encoder.fit_transform(y)
            predictions = label_encoder.inverse_transform(predictions)
        _classification_report = classification_report(y, predictions)
        _confusion_matrix = confusion_matrix(y, predictions)
        print(_confusion_matrix)
        vocab = np.unique(y)
        print(vocab)
        _klass = ['discord', 'facebook_messenger', 'microsoft_teams', 'others', 'signal', 'telegram', 'whatsapp']

        confusion_matrix_report = get_classification_report(_confusion_matrix=_confusion_matrix,
                                                            traffic_classes=vocab)

        click.echo(confusion_matrix_report)
        confusion_matrix_report.to_csv(output_file)
        scores = confusion_matrix_report.f1_score.to_list()
        plot_bargraph_cm(target_labels, scores)

    # def predict_model(self, model_file: InputPath('File'),
    #                   target_view_file: InputPath('File'),
    #                   data_file: InputPath('File'),
    #                   predictions_file: OutputPath('File')):
    #
    #     """
    #     Parameters:
    #         model_file: trained model filename.
    #         target_view_file: target features view filename
    #         data_file: test data filename.
    #         predictions_file: output prediction result filename.
    #     """
    #
    #     Path(predictions_file).parent.mkdir(parents=True, exist_ok=True)
    #     with open(model_file, 'rb') as f:
    #         model = load(f)
    #
    #     target_features = []
    #     with open(target_view_file) as f:
    #         config = json.load(f)
    #
    #     def is_xgboost():
    #         xgb = False
    #         if isinstance(model.best_estimator_.steps[-1][1], XGBClassifier):
    #             xgb = True
    #         return xgb
    #
    #     target_features = model.feature_names_in_
    #
    #     data = pd.read_csv(data_file)
    #     X = data[target_features]
    #     predictions = model.predict(X).tolist()
    #
    #     data['label'] = predictions
    #
    #     if is_xgboost():
    #         target_labels_values = sorted(config['target_label'])
    #         target_labels_keys = [target_labels_values.index(i) for i in target_labels_values]
    #         _dict = dict(map(lambda i, j: (i, j), target_labels_keys, target_labels_values))
    #         data['label'] = data['label'].map(_dict)
    #     np.savetxt(predictions_file, data, fmt="%s")
    #
    # def export_model(self, trained_model_file: InputPath('File'),
    #                  output_file: OutputPath('File')):
    #     Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    #     _model = open(trained_model_file, 'rb')
    #     model = load(_model)
    #     feature_names = model.feature_names_in_
    #     print(feature_names)
    #     initial_types = [(name, FloatTensorType([None, 1])) for name in feature_names]
    #     onnx_model = convert_sklearn(model, initial_types=initial_types, options={type(model): {'zipmap': False}})
    #     with open(output_file, "wb") as f:
    #         f.write(onnx_model.SerializeToString())
import warnings
warnings.filterwarnings('ignore')

import os
import click
import json
from datetime import datetime
from dl_models import DLModels, ModelBuilder, ModelBuilderTransferLearning


@click.group(name='dl')
def mai_dl():
    pass


@mai_dl.command(name="train")
@click.argument('train_data_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('config_file', type=click.Path(exists=True, dir_okay=False))
def mai_ml_train(train_data_file, config_file):
    
    """ Train DL model
        creates model architecture, trains the model and saves the model
        Args:
            train_data_file (str): Path to the training data file
            config_file (str): Path to the configuration file

    """
    
    """ Load config file """
    with open(config_file) as f:
            params = json.load(f)
            
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = "_".join(model_type for model_type in params['model_types'])
    params['model_name'] = model_name
   
    params['model_plot'] = f"{params['project']['project_home']}/models/models_jan13/{model_name}_{current_datetime}.png"
    MODEL_H5 = f"{model_name}_{params['epochs']}_{params['learning_rate']}_{current_datetime}.h5"    
    MODEL_H5_PATH = f"{params['project']['project_home']}/models/models_jan21/{MODEL_H5}"
    params['model_h5_path'] = MODEL_H5_PATH
    MODEL_JSON = f"{model_name}_{params['epochs']}_{params['learning_rate']}_{current_datetime}.json"
    params['output_units'] = len(params['labels'])
    hyper_params = {
        'dropout_rate': [0.1],
        # 'learning_rate': [0.01],
        # 'l1': [0.0001, 0.001, 0.01, 0.1], 
        # 'l2': [0.0001, 0.001, 0.01, 0.1]
        
    }
    # dropout_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    # learning_rate_list  = [0.0001, 0.001, 0.01, 0.1]
    # regularization_value_list = [0.0001, 0.001, 0.01, 0.1]
    
    params['experiment_name'] = f"cnn_stat_solana_data_tr_ext_2024-03-02_v1"
    
    for k, v in hyper_params.items():
        for hyper_param in v:
            if k == 'l1':
                params['regularizer'] = k
                params['regularizer_value'] = hyper_param
            elif k == 'l2':
                params['regularizer'] = k
                params['regularizer_value'] = hyper_param
            else:
                params[k] = hyper_param
            params['run_name'] = f"Solana_data_twc_features_24_{params['model_name']}_v2_{k}_{hyper_param}"
            """ Build model using ModelBuilder 
                Return: model architecture 
            """
            if params.get("use_transfer_learning", False):
                model_builder = ModelBuilderTransferLearning(params, params['output_units'])
            else:
                model_builder = ModelBuilder(params)
            model_arch  = model_builder.build_model()
            model_json = model_arch.to_json()
            with open(f"/home/mpaul/projects/mpaul/mai2/models/models_feb02/{MODEL_JSON}", "w") as json_file:
                json.dump(model_json, json_file)
            
            """ Model Summary """
            model_arch.summary()
            
            """ Train DL model
                Return: trained model
            """
            _model = DLModels(
                train_file=train_data_file,
                params=params,
                model_arch=model_arch,
                test_file=None,
                trained_model_file=None
            )
            
            model = _model.train_model()
            model.save(params['model_h5_path'], save_format='h5')

@mai_dl.command("test")
@click.argument('trained_model_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('config_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('test_data_file', type=click.Path(exists=True, dir_okay=False))
# @click.argument('output', type=click.Path(exists=True, dir_okay=True))
def test_model_cmd(trained_model_file, config_file, test_data_file):
    
    with open(config_file) as f:
        params = json.load(f)
    
    params['output_units'] = len(params['labels'])
    
    if trained_model_file is not None:
        head, tail = os.path.split(trained_model_file)
        # results/results_feb05/non_augmented
        output_file = os.path.join( f'{params["project"]["project_home"]}/results/results_feb09/non_augmented/', tail.replace('.h5', '.csv'))
    
    """Evaluate DL model"""
    model = DLModels(
        train_file=None, 
        params=params,
        model_arch= None,
        test_file=test_data_file,
        trained_model_file=trained_model_file
    )
    
    matrix = model.test_model()
    nl = '\n'
    click.echo(f"{nl}Confusion Matrix Flow Count Based{nl}{'=' * 33}{nl}{matrix}{nl}")
    matrix.to_csv(output_file)
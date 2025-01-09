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
    params['model_plot'] = f"{params['project_home']}/models_jan06/{model_name}_{current_datetime}.png"
    MODEL_H5 = f"{model_name}_{params['epochs']}_{current_datetime}.h5"    
    params['output_units'] = len(params['labels'])

    """ Build model using ModelBuilder 
        Return: model architecture 
    """
    if params.get("use_transfer_learning", False):
        model_builder = ModelBuilderTransferLearning(params, params['output_units'])
    else:
        model_builder = ModelBuilder(params)
    model_arch  = model_builder.build_model()
    
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
    model.save(f"{params['project_home']}/models_jan06/{MODEL_H5}", save_format='h5')

@mai_dl.command("test")
@click.argument('trained_model_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('config_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('test_data_file', type=click.Path(exists=True, dir_okay=False))
def test_model_cmd(trained_model_file, config_file, test_data_file):
    
    with open(config_file) as f:
        params = json.load(f)
    
    params['output_units'] = len(params['labels'])
    
    if trained_model_file is not None:
        head, tail = os.path.split(trained_model_file)
        output_file = os.path.join( f'{params["project_home"]}/results_jan06/cross_dataset_v2', tail.replace('.h5', '.csv'))
    
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
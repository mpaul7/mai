import warnings
warnings.filterwarnings('ignore')

import click
from ml_models import MLModels


@click.group(name='ml')
def mai_ml():
    pass


@mai_ml.command(name="train")
@click.argument('train_data_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('config_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('trained-model', type=click.Path(exists=False, dir_okay=False))
def mai_ml_train(train_data_file, config_file, trained_model):
    """Train ML model"""
    MLModels().train_model(train_data_file, config_file, trained_model)
    
    
@mai_ml.command(name="train_anomaly")
@click.argument('train_data_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('config_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('trained-model', type=click.Path(exists=False, dir_okay=False))
def mai_ml_train(train_data_file, config_file, trained_model):
    """Train ML model"""
    MLModels().train_model_anomaly(train_data_file, config_file, trained_model)


@mai_ml.command("test")
@click.argument('trained_model_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('config_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('test_data_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_file', type=click.Path(exists=False, dir_okay=False))
def test_model_cmd(trained_model_file, config_file, test_data_file, output_file):
    """Evaluate ML model"""
    MLModels().test_model(trained_model_file, config_file, test_data_file, output_file)
import warnings
warnings.filterwarnings('ignore')

import click
from dl_models import DLModels


@click.group(name='dl')
def mai_dl():
    pass


@mai_dl.command(name="train")
@click.argument('model_type', type=str)
@click.argument('train_data_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('config_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('test_data_file', type=click.Path(exists=True, dir_okay=False))
def mai_ml_train(model_type,train_data_file, config_file, test_data_file):
    """Train DL model"""
    DLModels().train_model(model_type, train_data_file, config_file, test_data_file)


# @mai_dl.command("test")
# @click.argument('trained_model_file', type=click.Path(exists=True, dir_okay=False))
# @click.argument('config_file', type=click.Path(exists=True, dir_okay=False))
# @click.argument('test_data_file', type=click.Path(exists=True, dir_okay=False))
# @click.argument('output_file', type=click.Path(exists=False, dir_okay=False))
# def test_model_cmd(trained_model_file, config_file, test_data_file, output_file):
#     """Evaluate DL model"""
#     DLModels().test_model(trained_model_file, config_file, test_data_file, output_file)
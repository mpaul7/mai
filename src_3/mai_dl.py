import warnings
warnings.filterwarnings('ignore')

import click
from dl_models import DLModels


@click.group(name='dl')
def mai_dl():
    pass


@mai_dl.command(name="train")
@click.argument('train_data_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('config_file', type=click.Path(exists=True, dir_okay=False))
def mai_ml_train(train_data_file, config_file):
    
    """Train DL model"""
    model = DLModels(
        # model_type=model_type,
        train_file=train_data_file,
        config=config_file,
        test_file=None,
        trained_model_file=None
    )
    
    model = model.train_model()
        

@mai_dl.command("test")
@click.argument('trained_model_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('config_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('test_data_file', type=click.Path(exists=True, dir_okay=False))
def test_model_cmd(trained_model_file, config_file, test_data_file):
    
    """Evaluate DL model"""
    model = DLModels(
        train_file=None, 
        config=config_file,
        test_file=test_data_file,
        trained_model_file=trained_model_file
    )
    
    model.test_model()

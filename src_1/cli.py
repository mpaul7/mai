import click
from mai_ml import mai_ml
from mai_dl import mai_dl

# from src.test import test_model


@click.group()
def cli():
    """Main CLI for training and testing models."""
    pass

# Add train and test commands
cli.add_command(mai_ml)
cli.add_command(mai_dl)
# cli.add_command(test_model)

if __name__ == '__main__':
    cli()

import click
from mai_dl_enta import mai_dl

# from src.test import test_model


@click.group()
def cli():
    """Main CLI for training and testing models."""
    pass

# Add train and test commands
cli.add_command(mai_dl)
# cli.add_command(test_model)

if __name__ == '__main__':
    cli()

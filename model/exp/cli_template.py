import logging
import logging.config

import click

logger = logging.getLogger()

logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("exp-name", type=str)
def model_train(exp_name):
    pass


@cli.command()
@click.argument("exp-name", type=str)
def model_estim(exp_name):
    pass


if __name__ == "__main__":
    cli()

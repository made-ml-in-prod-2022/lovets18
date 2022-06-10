"""Main module to run app"""
import pickle
import logging
from datetime import datetime
import click
import pandas as pd


from ml_project.dataclasses.config_params_read import read_params_config
from ml_project.train import train_model
from ml_project.predict import predict


def load_data(path: str) -> pd.DataFrame:
    """Return dataframe from given path to csv file"""
    if path is not None and path != "":
        return pd.read_csv(path)
    return None


def run(config_path: str):
    """Main function"""
    params = read_params_config(config_path)
    with open(params.pathes.logger_path, "w", encoding="utf-8") as file:
        file.write(f"LOGGER FILE. SCRIPT RUN AT {datetime.utcnow()}\n\n")
    logging.basicConfig(level=logging.DEBUG, filename=params.pathes.logger_path)
    train_data = load_data(params.pathes.train_data)
    test_data = load_data(params.pathes.test_data)
    if train_data is None and test_data is None:
        logging.error("Given data incorrect")
        return
    if params.stages.train and train_data is not None:
        train_model(train_data, params)
    if params.stages.predict and test_data is not None:
        with open(params.pathes.model_path, "rb") as file:
            model = pickle.load(file)
        predict(test_data, params, model)
    logging.info("Script successfully executed")


@click.command(name="run")
@click.argument("config_path")
def run_command(config_path: str):
    """Read path argument and run"""
    run(config_path)


if __name__ == "__main__":
    run_command()  # pylint:disable=E1120

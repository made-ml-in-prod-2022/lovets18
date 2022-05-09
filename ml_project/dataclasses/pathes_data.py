"""Module for Pathes dataclass"""
from dataclasses import dataclass


@dataclass()
class Pathes:
    """Dataclass with hyperparams for model tuning from yaml"""
    train_data: str
    test_data: str
    output_data: str
    model_path: str
    logger_path: str

"""Module for Hyperparams dataclass"""
from dataclasses import dataclass


@dataclass()
class Hyperparams:
    """Dataclass with hyperparams for model tuning from yaml"""
    n_estimators: int
    random_state: int

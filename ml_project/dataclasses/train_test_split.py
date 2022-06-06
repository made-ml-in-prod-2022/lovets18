"""Module for TrainTestSplit dataclass"""
from dataclasses import dataclass


@dataclass()
class TrainTestSplit:
    """Dataclass with splitting params from yaml"""
    shuffle: bool
    test_size: float
    random_state: int

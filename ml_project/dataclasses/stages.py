"""Module for FeatureProcessing dataclass"""
from dataclasses import dataclass


@dataclass()
class Stages:
    """Dataclass with feature processing actions from yaml"""
    train: bool
    predict: bool

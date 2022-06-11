"""Module for FeatureProcessing dataclass"""
from dataclasses import dataclass


@dataclass()
class FeatureProcessing:
    """Dataclass with feature processing actions from yaml"""
    drop_low_cor: list[str]
    scale: bool

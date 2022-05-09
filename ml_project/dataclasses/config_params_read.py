"""Module for  dataclass"""
from dataclasses import dataclass
import yaml
from marshmallow_dataclass import class_schema
from .features_processing import FeatureProcessing
from .hyperpaprams import Hyperparams
from .train_test_split import TrainTestSplit
from .pathes_data import Pathes


@dataclass()
class PipelineParams:
    """Dataclass for reading all params from yaml"""
    model: str
    feature_processing: FeatureProcessing
    hyperparams: Hyperparams
    train_test_split: TrainTestSplit
    pathes: Pathes


PipelineParamsSchema = class_schema(PipelineParams)


def read_params_config(path: str) -> PipelineParams:
    """Read file according to the schema"""
    with open(path, 'r', encoding='UTF-8') as file:
        schema = PipelineParamsSchema()
        return schema.load(yaml.safe_load(file))

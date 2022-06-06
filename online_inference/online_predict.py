"""Module with classes and funcs for online_predict"""
import pickle
import pandas as pd
from pydantic import BaseModel  # pylint:disable=E0611
from sklearn.pipeline import Pipeline


from ml_project.preprocess import preprocess_data  # pylint:disable=E0401


class Dataset(BaseModel):  # pylint:disable=R0903
    """Typizide class for data"""
    data: list
    features: list[str]


class Response(BaseModel):  # pylint:disable=R0903
    """Typizide class for response"""
    target: int


def load_pickle(path: str) -> Pipeline:
    """func for loading model from pickle"""
    with open(path, "rb") as file:
        return pickle.load(file)


def online_prediction(
    data: list, features: list[str], scale: bool, drop: bool, model
) -> list[Response]:
    """perform online preict"""
    data = pd.DataFrame(data, columns=features)
    data = preprocess_data(data, drop)
    if scale:
        model["scaler"].set_test_mode()
    result = model.predict(data)
    return [Response(target=int(pred)) for pred in result]

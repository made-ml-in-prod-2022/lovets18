"""Module for training model"""
import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


from ml_project.preprocess import preprocess_data, Standartizer
from ml_project.predict import predict
from ml_project.dataclasses.config_params_read import PipelineParams


def prepare_split(data: pd.DataFrame, params: PipelineParams):
    """Prepare splitted data"""
    data = data.drop_duplicates()
    x_data = data.drop("condition", axis=1)
    y_data = data.condition
    logging.debug("target was separated")
    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=params.train_test_split.test_size,
        random_state=params.train_test_split.random_state,
        shuffle=params.train_test_split.shuffle,
    )
    logging.debug("data split in ratio %f", params.train_test_split.test_size)
    logging.debug("train shape (%d, %d)", x_train.shape[0], x_train.shape[1])
    logging.debug("test shape (%d, %d)", x_test.shape[0], x_test.shape[1])
    x_train = preprocess_data(x_train, params.feature_processing.drop_low_cor)
    return x_train, x_test, y_train, y_test


def create_model(params: PipelineParams) -> Pipeline:
    """Build model according the params"""
    if params.feature_processing.scale:
        model = Pipeline(
            [
                ("scaler", Standartizer()),
            ]
        )
    else:
        model = Pipeline([])
    if params.model == "LightGBM":
        model.steps.append(
            [
                "lgbm",
                LGBMClassifier(
                    n_estimators=params.hyperparams.n_estimators,
                    random_state=params.hyperparams.random_state,
                ),
            ]
        )
    elif params.model == "XGBoost":
        model.steps.append(
            [
                "xgb",
                XGBClassifier(
                    n_estimators=params.hyperparams.n_estimators,
                    random_state=params.hyperparams.random_state,
                    use_label_encoder=False,
                    verbosity=0,
                ),
            ]
        )
    elif params.model == "CatBoost":
        model.steps.append(
            [
                "cb",
                CatBoostClassifier(
                    n_estimators=params.hyperparams.n_estimators,
                    random_state=params.hyperparams.random_state,
                    verbose=None,
                    silent=True,
                    logging_level=None,
                ),
            ]
        )
    elif params.model == "RandomForest":
        model.steps.append(
            [
                "rf",
                RandomForestClassifier(
                    n_estimators=params.hyperparams.n_estimators,
                    random_state=params.hyperparams.random_state,
                ),
            ]
        )
    else:
        logging.error(
            "Incorrect model: %s, set to lgbm by default", params.model
        )
        model.steps.append(
            [
                "lgbm",
                LGBMClassifier(
                    n_estimators=params.hyperparams.n_estimators,
                    random_state=params.hyperparams.random_state,
                ),
            ]
        )
    return model


def train_model(data: pd.DataFrame, params: PipelineParams):
    """Train model, evaluate and save model"""
    logging.info("Train started")
    x_train, x_test, y_train, y_test = prepare_split(data, params)
    logging.debug("data was processed")
    model = create_model(params)
    logging.debug("model was created")
    model.fit(x_train, y_train)
    logging.debug("model was fitted")
    if model is not None and params.pathes.model_path != "":
        with open(params.pathes.model_path, "wb") as file:
            pickle.dump(model, file)
        logging.debug("model saved to %s", params.pathes.model_path)
    y_pred = predict(x_test, params, model, True)
    logging.debug("\n %s", str(classification_report(y_test, y_pred)))
    logging.info("Train finished")

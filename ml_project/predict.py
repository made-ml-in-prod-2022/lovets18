"""Module for prediction func"""
import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


from ml_project.preprocess import preprocess_data
from ml_project.dataclasses.config_params_read import PipelineParams


def predict(
    data_features: pd.DataFrame,
    predict_params: PipelineParams,
    model: Pipeline,
    evaluate_mode=False,
) -> np.array:
    """Get dataframe of features, predict, return and save prediction"""
    logging.info('%s started', "evaluation" if evaluate_mode else "prediction")
    data_features = preprocess_data(
        data_features, predict_params.feature_processing.drop_low_cor
    )
    logging.debug("data processed")
    if predict_params.feature_processing.scale:
        model["scaler"].set_test_mode()
        logging.debug("scaler was set to test mode")
    logging.debug("model was loaded")
    result = pd.DataFrame(
        data={
            "id": data_features.index, "predict": model.predict(data_features)
        }
    )
    logging.debug("predict preformed")
    if predict_params.pathes.output_data and not evaluate_mode:
        result.to_csv(predict_params.pathes.output_data, index=False)
        logging.debug(
            "prediction was saved to %s", predict_params.pathes.output_data
        )
    logging.info('%s finished', "evaluation" if evaluate_mode else "prediction")
    return result.predict.values

"""Module for prediction func"""
import pickle
import logging
import numpy as np
import pandas as pd
from ml_project.preprocess import preprocess_data


def predict(df_x: pd.DataFrame, predict_params) -> np.array:
    """Get dataframe of features, predict, return and save prediction"""
    logging.info('prediction started')
    df_x = preprocess_data(df_x,
                           predict_params.feature_processing.drop_low_cor)
    logging.debug('data processed')
    with open(predict_params.pathes.model_path, 'rb') as file:
        model = pickle.load(file)
    if predict_params.feature_processing.scale:
        model['scaler'].set_test_mode()
        logging.debug('scaler was set to test mode')
    logging.debug('model was loaded')
    result = pd.DataFrame(data={
        'id': df_x.index,
        'predict': model.predict(df_x)
    })
    logging.debug('predic preformed')
    if predict_params.pathes.output_data:
        result.to_csv(predict_params.pathes.output_data, index=False)
        logging.debug('prediction was saved to %s',
                      predict_params.pathes.output_data)
    logging.info('prediction finished')
    return result.predict.values

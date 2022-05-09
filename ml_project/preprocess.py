"""Module for preprocessing funcs and classes"""
import logging
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class Standartizer(BaseEstimator, TransformerMixin):
    """Transformer that standartize data"""
    @staticmethod
    def standartize(data: pd.DataFrame,
                    save_dir: str, test_mode=False) -> pd.DataFrame:
        """Preform standartization and save data"""
        if test_mode:
            df_mean = pd.read_csv(f'{save_dir}mean.csv')['0'].values
            df_std = pd.read_csv(f'{save_dir}std.csv')['0'].values
            logging.debug('data standartized, constants were read')
        else:
            df_mean = data.mean()
            df_std = data.std()
            pd.DataFrame(df_mean).to_csv(f'{save_dir}mean.csv')
            pd.DataFrame(df_std).to_csv(f'{save_dir}std.csv')
            logging.debug('data standartized, constants were saved')
        return (data-df_mean)/df_std

    def __init__(self, save_dir='./ml_project/train_constants/'):
        self.test_mode = False
        self.save_dir = save_dir

    def set_test_mode(self):
        """Turn to the test mode (read mean and std)"""
        self.test_mode = True

    def set_train_mode(self):
        """Turn to the train mode (calculate and save mean and std)"""
        self.test_mode = False

    def fit(self, *args):  # pylint:disable=W0613
        """for call .fit()"""
        self.set_train_mode()
        return self

    def transform(self, df_x, *args):  # pylint:disable=W0613
        """Call transformation of the data"""
        return self.standartize(df_x, self.save_dir, self.test_mode)


def preprocess_data(data: pd.DataFrame, drop_low_cor=False) -> pd.DataFrame:
    """Fillna and drop columns if choosed"""
    data = data.fillna(0)
    if drop_low_cor:
        data = data.drop(["chol", "restecg"], axis=1)
    return data

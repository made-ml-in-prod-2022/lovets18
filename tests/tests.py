"""Tests for preprocessing data and running model"""
import os
import unittest
import pandas as pd
import numpy as np
from ml_project.preprocess import Standartizer, preprocess_data
from ml_project.main import run


def create_df(num: int, target=False):
    """Created df alike the root/data/heart_cleveland_uplod.csv"""
    data = pd.DataFrame([])
    data['age'] = [np.random.randint(18, 80) for _ in range(num)]
    data['sex'] = [np.random.randint(0, 2) for _ in range(num)]
    data['cp'] = [np.random.randint(0, 4) for _ in range(num)]
    data['trestbps'] = [np.random.randint(90, 200) for _ in range(num)]
    data['chol'] = [np.random.randint(130, 570) for _ in range(num)]
    data['fbs'] = [np.random.randint(0, 2) for _ in range(num)]
    data['restecg'] = [np.random.randint(0, 3) for _ in range(num)]
    data['thalach'] = [np.random.randint(70, 205) for _ in range(num)]
    data['exang'] = [np.random.randint(0, 2) for _ in range(num)]
    data['oldpeak'] = [np.random.randint(0, 65) / 10.0 for _ in range(num)]
    data['slope'] = [np.random.randint(0, 3) for _ in range(num)]
    data['ca'] = [np.random.randint(0, 4) for _ in range(num)]
    data['thal'] = [np.random.randint(0, 3) for _ in range(num)]
    if target:
        data['condition'] = [np.random.randint(0, 2) for _ in range(num)]
    return data


class TestPreprocess(unittest.TestCase):
    """Class for testing preprocessing"""
    def setUp(self) -> None:
        self.data = create_df(10)

    def test_standartizer_train(self):
        """Test for standartizer mode train"""
        data = pd.DataFrame(data={'a': [1, 2, 3], 'b': [-20, 0, 10]})
        exp = pd.DataFrame(data={'a': [-1.0, 0.0, 1.0],
                                 'b': [-1.091089, 0.218218, 0.872872]})
        std = Standartizer('tests/data/')
        std.fit(data)
        data = std.transform(data)
        self.assertTrue(np.allclose(data.a.values, exp.a.values))
        self.assertTrue(np.allclose(data.b.values, exp.b.values))

    def test_standartizer_test(self):
        """Test for standartizer mode test"""
        data = pd.DataFrame(data={'a': [1, 4, 3], 'b': [-5, 0, 1]})
        exp = pd.DataFrame(data={'a': [-1.0,  2.0,  1.0],
                                 'b': [-0.10910895, 0.21821789, 0.28368326]})
        std = Standartizer('tests/data/')
        std.set_test_mode()
        data = std.transform(data)
        self.assertTrue(np.allclose(data.a.values, exp.a.values))
        self.assertTrue(np.allclose(data.b.values, exp.b.values))

    def test_preprocess_fillna(self):
        """Test for fiilna"""
        data = self.data.copy()
        data.loc[0, 0] = np.nan
        data = preprocess_data(data)
        self.assertFalse(data.isna().sum().sum())

    def test_preprocess_drop(self):
        """Test for drop low_cor_cols"""
        data = self.data.copy()
        data = preprocess_data(data, True)
        self.assertEqual(data.shape[1], 11)

    def test_preprocess_no_drop(self):
        """Test for no drop"""
        data = self.data.copy()
        data = preprocess_data(data)
        self.assertEqual(data.shape[1], 13)


class TestRun(unittest.TestCase):
    """Class for testing running model"""
    def setUp(self) -> None:
        train_df = create_df(500, True)
        train_df.to_csv('tests/data/train.csv', index=False)
        test_df = create_df(100, False)
        test_df.to_csv('tests/data/test.csv', index=False)

    def test_run_train(self):
        """Test for training model"""
        run('tests/test_configs/test_config_train.yaml')
        self.assertTrue(os.path.exists('tests/data/lgbm.pkl'))

    def test_run_test(self):
        """Test for predicting model"""
        run('tests/test_configs/test_config_test.yaml')
        self.assertTrue(os.path.exists('tests/data/prediction.csv'))

    def test_run_all(self):
        """Test for training and predicting model"""
        run('tests/test_configs/test_config_all.yaml')
        self.assertTrue(os.path.exists('tests/data/lgbm.pkl'))
        self.assertTrue(os.path.exists('tests/data/prediction.csv'))


if __name__ == '__main__':
    unittest.main()

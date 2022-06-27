import os
import pandas as pd
import click
from xgboost import XGBClassifier
import pickle


TRAIN_FEATURES_FILENAME = "train_features.csv"
TRAIN_TARGETS_FILENAME = "train_targets.csv"
MODEL_PATH = "model.pkl"


@click.command("train")
@click.option("--dir_from", default="../data/processed")
@click.option("--dir_in", default="../data/models")
def train(dir_from: str, dir_in: str) -> None:
    os.makedirs(dir_from, exist_ok=True)
    os.makedirs(dir_in, exist_ok=True)

    features = pd.read_csv(os.path.join(dir_from, TRAIN_FEATURES_FILENAME))
    targets = pd.read_csv(os.path.join(dir_from, TRAIN_TARGETS_FILENAME))

    xgb_clf = XGBClassifier()
    xgb_clf.fit(features, targets)

    with open(os.path.join(dir_in, MODEL_PATH), 'wb') as file:
        pickle.dump(xgb_clf, file)


if __name__ == '__main__':
    train()

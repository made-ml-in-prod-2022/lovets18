import os
import pandas as pd
import click
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

VAL_FEATURES_FILENAME = "val_features.csv"
VAL_TARGETS_FILENAME = "val_targets.csv"
MODEL_FILENAME = "model.pkl"
METRICS_FILENAME = "metrics.json"


@click.command("validation")
@click.option("--model_dir_from", default="../data/models")
@click.option("--data_dir_from", default="../data/processed")
@click.option("--metric_dir", default="../data/metrics")
def validation(model_dir_from: str, data_dir_from: str, metric_dir: str) -> None:
    os.makedirs(model_dir_from, exist_ok=True)
    os.makedirs(data_dir_from, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    features = pd.read_csv(os.path.join(data_dir_from, VAL_FEATURES_FILENAME))
    targets = pd.read_csv(os.path.join(data_dir_from, VAL_TARGETS_FILENAME)).to_numpy()
    with open(os.path.join(model_dir_from, MODEL_FILENAME), 'rb') as fin:
        model = pickle.load(fin)
    pred = model.predict(features)
    acc_score = accuracy_score(pred, targets)
    prec_score = precision_score(pred, targets, average="macro")
    rec_score = recall_score(pred, targets, average="macro")
    f1 = f1_score(pred, targets, average="macro")
    metrics = {"accuracy_score": acc_score, "precision_score": prec_score, "recall_score": rec_score, "f1_macro": f1}
    with open(os.path.join(metric_dir, METRICS_FILENAME), "w") as file:
        json.dump(metrics, file)


if __name__ == '__main__':
    validation()
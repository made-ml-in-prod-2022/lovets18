import os
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import click

N_ROWS = 500
FEATURES_FILENAME = "features.csv"
TARGETS_FILENAME = "targets.csv"


@click.command("generate")
@click.option("--dir_in", default="../data/raw")
def generate(dir_in: str) -> None:
    os.makedirs(dir_in, exist_ok=True)
    features, targets = make_classification(
        n_samples=N_ROWS + np.random.randint(100) - 50
    )
    features = pd.DataFrame(
        features, columns=[f"col_{i}" for i in range(features.shape[1])]
    )
    targets = pd.DataFrame(targets, columns=["target"])
    features.to_csv(os.path.join(dir_in, FEATURES_FILENAME), index=False)
    targets.to_csv(os.path.join(dir_in, TARGETS_FILENAME), index=False)


if __name__ == "__main__":
    generate()

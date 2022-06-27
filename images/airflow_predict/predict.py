import os
import pandas as pd
import pickle
import click

DATA_FILE = "features.csv"
PREDICT_FILE = "predict.csv"


@click.command("predict")
@click.option("--dir_from", default="../data/raw")
@click.option(
    "--preprocessor_path",
    default="../data/preprocess_pipeline/preprocessor.pkl"
)
@click.option("--model_path", default="../data/models/model.pkl")
@click.option("--dir_in", default="../data/predictions")
def predict(
    dir_from: str, dir_in: str, model_path: str, preprocessor_path: str
) -> None:
    os.makedirs(dir_from, exist_ok=True)
    os.makedirs(dir_in, exist_ok=True)

    data = pd.read_csv(os.path.join(dir_from, DATA_FILE))

    with open(preprocessor_path, "rb") as file:
        preprocessor = pickle.load(file)
    with open(model_path, "rb") as fin:
        model = pickle.load(fin)

    data = pd.DataFrame(preprocessor.transform(data))
    prediction = pd.DataFrame(model.predict(data))
    prediction.to_csv(os.path.join(dir_in, PREDICT_FILE), index=False)


if __name__ == "__main__":
    predict()

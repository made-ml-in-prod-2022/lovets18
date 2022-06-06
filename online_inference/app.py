"""Module for server application"""
import os
import logging
import uvicorn
from fastapi import FastAPI


from online_inference.online_predict import (  # pylint:disable=E0401
    Dataset,
    Response,
    online_prediction,
    load_pickle,
)
from ml_project.dataclasses.config_params_read import read_params_config  # pylint:disable=E0401


CONFIG_PATH = os.environ.get("PATH_TO_CONFIG")
if CONFIG_PATH is None:
    CONFIG_PATH = "configs/lightgbm_200.yaml"
params = read_params_config(CONFIG_PATH)
logging.basicConfig(level=logging.DEBUG, filename=params.pathes.logger_path)
app = FastAPI()


@app.get("/")
def main():
    """index page"""
    return "HM2"


@app.on_event("startup")
def load_model():
    """load model and print info"""
    logging.info("start")
    global model  # pylint:disable=C0103, W0601
    model_path = params.pathes.model_path
    if model_path is None:
        error = "No model path given"
        logging.error(error)
        raise RuntimeError(error)
    model = load_pickle(model_path)
    logging.info("model loaded")


@app.get("/health")
def health() -> int:
    """check if server is ready"""
    return 200 if not (model is None) else 400


@app.get("/predict", response_model=list[Response])
def predict(request: Dataset):
    """handler for predict"""
    if health() != 200:
        return [Response(target=-2)]
    lens = all(list(map(len, request.data)))
    if not lens:
        return [Response(target=-1)]
    return online_prediction(
        request.data,
        request.features,
        params.feature_processing.scale,
        params.feature_processing.drop_low_cor,
        model,
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", default="8888"))
    uvicorn.run(
        "online_inference.app:app",
        host="0.0.0.0",
        port=port
    )

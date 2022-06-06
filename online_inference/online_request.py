"""send online request to server"""
import sys
import logging
import argparse
import requests
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", dest="data_path", type=str, default="./data/test_data.csv"
    )
    parser.add_argument("-l", dest="logger_path", type=str)
    parser.add_argument("-a", dest="address", type=str, default="localhost")
    parser.add_argument("-p", dest="port", type=str, default="8888")
    logger_path = vars(parser.parse_args())["logger_path"]
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # if no logger path - std output
    if logger_path is None:
        std_stream = logging.StreamHandler(stream=sys.stdout)
        logger.addHandler(std_stream)
    else:
        logging.basicConfig(filename=logger_path)
        logger = logging.getLogger(__name__)
    data_path = vars(parser.parse_args())["data_path"]
    address = vars(parser.parse_args())["address"]
    port = vars(parser.parse_args())["port"]
    data = pd.read_csv(data_path)
    request_features = list(data.columns)
    url = f"http://{address}:{port}/predict/"
    logger.info("Url: %s", url)
    for _, row in data.iterrows():
        request_data = row.tolist()
        response = requests.get(
            url, json={"data": [request_data], "features": request_features}
        )
        if response.json() == [{"target": -1}]:
            resp = "Wrong input data length"  # pylint:disable=C0103
        elif response.json() == [{"target": -2}]:
            resp = "Server is starting, wait"  # pylint:disable=C0103
        else:
            resp = response.json()
        log_info = f"Response:\n\tstatus: {response.status_code}\n\ttarget: {resp}"
        logger.info(log_info)

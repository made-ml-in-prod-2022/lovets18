FROM python:3.9

ENV PATH_TO_CONFIG="configs/lightgbm_200.yaml"

ADD models models
ADD data data
ADD ml_project ml_project

COPY requirements.txt requirements.txt
COPY online_inference/online_predict.py online_inference/online_predict.py
COPY online_inference/app.py online_inference/app.py
COPY $PATH_TO_CONFIG $PATH_TO_CONFIG

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8888

CMD ["python", "-m", "online_inference.app"]

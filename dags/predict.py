from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from docker.types import Mount


DATA_DIR_NAME = "/data/raw/{{ ds }}"
PROCESSED_DIR_NAME = "/data/processed/{{ ds }}"
PREDICTIONS_PATH = "/data/predictions/{{ ds }}"
PREPROCESSOR_PATH = "/data/preprocess_pipeline/{{ds}}/preprocessor.pkl"
MODEL_PATH = Variable.get("MODEL_PATH")


MOUNT_SOURCE = Mount(
    source="C:/Users/Вова/Desktop/Учеба/ML_PROD/hm3/data",
    target="/data",
    type='bind'
    )


default_args = {
    "owner": "lovets",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(0),
) as dag:

    predict = DockerOperator(
        image="airflow_predict",
        command=f"--dir_from {DATA_DIR_NAME} --dir_in {PREDICTIONS_PATH} --preprocessor_path {PREPROCESSOR_PATH} --model_path {MODEL_PATH}",
        task_id="predict",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )

    predict

from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

VAL_SIZE = 0.2
METRICS_DIR_NAME = "/data/metrics/{{ ds }}"
GENERATE_DIR_NAME = "/data/raw/{{ ds }}"
PROCESSED_DIR_NAME = "/data/processed/{{ ds }}"
TRANSFORMER_DIR_NAME = "/data/preprocess_pipeline/{{ ds }}"
MODEL_DIR_NAME = "/data/models/{{ ds }}"
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
        "train",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(0),
) as dag:

    preprocess_data = DockerOperator(
        image="airflow_preprocess",
        command=f"--dir_from {GENERATE_DIR_NAME} --dir_in {PROCESSED_DIR_NAME} --transform_dir {TRANSFORMER_DIR_NAME}",
        task_id="preprocess",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )

    split_data = DockerOperator(
        image="airflow_split",
        command=f"--dir_from {PROCESSED_DIR_NAME} --val_size {VAL_SIZE}",
        task_id="split",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )

    train_model = DockerOperator(
        image="airflow_train",
        command=f"--dir_from {PROCESSED_DIR_NAME} --dir_in {MODEL_DIR_NAME}",
        task_id="fit",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )

    validate_model = DockerOperator(
        image="airflow_validation",
        command=f"--model_dir_from {MODEL_DIR_NAME} --data_dir_from {PROCESSED_DIR_NAME} --metric_dir {METRICS_DIR_NAME}",
        task_id="validation",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=[MOUNT_SOURCE]
    )


    preprocess_data >> split_data >> train_model >> validate_model

from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount


GENERATE_DIR_NAME = "data/raw/{{ ds }}"
MOUNT_SOURCE = [
    Mount(
        source="C:/Users/Вова/Desktop/Учеба/ML_PROD/hm3/data",
        target="/data",
        type="bind",
    )
]
default_args = {
    "owner": "lovets",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    "data_generator",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(0),
) as dag:
    generate = DockerOperator(
        image="airflow_data_generate",
        command=f"--dir_in {GENERATE_DIR_NAME}",
        task_id="generate_data",
        do_xcom_push=False,
        network_mode="bridge",
        mounts=MOUNT_SOURCE,
    )

    generate

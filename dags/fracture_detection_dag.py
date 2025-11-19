import sys
import os
from datetime import timedelta

PROJECT_ROOT = "/opt/airflow"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

from pipeline.ingestion import ImageIngestor
from pipeline.cleaning import ImageCleaner
from pipeline.evaluation import ModelEvaluator
from pipeline.explanation import GradCAMGenerator
from pipeline.results import ResultBuilder
from pipeline.router import TaskRouter

default_args = {
    "owner": "hector",
    "retries": 0,
    "retry_delay": timedelta(seconds=5),
}

with DAG(
    dag_id="fracture_detection_pipeline",
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=["fracture"],
    params={"filename": "00001.png"}
) as dag:

    t_ingest = PythonOperator(
        task_id="ingest",
        python_callable=ImageIngestor.task,
    )

    t_clean = PythonOperator(
        task_id="clean",
        python_callable=ImageCleaner.task,
    )

    t_evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=ModelEvaluator.task,
        provide_context=True,
    )

    t_branch = BranchPythonOperator(
        task_id="branch",
        python_callable=TaskRouter.branch_task,
    )

    t_gradcam = PythonOperator(
        task_id="gradcam",
        python_callable=GradCAMGenerator.task,
    )

    t_save = PythonOperator(
        task_id="save_results",
        python_callable=ResultBuilder.task,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    t_ingest >> t_clean >> t_evaluate >> t_branch
    t_branch >> t_gradcam >> t_save
    t_branch >> t_save

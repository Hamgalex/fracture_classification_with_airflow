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

    def ingest_task(**kwargs):
        print("\n" + "="*80)
        print("TAREA 1: INGESTION")
        print("="*80)

        dag_run = kwargs.get("dag_run")
        ti = kwargs["ti"]

        if dag_run is None or "filename" not in dag_run.conf:
            raise ValueError("Debe enviarse un parámetro filename: {\"filename\": \"00001.png\"}")

        filename = dag_run.conf["filename"]
        print(f"Archivo solicitado: {filename}")

        try:
            ingestor = ImageIngestor()
            raw_path, filename = ingestor.ingest_from_disk(filename)

            ti.xcom_push(key="filename", value=filename)
            ti.xcom_push(key="raw_path", value=raw_path)

            print(f"Ingestion exitosa: {raw_path}")
            print("="*80 + "\n")

        except Exception as e:
            print(f"Error en ingestion: {str(e)}")
            print("="*80 + "\n")
            raise

    t_ingest = PythonOperator(
        task_id="ingest",
        python_callable=ingest_task,
    )

    def clean_task(**kwargs):
        print("\n" + "="*80)
        print("TAREA 2: LIMPIEZA")
        print("="*80)

        ti = kwargs["ti"]

        filename = ti.xcom_pull(task_ids="ingest", key="filename")
        raw_path = ti.xcom_pull(task_ids="ingest", key="raw_path")

        print(f"Procesando: {filename}")
        print(f"Ruta raw: {raw_path}")

        try:
            cleaner = ImageCleaner(save_debug_images=True)
            clean_path = cleaner.clean(raw_path, filename)

            ti.xcom_push(key="clean_path", value=clean_path)

            print(f"Limpieza exitosa: {clean_path}")
            print("="*80 + "\n")

        except Exception as e:
            print(f"Error en limpieza: {str(e)}")
            print("="*80 + "\n")
            raise

    t_clean = PythonOperator(
        task_id="clean",
        python_callable=clean_task,
    )

    def evaluate_task(**kwargs):
        print("\n" + "="*80)
        print("TAREA 3: EVALUACION DEL MODELO")
        print("="*80)

        ti = kwargs["ti"]

        clean_path = ti.xcom_pull(task_ids="clean", key="clean_path")
        print(f"Evaluando imagen: {clean_path}")

        MODEL_PATH = "/opt/airflow/model/modelo_fractura_resnet50.pth"
        print(f"Modelo: {MODEL_PATH}")

        try:
            evaluator = ModelEvaluator(MODEL_PATH)
            label, score, tensor = evaluator.evaluate(clean_path)

            ti.xcom_push(key="label", value=label)
            ti.xcom_push(key="score", value=score)
            ti.xcom_push(key="tensor", value=tensor.cpu().numpy().tolist())

            print(f"Evaluacion completada")
            print("="*80 + "\n")

        except Exception as e:
            print(f"Error en evaluacion: {str(e)}")
            print("="*80 + "\n")
            raise

    t_evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=evaluate_task,
        provide_context=True,
    )

    def branch_task(**kwargs):
        ti = kwargs["ti"]
        label = ti.xcom_pull(task_ids="evaluate", key="label")

        print("BRANCH: LABEL RECIBIDO =", label)

        if label is None:
            print("ERROR: label llego como None, fallback a save_results")
            return "save_results"

        return "gradcam" if label == "fractured" else "save_results"

    t_branch = BranchPythonOperator(
        task_id="branch",
        python_callable=branch_task,
    )

    def gradcam_task(**kwargs):
        print("\n" + "="*80)
        print("TAREA 5: GRAD-CAM")
        print("="*80)

        ti = kwargs["ti"]

        clean_path = ti.xcom_pull(task_ids="clean", key="clean_path")
        label = ti.xcom_pull(task_ids="evaluate", key="label")

        print(f"Generando Grad-CAM para: {clean_path}")
        print(f"Diagnostico: {label}")

        try:
            MODEL_PATH_GRADCAM = "/opt/airflow/model/modelo_fractura_resnet50.pth"
            explainer = GradCAMGenerator(MODEL_PATH_GRADCAM)
            heatmap_path = explainer.generate(clean_path)

            ti.xcom_push(key="heatmap_path", value=heatmap_path)

            print(f"Grad-CAM generado exitosamente")
            print("="*80 + "\n")

        except Exception as e:
            print(f"Error en Grad-CAM: {str(e)}")
            print("="*80 + "\n")
            raise

    t_gradcam = PythonOperator(
        task_id="gradcam",
        python_callable=gradcam_task,
    )

    def save_results_task(**kwargs):
        print("\n" + "="*80)
        print("TAREA 6: GUARDAR RESULTADOS")
        print("="*80)

        ti = kwargs["ti"]

        filename = ti.xcom_pull(task_ids="ingest", key="filename")
        label    = ti.xcom_pull(task_ids="evaluate", key="label")
        score    = ti.xcom_pull(task_ids="evaluate", key="score")
        heatmap  = ti.xcom_pull(task_ids="gradcam", key="heatmap_path")

        print(f"Archivo: {filename}")
        print(f"Diagnostico: {label}")
        print(f"Confianza: {score:.4f} ({score*100:.2f}%)")
        if heatmap:
            print(f"Heatmap: {heatmap}")
        else:
            print(f"Heatmap: No generado (imagen sin fractura)")

        try:
            builder = ResultBuilder()
            result = builder.build_and_save(filename, label, score, heatmap)

            print(f"\nResultados guardados exitosamente")
            print(f"JSON: data/results/{filename}_result.json")
            print("="*80 + "\n")

            return result

        except Exception as e:
            print(f"Error al guardar resultados: {str(e)}")
            print("="*80 + "\n")
            raise

    t_save = PythonOperator(
        task_id="save_results",
        python_callable=save_results_task,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    t_ingest >> t_clean >> t_evaluate >> t_branch
    t_branch >> t_gradcam >> t_save
    t_branch >> t_save

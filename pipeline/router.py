class TaskRouter:
    """Clase para manejar el enrutamiento condicional de tareas en Airflow"""

    @staticmethod
    def branch_task(**kwargs):
        """Tarea de Airflow para decidir si generar Grad-CAM o guardar directamente"""
        ti = kwargs["ti"]
        label = ti.xcom_pull(task_ids="evaluate", key="label")

        print("BRANCH: LABEL RECIBIDO =", label)

        if label is None:
            print("ERROR: label llego como None, fallback a save_results")
            return "save_results"

        return "gradcam" if label == "fractured" else "save_results"

import os
from config.config import RAW_DIR

class ImageIngestor:
    def ingest_from_disk(self, filename: str):
        path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe la imagen en RAW_DIR: {path}")
        return path, filename

    @staticmethod
    def task(**kwargs):
        """Tarea de Airflow para ingestión de imágenes"""
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

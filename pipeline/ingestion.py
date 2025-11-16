import os
from config.config import RAW_DIR

class ImageIngestor:
    """
    Etapa 1: Ingesta de imagen desde disco (para Airflow).
    """

    def ingest_from_disk(self, filename: str):
        """
        Valida que la imagen existe en data/raw y devuelve su ruta completa.
        """
        path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe la imagen en RAW_DIR: {path}")
        return path, filename

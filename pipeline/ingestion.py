import os
from config.config import RAW_DIR

class ImageIngestor:
    def ingest_from_disk(self, filename: str):
        path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe la imagen en RAW_DIR: {path}")
        return path, filename

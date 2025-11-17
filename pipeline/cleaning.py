import os
import shutil
from config.config import CLEAN_DIR

class ImageCleaner:
    def __init__(self, save_debug_images=True):
        self.save_debug_images = save_debug_images

    def clean(self, raw_path: str, filename: str) -> str:
        clean_path = os.path.join(CLEAN_DIR, filename)

        print("\n" + "="*60)
        print("LIMPIEZA DE IMAGEN - SIN PREPROCESSING")
        print("="*60)
        print(f"Imagen raw: {raw_path}")
        print(f"El modelo fue entrenado SIN preprocessing")
        print(f"Solo se aplicara: Resize + ToTensor + Normalize (en evaluation.py)")

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"No existe la imagen: {raw_path}")

        shutil.copy(raw_path, clean_path)

        print(f"\nImagen copiada sin modificaciones")
        print(f"Guardada en: {clean_path}")
        print("="*60 + "\n")

        return clean_path

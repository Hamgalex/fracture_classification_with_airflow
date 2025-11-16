import os
import cv2
import numpy as np
import shutil
from config.config import CLEAN_DIR

class ImageCleaner:
    """
    Limpieza EXACTA usada en tu modelo entrenado.
    """

    def __init__(self, save_debug_images=True):
        """
        Args:
            save_debug_images: Si True, guarda imágenes intermedias para debugging
        """
        self.save_debug_images = save_debug_images

    def clean(self, raw_path: str, filename: str) -> str:
        """
        SIN PREPROCESSING - El modelo fue entrenado con imágenes RAW.
        Solo copiamos la imagen tal cual para que el modelo la reconozca.
        """
        clean_path = os.path.join(CLEAN_DIR, filename)

        print(f"\n{'='*60}")
        print(f"🧹 LIMPIEZA DE IMAGEN - SIN PREPROCESSING")
        print(f"{'='*60}")
        print(f"📁 Imagen raw: {raw_path}")
        print(f"ℹ️  El modelo fue entrenado SIN preprocessing")
        print(f"ℹ️  Solo se aplicará: Resize + ToTensor + Normalize (en evaluation.py)")

        # Verificar que la imagen existe
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"No existe la imagen: {raw_path}")

        # Copiar imagen RAW directamente (sin modificaciones)
        shutil.copy(raw_path, clean_path)

        print(f"\n✅ Imagen copiada sin modificaciones")
        print(f"💾 Guardada en: {clean_path}")
        print(f"{'='*60}\n")

        return clean_path

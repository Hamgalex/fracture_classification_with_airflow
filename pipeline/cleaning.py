import os
import cv2
import numpy as np
from config.config import CLEAN_DIR

class ImageCleaner:
    def __init__(self, save_debug_images=True):
        self.save_debug_images = save_debug_images

    def clean(self, raw_path: str, filename: str) -> str:
        clean_path = os.path.join(CLEAN_DIR, filename)

        print("\n" + "="*60)
        print("LIMPIEZA DE IMAGEN")
        print("="*60)
        print(f"Imagen raw: {raw_path}")

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"No existe la imagen: {raw_path}")

        # Leer imagen
        img = cv2.imread(raw_path)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen: {raw_path}")

        print(f"Dimensiones originales: {img.shape}")

        # Normalización de brillo
        # Ajustar brillo si la imagen está muy oscura o clara
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)

        if mean_brightness < 100:  # Imagen muy oscura
            alpha = 1.2  # Aumentar contraste
            beta = 20    # Aumentar brillo
            img_normalized = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            print(f"✓ Aplicado: Normalización de brillo (original={mean_brightness:.1f})")
        elif mean_brightness > 180:  # Imagen muy clara
            alpha = 0.9  # Reducir contraste
            beta = -10   # Reducir brillo
            img_normalized = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            print(f"✓ Aplicado: Normalización de brillo (original={mean_brightness:.1f})")
        else:
            img_normalized = img
            print(f"✓ Brillo adecuado (mean={mean_brightness:.1f}), sin ajuste")

        # Guardar imagen limpia
        cv2.imwrite(clean_path, img_normalized)

        print(f"\nImagen limpia guardada en: {clean_path}")
        print("="*60 + "\n")

        return clean_path

    @staticmethod
    def task(**kwargs):
        """Tarea de Airflow para limpieza de imágenes"""
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

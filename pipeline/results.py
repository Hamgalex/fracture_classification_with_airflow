import os
import json
from config.config import RESULTS_DIR

class ResultBuilder:
    def build_and_save(self, filename: str, label: str, score: float, heatmap_path: str | None):
        result = {
            "filename": filename,
            "label": label,
            "score": score,
            "heatmap_path": heatmap_path,
        }

        out_path = os.path.join(RESULTS_DIR, f"{filename}_result.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        return result

    @staticmethod
    def task(**kwargs):
        """Tarea de Airflow para guardar resultados"""
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

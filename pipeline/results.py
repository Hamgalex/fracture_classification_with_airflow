# pipeline/results.py
import os
import json
from config.config import RESULTS_DIR

class ResultBuilder:
    """
    Etapa 5: Construcción y guardado del JSON de resultado.
    """

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

import os

BASE_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "modelo_fractura_resnet50.pth")

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "clean")
HEATMAP_DIR = os.path.join(BASE_DIR, "data", "heatmaps")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

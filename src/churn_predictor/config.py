from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"
COLUMNS_PATH = MODEL_DIR / "model_columns.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

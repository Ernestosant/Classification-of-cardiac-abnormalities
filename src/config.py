from pathlib import Path

SEED = 42
VALIDATION_SIZE = 0.2
N_TIMESTEPS = 140
N_CLASSES = 5
CLASS_VALUES = [1, 2, 3, 4, 5]
CLASS_NAMES = {
    1: "Normal",
    2: "PVC",
    3: "Supraventricular premature beat",
    4: "Ectopic beat",
    5: "Unknown abnormal pathology",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

SPLIT_PATH = MODELS_DIR / "split_indices.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_model.json"
XGBOOST_META_PATH = MODELS_DIR / "xgboost_metadata.json"
IFOREST_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"
IFOREST_CONFIG_PATH = MODELS_DIR / "isolation_forest_config.json"
INCEPTION_MODEL_PATH = MODELS_DIR / "inception_cpu.pkl"
ENSEMBLE_CONFIG_PATH = MODELS_DIR / "ensemble_config.json"


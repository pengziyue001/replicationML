"""Project paths. All artifacts stay under 5.1/replication."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
DATASET_DIR = ROOT / "dataset"
GKX_CSV = DATASET_DIR / "GKX_20201231.csv"
GOYAL_XLSX = DATASET_DIR / "goyal" / "PredictorData2025.xlsx"
GOYAL_PARQUET = DATASET_DIR / "goyal" / "monthly_predictors.parquet"
PROCESSED_DIR = ROOT / "outputs" / "processed"
PREDICTION_DIR = ROOT / "outputs" / "predictions"
TABLE_DIR = ROOT / "outputs" / "tables"
FIGURE_DIR = ROOT / "outputs" / "figures"
AUDIT_DIR = ROOT / "outputs" / "audit"
LOG_DIR = ROOT / "outputs" / "logs"
TMP_DIR = ROOT / "outputs" / "tmp"

for _path in (
    PROCESSED_DIR,
    PREDICTION_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    AUDIT_DIR,
    LOG_DIR,
    TMP_DIR,
    GOYAL_XLSX.parent,
):
    _path.mkdir(parents=True, exist_ok=True)

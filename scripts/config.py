from pathlib import Path

# Project root (…/your_project/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Data directories
DATA_DIR = PROJECT_ROOT / "data"


# Raw data files
LOCATIE_REQUESTS = DATA_DIR / "requests"
LOCATIE_RESPONSES = DATA_DIR / "responses"
RUWE_DATA_PARQUET = DATA_DIR / "ruw" / "all_ruwe_data.parquet"
RUWE_RESPONSES_PARQUET = DATA_DIR / "ruw" / "all_ruwe_responses.parquet"
GEKREGEN_EXCEL_FILE = DATA_DIR / "ModifiedQueryRows.xlsx"
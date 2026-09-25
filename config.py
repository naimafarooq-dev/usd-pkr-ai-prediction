from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

CSV_FILE = DATA_DIR / "usd_pkr_data.csv"
MODEL_FILE = MODEL_DIR / "usd_pkr_model.pkl"

SYMBOL = "USD/PKR"
INTERVAL = "1min"

HISTORICAL_RECORDS = 500
MAX_CSV_LENGTH = 2000
DISPLAY_WINDOW = 100

DEFAULT_UPDATE_DELAY = 10
HIT_THRESHOLD = 0.15
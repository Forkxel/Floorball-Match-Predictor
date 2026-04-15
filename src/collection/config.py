from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_MATCHES_DIR = DATA_DIR / "raw" / "matches"
RAW_PLAYERS_DIR = DATA_DIR / "raw" / "players"
PROCESSED_DIR = DATA_DIR / "processed"

for d in [RAW_MATCHES_DIR, RAW_PLAYERS_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

API_KEY = os.getenv("SPORTRADAR_API_KEY")
ACCESS_LEVEL = "trial"
LANGUAGE_CODE = "en"
FORMAT = "json"

COMPETITION_IDS = [
    "sr:competition:255",
    "sr:competition:306",
    "sr:competition:829",
]

BASE_URL = f"https://api.sportradar.com/floorball/{ACCESS_LEVEL}/v2/{LANGUAGE_CODE}"

RAW_DATASET_PATH = RAW_MATCHES_DIR / "floorball_dataset_raw.csv"
OFFICIAL_STANDINGS_PATH = RAW_MATCHES_DIR / "floorball_official_standings.csv"

PROCESSED_DATASET_PATH = PROCESSED_DIR / "floorball_dataset_processed.csv"
ML_DATASET_PATH = PROCESSED_DIR / "floorball_dataset_ml.csv"

UNIFIED_PLAYERS_PATH = PROCESSED_DIR / "player_season_stats_unified.csv"
ROSTER_STRENGTH_PATH = PROCESSED_DIR / "team_roster_strength.csv"
FINAL_DATASET_PATH = PROCESSED_DIR / "final_training_dataset.csv"
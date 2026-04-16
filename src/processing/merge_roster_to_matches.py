from pathlib import Path
import re
import pandas as pd

from src.collection.config import (
    PROCESSED_DATASET_PATH,
    ROSTER_STRENGTH_PATH,
    PROCESSED_DIR,
)

MATCHES_WITH_ROSTER_PATH = PROCESSED_DIR / "floorball_dataset_processed_with_roster.csv"
ML_WITH_ROSTER_PATH = PROCESSED_DIR / "floorball_dataset_ml_with_roster.csv"


def normalize_season(value: str) -> str | None:
    """
    Normalize season string to YYYY-YYYY format.

    :param value: Raw season value.
    :return: Normalized season string or None.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    match = re.search(r"(20\d{2})-(20\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    match = re.search(r"(20\d{2})/(20\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    match = re.search(r"(\d{2})/(\d{2})", text)
    if match:
        return f"20{int(match.group(1)):02d}-20{int(match.group(2)):02d}"

    return None


def previous_season(season: str) -> str | None:
    """
    Compute the previous season label.

    :param season: Season string in YYYY-YYYY format.
    :return: Previous season string or None.
    """
    if not isinstance(season, str):
        return None

    parts = season.split("-")
    if len(parts) != 2:
        return None

    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError:
        return None

    return f"{start - 1}-{end - 1}"


def build_ml_from_processed_with_roster(df: pd.DataFrame, drop_draws: bool = True) -> pd.DataFrame:
    """
    Build an ML-ready dataset from processed matches with roster features.

    :param df: Processed matches DataFrame with roster columns.
    :param drop_draws: Whether to remove draw matches.
    :return: ML-ready DataFrame with features and target.
    """
    work = df.copy()

    work["start_time"] = pd.to_datetime(work["start_time"], utc=True, errors="coerce")
    work = work.sort_values("start_time").reset_index(drop=True)

    if drop_draws and "target_result" in work.columns:
        work = work[work["target_result"] != "draw"].copy()

    target_column = "target_home_win"

    drop_columns = [
        "match_id",
        "season_id",
        "season_name",
        "competition_name",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "target_result",
        "home_score_final",
        "away_score_final",
        "start_time",
    ]

    feature_columns = [col for col in work.columns if col not in drop_columns + [target_column]]

    ml_df = work[feature_columns].copy()
    ml_df[target_column] = work[target_column]

    return ml_df


def main():
    print("Loading processed matches...")
    matches = pd.read_csv(PROCESSED_DATASET_PATH)

    print("Loading roster strength...")
    roster = pd.read_csv(ROSTER_STRENGTH_PATH)

    required_match_cols = {"league", "season", "home_team_name", "away_team_name"}
    missing_match_cols = required_match_cols - set(matches.columns)
    if missing_match_cols:
        raise RuntimeError(f"Missing match columns: {sorted(missing_match_cols)}")

    required_roster_cols = {"league", "season", "team_name", "roster_strength"}
    missing_roster_cols = required_roster_cols - set(roster.columns)
    if missing_roster_cols:
        raise RuntimeError(f"Missing roster columns: {sorted(missing_roster_cols)}")

    matches["season_norm"] = matches["season"].apply(normalize_season)
    roster["season_norm"] = roster["season"].apply(normalize_season)
    matches["roster_prev_season"] = matches["season_norm"].apply(previous_season)

    roster_lookup = {
        (row["league"], row["season_norm"], row["team_name"]): row["roster_strength"]
        for _, row in roster.iterrows()
    }

    home_strength = []
    away_strength = []

    for _, row in matches.iterrows():
        league = row["league"]
        prev_season = row["roster_prev_season"]

        home_team = row["home_team_name"]
        away_team = row["away_team_name"]

        home_val = roster_lookup.get((league, prev_season, home_team), None)
        away_val = roster_lookup.get((league, prev_season, away_team), None)

        home_strength.append(home_val)
        away_strength.append(away_val)

    matches["home_roster_strength"] = home_strength
    matches["away_roster_strength"] = away_strength
    matches["roster_strength_diff"] = matches["home_roster_strength"] - matches["away_roster_strength"]

    matches.to_csv(MATCHES_WITH_ROSTER_PATH, index=False)

    ml_df = build_ml_from_processed_with_roster(matches, drop_draws=True)
    ml_df.to_csv(ML_WITH_ROSTER_PATH, index=False)

    print(f"\nSaved processed+roster: {MATCHES_WITH_ROSTER_PATH}")
    print(f"Rows: {len(matches)}")

    print(f"\nSaved ml+roster: {ML_WITH_ROSTER_PATH}")
    print(f"Rows: {len(ml_df)}")

    print("\nMissing roster stats:")
    print("home missing:", int(matches["home_roster_strength"].isna().sum()))
    print("away missing:", int(matches["away_roster_strength"].isna().sum()))

    print("\nPreview:")
    print(matches[[
        "league",
        "season",
        "season_norm",
        "roster_prev_season",
        "home_team_name",
        "away_team_name",
        "home_roster_strength",
        "away_roster_strength",
        "roster_strength_diff",
    ]].head(15).to_string())


if __name__ == "__main__":
    main()
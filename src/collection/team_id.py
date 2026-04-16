import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/processed", "floorball_dataset_processed.csv")

SELECTED_COMPETITIONS = [
    "sr:competition:255",
    "sr:competition:306",
    "sr:competition:313",
    "sr:competition:318",
    "sr:competition:829",
]

df = pd.read_csv(DATA_PATH)

for competition_id in SELECTED_COMPETITIONS:
    league_df = df[df["competition_id"] == competition_id].copy()

    if league_df.empty:
        print(f"\n{competition_id} -> no data")
        continue

    competition_name = league_df["competition_name"].iloc[0]

    teams_df = pd.concat([
        league_df[["home_team_id", "home_team_name"]].rename(
            columns={"home_team_id": "team_id", "home_team_name": "team_name"}
        ),
        league_df[["away_team_id", "away_team_name"]].rename(
            columns={"away_team_id": "team_id", "away_team_name": "team_name"}
        )
    ], ignore_index=True).drop_duplicates().sort_values("team_name")

    print(f"\n{competition_name} ({competition_id})")
    print("-" * 50)

    for _, row in teams_df.iterrows():
        print(f"{row['team_id']} | {row['team_name']}")
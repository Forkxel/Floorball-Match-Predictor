import time
import requests
import pandas as pd
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.collection.config import (
    API_KEY,
    BASE_URL,
    FORMAT,
    COMPETITION_IDS,
    RAW_DATASET_PATH,
    OFFICIAL_STANDINGS_PATH,
)

HEADERS = {"accept": "application/json"}
COMMON_PARAMS = {"api_key": API_KEY}


def get_json(path: str, extra_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{BASE_URL}/{path}.{FORMAT}"
    params = dict(COMMON_PARAMS)
    if extra_params:
        params.update(extra_params)

    response = requests.get(url, headers=HEADERS, params=params, timeout=30)

    if response.status_code == 429:
        print("Rate limit hit, waiting 5 seconds...")
        time.sleep(5)
        return get_json(path, extra_params)

    response.raise_for_status()
    time.sleep(1)
    return response.json()


def fetch_competition_seasons(competition_id: str) -> List[Dict[str, Any]]:
    return get_json(f"competitions/{competition_id}/seasons").get("seasons", [])


def fetch_season_summaries(season_id: str) -> List[Dict[str, Any]]:
    all_items = []
    start = 0
    limit = 200

    while True:
        data = get_json(f"seasons/{season_id}/summaries", {"start": start, "limit": limit})
        items = data.get("summaries", [])
        if not items:
            break

        all_items.extend(items)

        if len(items) < limit:
            break

        start += limit
        time.sleep(1)

    return all_items


def fetch_season_standings(season_id: str) -> Dict[str, Any]:
    return get_json(f"seasons/{season_id}/standings", {"live": "false"})


def parse_match(summary: Dict[str, Any], competition_id: str) -> Optional[Dict[str, Any]]:
    sport_event = summary.get("sport_event", {})
    status = summary.get("sport_event_status", {})
    context = sport_event.get("sport_event_context", {})

    competitors = sport_event.get("competitors", [])
    if len(competitors) != 2:
        return None

    home = next((c for c in competitors if c.get("qualifier") == "home"), None)
    away = next((c for c in competitors if c.get("qualifier") == "away"), None)

    if not home or not away:
        return None

    home_score = status.get("home_score")
    away_score = status.get("away_score")

    if home_score is None or away_score is None:
        return None

    season = context.get("season", {})
    competition = context.get("competition", {})

    return {
        "competition_id": competition_id,
        "competition_name": competition.get("name"),
        "season_id": season.get("id"),
        "season_name": season.get("name"),
        "match_id": sport_event.get("id"),
        "start_time": sport_event.get("start_time"),
        "home_team_id": home.get("id"),
        "home_team_name": home.get("name"),
        "away_team_id": away.get("id"),
        "away_team_name": away.get("name"),
        "home_score": int(home_score),
        "away_score": int(away_score),
    }


def _collect_nodes_with_standings(obj: Any, nodes: List[Dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        if "standings" in obj and isinstance(obj["standings"], list):
            nodes.append(obj)
        for value in obj.values():
            _collect_nodes_with_standings(value, nodes)
    elif isinstance(obj, list):
        for item in obj:
            _collect_nodes_with_standings(item, nodes)


def build_official_regular_standings(season_rows: List[Dict[str, str]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    for row in season_rows:
        competition_id = row["competition_id"]
        season_id = row["season_id"]
        season_name = row.get("season_name")

        print(f"Downloading standings for {season_id}...")
        try:
            standings_data = fetch_season_standings(season_id)
        except Exception as exc:
            print(f"Standings failed for {season_id}: {exc}")
            continue

        nodes: List[Dict[str, Any]] = []
        _collect_nodes_with_standings(standings_data, nodes)

        for node in nodes:
            phase_candidates = [
                node.get("phase"),
                node.get("type"),
                node.get("name"),
                node.get("description"),
            ]

            parent_stage = node.get("stage", {})
            if isinstance(parent_stage, dict):
                phase_candidates.extend([
                    parent_stage.get("phase"),
                    parent_stage.get("type"),
                    parent_stage.get("name"),
                    parent_stage.get("description"),
                ])

            phase_text = " ".join(str(x).lower() for x in phase_candidates if x)
            if "regular" not in phase_text and "reg" not in phase_text:
                continue

            for standing_row in node.get("standings", []):
                competitor = standing_row.get("competitor", {})
                competitor_id = competitor.get("id")
                competitor_name = competitor.get("name")

                if not competitor_id:
                    continue

                records.append({
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "season_name": season_name,
                    "team_id": competitor_id,
                    "team_name": competitor_name,
                    "official_rank": standing_row.get("rank"),
                    "official_points": standing_row.get("points"),
                    "official_played": standing_row.get("played"),
                    "official_wins": standing_row.get("wins"),
                    "official_losses": standing_row.get("losses"),
                })

    standings_df = pd.DataFrame(records)
    if standings_df.empty:
        return standings_df

    return standings_df.drop_duplicates(
        subset=["competition_id", "season_id", "team_id"],
        keep="first"
    ).reset_index(drop=True)


def collect_raw_matches_and_seasons() -> tuple[pd.DataFrame, List[Dict[str, str]]]:
    all_matches = []
    collected_seasons = []

    for competition_id in COMPETITION_IDS:
        print(f"Downloading seasons for {competition_id}...")
        seasons = fetch_competition_seasons(competition_id)

        for season in seasons:
            season_id = season.get("id")
            season_name = season.get("name")
            if not season_id:
                continue

            collected_seasons.append({
                "competition_id": competition_id,
                "season_id": season_id,
                "season_name": season_name,
            })

            print(f"Downloading summaries for {season_id}...")
            summaries = fetch_season_summaries(season_id)

            for summary in summaries:
                parsed = parse_match(summary, competition_id)
                if parsed:
                    all_matches.append(parsed)

    if not all_matches:
        raise RuntimeError("No matches parsed.")

    raw_df = pd.DataFrame(all_matches).sort_values("start_time").reset_index(drop=True)
    return raw_df, collected_seasons


def main():
    raw_df, collected_seasons = collect_raw_matches_and_seasons()
    raw_df.to_csv(RAW_DATASET_PATH, index=False)

    standings_df = build_official_regular_standings(collected_seasons)
    standings_df.to_csv(OFFICIAL_STANDINGS_PATH, index=False)

    print("Saved raw matches:", RAW_DATASET_PATH)
    print("Saved official standings:", OFFICIAL_STANDINGS_PATH)


if __name__ == "__main__":
    main()
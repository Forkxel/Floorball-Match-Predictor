import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

from src.collection.config import (
    API_KEY,
    BASE_URL,
    FORMAT,
    COMPETITION_IDS,
    RAW_DATASET_PATH,
    PROCESSED_DATASET_PATH,
    ML_DATASET_PATH,
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


def parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def normalize_season_name(value: str) -> Optional[str]:
    if not value:
        return None

    text = str(value).strip()

    # najde třeba 2025/2026 nebo 2025-2026
    import re
    m = re.search(r"(20\d{2})[/-](20\d{2})", text)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    return text


def infer_league_from_competition_name(name: str) -> Optional[str]:
    if not name:
        return None

    n = str(name).lower()

    if "svenska superligan" in n or "ssl" in n:
        return "sweden"

    if "f-liiga" in n or "liiga" in n:
        return "finland"

    if "extraliga" in n or "superliga" in n:
        return "czech"

    return None


def result_label(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "home_win"
    if away_score > home_score:
        return "away_win"
    return "draw"


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

    competition_name = competition.get("name")
    season_name = season.get("name")

    return {
        "competition_id": competition_id,
        "competition_name": competition_name,
        "league": infer_league_from_competition_name(competition_name),
        "season_id": season.get("id"),
        "season_name": season_name,
        "season": normalize_season_name(season_name),
        "match_id": sport_event.get("id"),
        "start_time": sport_event.get("start_time"),
        "home_team_id": home.get("id"),
        "home_team_name": home.get("name"),
        "away_team_id": away.get("id"),
        "away_team_name": away.get("name"),
        "home_score": int(home_score),
        "away_score": int(away_score),
    }


def default_team_stats() -> Dict[str, Any]:
    return {
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "home_played": 0,
        "home_points": 0,
        "away_played": 0,
        "away_points": 0,
    }


def make_rank_map(team_stats: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    rows = []
    for team_id, stats in team_stats.items():
        goal_diff = stats["goals_for"] - stats["goals_against"]
        rows.append((team_id, stats["points"], goal_diff, stats["goals_for"]))

    rows.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))

    rank_map = {}
    for idx, row in enumerate(rows, start=1):
        rank_map[row[0]] = idx

    return rank_map


def avg_from_history(history: deque, key: str) -> float:
    if not history:
        return 0.0
    return sum(item[key] for item in history) / len(history)


def collect_raw_matches_and_seasons() -> tuple[pd.DataFrame, List[Dict[str, str]]]:
    all_matches: List[Dict[str, Any]] = []
    collected_seasons: List[Dict[str, str]] = []

    for competition_id in COMPETITION_IDS:
        print(f"Downloading seasons for {competition_id}...")
        seasons = fetch_competition_seasons(competition_id)

        if not seasons:
            print("No seasons found.")
            continue

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

            parsed_count = 0
            for summary in summaries:
                parsed = parse_match(summary, competition_id)
                if parsed:
                    all_matches.append(parsed)
                    parsed_count += 1

            print(f"Parsed finished matches: {parsed_count}")

    if not all_matches:
        raise RuntimeError("No matches parsed.")

    raw_df = pd.DataFrame(all_matches).sort_values("start_time").reset_index(drop=True)
    return raw_df, collected_seasons


def build_processed_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    matches = raw_df.to_dict(orient="records")
    matches = sorted(matches, key=lambda m: parse_iso_datetime(m["start_time"]))

    season_team_stats = defaultdict(lambda: defaultdict(default_team_stats))
    season_recent_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=5)))
    season_recent_home = defaultdict(lambda: defaultdict(lambda: deque(maxlen=5)))
    season_recent_away = defaultdict(lambda: defaultdict(lambda: deque(maxlen=5)))

    rows = []

    for match in matches:
        season_id = match["season_id"]
        home_id = match["home_team_id"]
        away_id = match["away_team_id"]

        team_stats = season_team_stats[season_id]
        recent_history = season_recent_history[season_id]
        recent_home = season_recent_home[season_id]
        recent_away = season_recent_away[season_id]

        _ = team_stats[home_id]
        _ = team_stats[away_id]

        rank_map = make_rank_map(team_stats)

        home_stats = team_stats[home_id]
        away_stats = team_stats[away_id]

        home_recent = recent_history[home_id]
        away_recent = recent_history[away_id]
        home_recent_home = recent_home[home_id]
        away_recent_away = recent_away[away_id]

        home_rank_value = 0 if home_stats["played"] == 0 else rank_map.get(home_id, 0)
        away_rank_value = 0 if away_stats["played"] == 0 else rank_map.get(away_id, 0)

        row = {
            "competition_id": match["competition_id"],
            "competition_name": match["competition_name"],
            "league": match["league"],
            "season_id": match["season_id"],
            "season_name": match["season_name"],
            "season": match["season"],
            "match_id": match["match_id"],
            "start_time": match["start_time"],

            "home_team_id": home_id,
            "home_team_name": match["home_team_name"],
            "away_team_id": away_id,
            "away_team_name": match["away_team_name"],

            "home_rank": home_rank_value,
            "away_rank": away_rank_value,
            "rank_diff": away_rank_value - home_rank_value,

            "home_table_points": home_stats["points"],
            "away_table_points": away_stats["points"],
            "table_points_diff": home_stats["points"] - away_stats["points"],

            "home_played": home_stats["played"],
            "away_played": away_stats["played"],

            "home_goal_diff": home_stats["goals_for"] - home_stats["goals_against"],
            "away_goal_diff": away_stats["goals_for"] - away_stats["goals_against"],
            "goal_diff_diff": (
                (home_stats["goals_for"] - home_stats["goals_against"]) -
                (away_stats["goals_for"] - away_stats["goals_against"])
            ),

            "home_form_last5": avg_from_history(home_recent, "points"),
            "away_form_last5": avg_from_history(away_recent, "points"),
            "form_diff_last5": avg_from_history(home_recent, "points") - avg_from_history(away_recent, "points"),

            "home_goals_for_avg_last5": avg_from_history(home_recent, "goals_for"),
            "home_goals_against_avg_last5": avg_from_history(home_recent, "goals_against"),
            "away_goals_for_avg_last5": avg_from_history(away_recent, "goals_for"),
            "away_goals_against_avg_last5": avg_from_history(away_recent, "goals_against"),

            "home_home_form_last5": avg_from_history(home_recent_home, "points"),
            "away_away_form_last5": avg_from_history(away_recent_away, "points"),

            "target_result": result_label(match["home_score"], match["away_score"]),
            "target_home_win": int(match["home_score"] > match["away_score"]),
            "home_score_final": match["home_score"],
            "away_score_final": match["away_score"],
        }

        rows.append(row)

        home_score = match["home_score"]
        away_score = match["away_score"]

        if home_score > away_score:
            home_points = 3
            away_points = 0
            home_stats["wins"] += 1
            away_stats["losses"] += 1
        elif away_score > home_score:
            home_points = 0
            away_points = 3
            away_stats["wins"] += 1
            home_stats["losses"] += 1
        else:
            home_points = 1
            away_points = 1
            home_stats["draws"] += 1
            away_stats["draws"] += 1

        home_stats["played"] += 1
        away_stats["played"] += 1

        home_stats["points"] += home_points
        away_stats["points"] += away_points

        home_stats["goals_for"] += home_score
        home_stats["goals_against"] += away_score
        away_stats["goals_for"] += away_score
        away_stats["goals_against"] += home_score

        home_stats["home_played"] += 1
        home_stats["home_points"] += home_points
        away_stats["away_played"] += 1
        away_stats["away_points"] += away_points

        home_recent.append({
            "points": home_points,
            "goals_for": home_score,
            "goals_against": away_score
        })
        away_recent.append({
            "points": away_points,
            "goals_for": away_score,
            "goals_against": home_score
        })

        home_recent_home.append({
            "points": home_points,
            "goals_for": home_score,
            "goals_against": away_score
        })
        away_recent_away.append({
            "points": away_points,
            "goals_for": away_score,
            "goals_against": home_score
        })

    processed_df = pd.DataFrame(rows).sort_values("start_time").reset_index(drop=True)
    return processed_df


def build_ml_dataset(processed_df: pd.DataFrame, drop_draws: bool = True) -> pd.DataFrame:
    df = processed_df.copy()

    df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
    df = df.sort_values("start_time").reset_index(drop=True)

    if drop_draws:
        df = df[df["target_result"] != "draw"].copy()

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

    feature_columns = [col for col in df.columns if col not in drop_columns + [target_column]]

    ml_df = df[feature_columns].copy()
    ml_df[target_column] = df[target_column]

    return ml_df


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
                    "season": normalize_season_name(season_name),
                    "team_id": competitor_id,
                    "team_name": competitor_name,
                    "official_rank": standing_row.get("rank"),
                    "official_points": standing_row.get("points"),
                    "official_played": standing_row.get("played"),
                    "official_wins": standing_row.get("wins"),
                    "official_losses": standing_row.get("losses"),
                    "source_phase_text": phase_text,
                })

    standings_df = pd.DataFrame(records)

    if standings_df.empty:
        return standings_df

    return standings_df.drop_duplicates(
        subset=["competition_id", "season_id", "team_id"],
        keep="first"
    ).reset_index(drop=True)


def main() -> None:
    if not API_KEY:
        raise RuntimeError("Missing SPORTRADAR_API_KEY in .env")

    raw_df, collected_seasons = collect_raw_matches_and_seasons()
    raw_df.to_csv(RAW_DATASET_PATH, index=False)
    print(f"\nSaved raw matches: {RAW_DATASET_PATH}")
    print("Raw rows:", len(raw_df))

    processed_df = build_processed_dataset(raw_df)
    processed_df.to_csv(PROCESSED_DATASET_PATH, index=False)
    print(f"Saved processed dataset: {PROCESSED_DATASET_PATH}")
    print("Processed rows:", len(processed_df))

    ml_df = build_ml_dataset(processed_df, drop_draws=True)
    ml_df.to_csv(ML_DATASET_PATH, index=False)
    print(f"Saved ML dataset: {ML_DATASET_PATH}")
    print("ML rows:", len(ml_df))

    official_standings_df = build_official_regular_standings(collected_seasons)
    official_standings_df.to_csv(OFFICIAL_STANDINGS_PATH, index=False)
    print(f"Saved official standings: {OFFICIAL_STANDINGS_PATH}")
    print("Official standings rows:", len(official_standings_df))

    print("\nRaw preview:")
    print(raw_df.head(5).to_string())

    print("\nProcessed preview:")
    print(processed_df.head(5).to_string())

    print("\nML preview:")
    print(ml_df.head(5).to_string())

    print("\nOfficial standings preview:")
    print(official_standings_df.head(5).to_string())


if __name__ == "__main__":
    main()
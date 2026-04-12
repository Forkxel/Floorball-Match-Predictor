import os
import pandas as pd
import joblib

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_dir = os.path.join(base_dir, "data")
models_dir = os.path.join(base_dir, "models")

dataset_path = os.path.join(data_dir, "floorball_dataset_processed.csv")
model_path = os.path.join(models_dir, "floorball_model.pkl")

df = pd.read_csv(dataset_path)
model = joblib.load(model_path)

df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
df = df.sort_values("start_time").reset_index(drop=True)


def compute_team_stats(team_id: str, matches: pd.DataFrame, home_context: bool) -> dict:
    team_matches = matches[
        (matches["home_team_id"] == team_id) |
        (matches["away_team_id"] == team_id)
    ].copy()

    if team_matches.empty:
        return {
            "points": 0,
            "played": 0,
            "goal_diff": 0,
            "form_last5": 0.0,
            "goals_for_avg_last5": 0.0,
            "goals_against_avg_last5": 0.0,
            "context_form_last5": 0.0,
        }

    team_matches = team_matches.sort_values("start_time")

    points = 0
    goals_for = 0
    goals_against = 0

    for _, row in team_matches.iterrows():
        if row["home_team_id"] == team_id:
            gf = row["home_score_final"]
            ga = row["away_score_final"]
        else:
            gf = row["away_score_final"]
            ga = row["home_score_final"]

        if gf > ga:
            points += 3
        elif gf == ga:
            points += 1

        goals_for += gf
        goals_against += ga

    last5 = team_matches.tail(5)

    form_points = 0
    gf_last5 = 0
    ga_last5 = 0

    for _, row in last5.iterrows():
        if row["home_team_id"] == team_id:
            gf = row["home_score_final"]
            ga = row["away_score_final"]
        else:
            gf = row["away_score_final"]
            ga = row["home_score_final"]

        if gf > ga:
            form_points += 3
        elif gf == ga:
            form_points += 1

        gf_last5 += gf
        ga_last5 += ga

    if home_context:
        context_matches = team_matches[team_matches["home_team_id"] == team_id].tail(5)
    else:
        context_matches = team_matches[team_matches["away_team_id"] == team_id].tail(5)

    context_points = 0
    for _, row in context_matches.iterrows():
        if row["home_team_id"] == team_id:
            gf = row["home_score_final"]
            ga = row["away_score_final"]
        else:
            gf = row["away_score_final"]
            ga = row["home_score_final"]

        if gf > ga:
            context_points += 3
        elif gf == ga:
            context_points += 1

    return {
        "points": points,
        "played": len(team_matches),
        "goal_diff": goals_for - goals_against,
        "form_last5": form_points / max(len(last5), 1),
        "goals_for_avg_last5": gf_last5 / max(len(last5), 1),
        "goals_against_avg_last5": ga_last5 / max(len(last5), 1),
        "context_form_last5": context_points / max(len(context_matches), 1) if len(context_matches) > 0 else 0.0,
    }


def make_rank_map(matches_before: pd.DataFrame, season_id: str) -> dict:
    season_matches = matches_before[matches_before["season_id"] == season_id].copy()

    team_ids = set(season_matches["home_team_id"]).union(set(season_matches["away_team_id"]))
    table_rows = []

    for team_id in team_ids:
        stats = compute_team_stats(team_id, season_matches, home_context=True)

        goals_for = (
            season_matches.loc[season_matches["home_team_id"] == team_id, "home_score_final"].sum()
            + season_matches.loc[season_matches["away_team_id"] == team_id, "away_score_final"].sum()
        )

        table_rows.append((
            team_id,
            stats["points"],
            stats["goal_diff"],
            goals_for,
        ))

    table_rows.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
    return {team_id: idx + 1 for idx, (team_id, *_rest) in enumerate(table_rows)}


test_index = int(len(df) * 0.9)
match_row = df.iloc[test_index]

competition_id = match_row["competition_id"]
season_id = match_row["season_id"]
match_time = match_row["start_time"]

home_team_id = match_row["home_team_id"]
away_team_id = match_row["away_team_id"]
home_team_name = match_row["home_team_name"]
away_team_name = match_row["away_team_name"]

matches_before = df[
    (df["competition_id"] == competition_id) &
    (df["season_id"] == season_id) &
    (df["start_time"] < match_time)
].copy()

home_stats = compute_team_stats(home_team_id, matches_before, home_context=True)
away_stats = compute_team_stats(away_team_id, matches_before, home_context=False)

rank_map = make_rank_map(matches_before, season_id)

home_rank = 0 if home_stats["played"] == 0 else rank_map.get(home_team_id, 0)
away_rank = 0 if away_stats["played"] == 0 else rank_map.get(away_team_id, 0)

X_input = pd.DataFrame([{
    "competition_id": competition_id,

    "home_rank": home_rank,
    "away_rank": away_rank,
    "rank_diff": away_rank - home_rank,

    "home_table_points": home_stats["points"],
    "away_table_points": away_stats["points"],
    "table_points_diff": home_stats["points"] - away_stats["points"],

    "home_played": home_stats["played"],
    "away_played": away_stats["played"],

    "home_goal_diff": home_stats["goal_diff"],
    "away_goal_diff": away_stats["goal_diff"],
    "goal_diff_diff": home_stats["goal_diff"] - away_stats["goal_diff"],

    "home_form_last5": home_stats["form_last5"],
    "away_form_last5": away_stats["form_last5"],
    "form_diff_last5": home_stats["form_last5"] - away_stats["form_last5"],

    "home_goals_for_avg_last5": home_stats["goals_for_avg_last5"],
    "home_goals_against_avg_last5": home_stats["goals_against_avg_last5"],
    "away_goals_for_avg_last5": away_stats["goals_for_avg_last5"],
    "away_goals_against_avg_last5": away_stats["goals_against_avg_last5"],

    "home_home_form_last5": home_stats["context_form_last5"],
    "away_away_form_last5": away_stats["context_form_last5"],
}])

prob = model.predict_proba(X_input)[0]
pred = model.predict(X_input)[0]

actual = int(match_row["home_score_final"] > match_row["away_score_final"])

print("Competition:", competition_id)
print("Season:", season_id)
print("Match date:", match_time)
print("Home team:", home_team_name)
print("Away team:", away_team_name)
print()

print("Model input:")
print(X_input.to_string(index=False))
print()

print("Prediction:")
print(f"Home win probability: {prob[1]:.4f}")
print(f"Away / not-home-win probability: {prob[0]:.4f}")
print(f"Predicted class: {pred}")
print(f"Actual class: {actual}")
print(f"Actual score: {match_row['home_score_final']}:{match_row['away_score_final']}")
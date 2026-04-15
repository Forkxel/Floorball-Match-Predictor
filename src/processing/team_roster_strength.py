from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "player_season_stats_unified.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "team_roster_strength.csv"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def safe_sum(series: pd.Series) -> float:
    return float(series.fillna(0).sum())


def safe_mean(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    return float(s.mean())


def build_team_roster_strength(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    for col in ["gp", "goals", "assists", "points"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["ppg"] = np.where(work["gp"] > 0, work["points"] / work["gp"], 0.0)

    rows = []

    group_cols = ["league", "season", "team_name"]

    for (league, season, team_name), grp in work.groupby(group_cols, dropna=True):
        grp = grp.copy()

        grp = grp.sort_values(
            by=["points", "goals", "ppg"],
            ascending=False,
            na_position="last"
        ).reset_index(drop=True)

        top3 = grp.head(3)
        top5 = grp.head(5)

        total_points = safe_sum(grp["points"])
        total_goals = safe_sum(grp["goals"])

        top3_points_sum = safe_sum(top3["points"])
        top5_points_sum = safe_sum(top5["points"])
        top3_goals_sum = safe_sum(top3["goals"])
        top5_goals_sum = safe_sum(top5["goals"])

        top3_points_share = top3_points_sum / total_points if total_points > 0 else 0.0
        top5_points_share = top5_points_sum / total_points if total_points > 0 else 0.0

        depth_10plus = int((grp["points"].fillna(0) >= 10).sum())
        depth_20plus = int((grp["points"].fillna(0) >= 20).sum())

        avg_top5_ppg = safe_mean(top5["ppg"])

        roster_strength = (
            top5_points_sum * 0.35
            + top3_points_sum * 0.20
            + top5_goals_sum * 0.15
            + avg_top5_ppg * 20 * 0.15
            + depth_10plus * 2.0
            + depth_20plus * 3.0
        )

        rows.append({
            "league": league,
            "season": season,
            "team_name": team_name,
            "roster_player_count": int(len(grp)),
            "team_total_points": total_points,
            "team_total_goals": total_goals,
            "top3_points_sum": top3_points_sum,
            "top5_points_sum": top5_points_sum,
            "top3_goals_sum": top3_goals_sum,
            "top5_goals_sum": top5_goals_sum,
            "top3_points_share": top3_points_share,
            "top5_points_share": top5_points_share,
            "depth_10plus": depth_10plus,
            "depth_20plus": depth_20plus,
            "avg_top5_ppg": avg_top5_ppg,
            "roster_strength": roster_strength,
        })

    out = pd.DataFrame(rows)

    out["roster_strength_rank"] = (
        out.groupby(["league", "season"])["roster_strength"]
        .rank(ascending=False, method="dense")
    )

    return out.sort_values(["league", "season", "roster_strength"], ascending=[True, True, False]).reset_index(drop=True)


def main():
    if not INPUT_PATH.exists():
        raise RuntimeError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required = {"league", "season", "team_name", "gp", "goals", "assists", "points"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    out = build_team_roster_strength(df)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(out)}")
    print(out.head(15).to_string())


if __name__ == "__main__":
    main()
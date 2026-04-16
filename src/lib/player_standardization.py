import pandas as pd


TARGET_COLUMNS = [
    "source",
    "league",
    "season",
    "team_name",
    "player_name",
    "gp",
    "goals",
    "assists",
    "points",
    "shots",
    "pim",
    "plus_minus",
]

COLUMN_ALIASES = {
    "source": ["source"],
    "league": ["league"],
    "season": ["season"],
    "team_name": ["team_name", "team", "club"],
    "player_name": ["player_name", "player", "name", "jmeno", "jméno"],
    "gp": ["gp", "games_played", "games", "z"],
    "goals": ["goals", "g", "b"],
    "assists": ["assists", "a"],
    "points": ["points", "tp", "kb", "p"],
    "shots": ["shots", "s", "st"],
    "pim": ["pim", "penalties", "tm"],
    "plus_minus": ["plus_minus", "+/-", "plusminus"],
}


def normalize_colname(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_season(value: str) -> str:
    value = str(value).strip()
    if "/" in value:
        parts = value.split("/")
        if len(parts) == 2:
            return f"{parts[0]}-{parts[1]}"
    return value


def clean_text(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if value in {"", "nan", "None", "null"}:
        return None
    return value


def infer_league_from_file_name(file_name: str) -> str | None:
    n = file_name.lower()
    if "ssl" in n:
        return "sweden"
    if "fliiga" in n:
        return "finland"
    if "extraliga" in n or "cesky" in n:
        return "czech"
    return None


def find_matching_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    normalized_columns = {normalize_colname(col): col for col in df.columns}

    for alias in aliases:
        alias_norm = normalize_colname(alias)
        if alias_norm in normalized_columns:
            return normalized_columns[alias_norm]

    return None


def infer_source_from_league(league: str | None) -> str | None:
    if league == "sweden":
        return "ssl"
    if league == "finland":
        return "fliiga"
    if league == "czech":
        return "extraliga"
    return None


def infer_season_from_file_name(file_name: str) -> str | None:
    lower_name = file_name.lower()

    for token in ["2025_2026", "2024_2025", "2023_2024", "2022_2023"]:
        if token in lower_name:
            return token.replace("_", "-")

    return None


def standardize_player_df(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    out = pd.DataFrame()

    for target_col, aliases in COLUMN_ALIASES.items():
        matched = find_matching_column(df, aliases)
        out[target_col] = df[matched] if matched else None

    inferred_league = infer_league_from_file_name(file_name)

    if inferred_league and out["league"].isna().all():
        out["league"] = inferred_league

    if out["source"].isna().all():
        out["source"] = infer_source_from_league(inferred_league)

    if out["season"].isna().all():
        out["season"] = infer_season_from_file_name(file_name)

    for col in ["source", "league", "season", "team_name", "player_name"]:
        out[col] = out[col].apply(clean_text)

    out["season"] = out["season"].apply(lambda x: normalize_season(x) if x else None)

    for col in ["gp", "goals", "assists", "points", "shots", "pim", "plus_minus"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["points"] = out.apply(
        lambda row: row["points"]
        if pd.notna(row["points"])
        else (
            (row["goals"] if pd.notna(row["goals"]) else 0)
            + (row["assists"] if pd.notna(row["assists"]) else 0)
        ),
        axis=1,
    )

    out = out[out["player_name"].notna()]
    out = out[out["team_name"].notna()]

    return out[TARGET_COLUMNS]
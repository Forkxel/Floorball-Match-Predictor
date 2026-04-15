from pathlib import Path
import re
import unicodedata
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_PLAYERS_DIR = BASE_DIR / "data" / "raw" / "players"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED_OUTPUT_PATH = OUTPUT_DIR / "player_season_stats_unified.csv"

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

TEAM_RENAME_RAW = {
    "czech": {
        "FBŠ Hummel Hattrick Brno": "Hattrick Brno",
        "BA SOKOLI Pardubice": "Sokoli Pardubice",
        "SOKOLI Pardubice": "Sokoli Pardubice",
        "HDT.cz Florbal Vary Bohemians": "FbS Bohemians Prague",
        "Florbal Vary": "FbS Bohemians Prague",
        "FbŠ Bohemians": "FbS Bohemians Prague",
        "FbS Bohemians": "FbS Bohemians Prague",
        "FAT PIPE FLORBAL CHODOV": "Florbal Chodov",
        "Florbal Ústí": "Usti Nad Labem",
        "FBC 4CLEAN Česká Lípa": "FBC Ceska Lipa",
        "FBC ČPP Bystroň Group OSTRAVA": "FBC Ostrava",
        "FBC ČPP Bystroň Group Ostrava": "FBC Ostrava",
        "Předvýběr.CZ Florbal MB": "Florbal Mlada Boleslav",
        "ESA logistika Tatran Střešovice": "Tatran Stresovice",
        "FBC Liberec": "FBC Liberec",
        "TJ Sokol Královské Vinohrady": "Sokol Kral Vinohrady",
        "1. SC NATIOS Vítkovice": "1. SC Vitkovice Ostrava",
        "1. SC TEMPISH Vítkovice NATIOS": "1. SC Vitkovice Ostrava",
        "1. SC TEMPISH Vítkovice": "1. SC Vitkovice Ostrava",
        "Kanonýři Kladno": "FBC Kladno",
        "ACEMA Sparta Praha": "Sparta Prague",
        "Bulldogs Brno": "Bulldogs Brno",
        "BLACK ANGELS": "Black Angels Prague",
        "BUTCHIS": "Buchis",
    },

    "finland": {
        "CLASSIC": "SC Classic Tampere",
        "ERÄVIIKINGIT": "EraViikingit",
        "ERÄVIIKINGIT #": "EraViikingit",
        "FBC TURKU": "FBC Turku",
        "TPS": "TPS Turku",
        "HAPPEE": "Happee Jyvaskyla",
        "HAWKS": "Hawks Helsinki",
        "INDIANS": "Westend Indians",
        "JYMY": "Nurmon Jymy",
        "JYMY #": "Nurmon Jymy",
        "KARHUT": "FBT Karhut Pori",
        "LASB": "LASB Lahti",
        "NOKIAN KRP": "Nokian KRP",
        "OILERS": "Oilers Espoo",
        "OILERS #": "Oilers Espoo",
        "OLS": "OLS Oulu",
        "OLS #": "OLS Oulu",
        "RANGERS": "Kirkkonummi Rangers",
        "SPV": "SPV",
        "SPV #": "SPV",
    },

    "sweden": {
        "AIK IBF": "AIK Innebandy",
        "FBC Kalmarsund": "FBC Kalmarsund",
        "FC Helsingborg": "FC Helsingborg",
        "Hagunda IF": "Hagunda IF",
        "Hovslätts IK": "Hovslatts IK",
        "IBF Falun": "IBF Falun",
        "IBK Dalen": "IBK Dalen",
        "Jönköpings IK": "Jonkopings IK",
        "Karlstad IBF": "Karlstad IBF",
        "Linköping IBK": "Linkoping IBK",
        "Mullsjö AIS": "Mullsjo AIS",
        "Nykvarns IBF": "Nykvarns IBF Ungdom",
        "Pixbo IBK": "Pixbo Wallenstam",
        "Storvreta IBK": "Storvreta IBK",
        "Strängnäs IBK": "Strangnas IBK",
        "Team Thorengruppen SK": "Thorengruppen SK",
        "Visby IBK": "Visby IBK",
        "Växjö Vipers": "Vaxjo IBK",
        "Warberg IC": "Warberg IC",
    },
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


def ascii_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"#\s*\d*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_manual_team_map(raw_map: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    out = {}
    for league, mapping in raw_map.items():
        out[league] = {}
        for raw_name, final_name in mapping.items():
            out[league][ascii_normalize(raw_name)] = final_name
    return out


TEAM_RENAME = build_manual_team_map(TEAM_RENAME_RAW)


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


def rename_team_name(team_name: str, league: str) -> tuple[str, str]:
    """
    returns: (new_team_name, method)
    """
    if team_name is None:
        return None, "missing"

    normalized = ascii_normalize(team_name)
    league_map = TEAM_RENAME.get(league, {})

    if normalized in league_map:
        return league_map[normalized], "manual_override"

    return team_name, "unchanged"


def standardize_player_df(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    out = pd.DataFrame()

    for target_col, aliases in COLUMN_ALIASES.items():
        matched = find_matching_column(df, aliases)
        if matched is not None:
            out[target_col] = df[matched]
        else:
            out[target_col] = None

    inferred_league = infer_league_from_file_name(file_name)

    if inferred_league is not None and out["league"].isna().all():
        out["league"] = inferred_league

    if out["source"].isna().all():
        if inferred_league == "sweden":
            out["source"] = "ssl"
        elif inferred_league == "finland":
            out["source"] = "fliiga"
        elif inferred_league == "czech":
            out["source"] = "extraliga"

    if out["season"].isna().all():
        for token in ["2025_2026", "2024_2025", "2023_2024", "2022_2023"]:
            if token in file_name.lower():
                out["season"] = token.replace("_", "-")

    for col in ["source", "league", "season", "team_name", "player_name"]:
        out[col] = out[col].apply(clean_text)

    out["season"] = out["season"].apply(lambda x: normalize_season(x) if x is not None else None)

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

    out = out[out["player_name"].notna()].copy()
    out = out[out["team_name"].notna()].copy()

    return out[TARGET_COLUMNS]


def main():
    player_files = sorted(RAW_PLAYERS_DIR.glob("*.csv"))
    if not player_files:
        raise RuntimeError(f"No player CSV files found in {RAW_PLAYERS_DIR}")

    all_players = []

    for path in player_files:
        print(f"Loading player file: {path.name}")
        raw_df = pd.read_csv(path)
        df = standardize_player_df(raw_df, path.name)

        new_team_names = []
        methods = []

        for _, row in df.iterrows():
            mapped_name, method = rename_team_name(
                team_name=row["team_name"],
                league=row["league"],
            )
            new_team_names.append(mapped_name)
            methods.append(method)

        df["team_name"] = new_team_names

        all_players.append(df)

    merged = pd.concat(all_players, ignore_index=True)

    merged = merged.drop_duplicates(
        subset=["league", "season", "team_name", "player_name"],
        keep="first",
    ).reset_index(drop=True)

    merged.to_csv(UNIFIED_OUTPUT_PATH, index=False)

    print(f"\nSaved unified players: {UNIFIED_OUTPUT_PATH}")
    print(f"Rows: {len(merged)}")

    print("\nLeagues:")
    print(merged["league"].value_counts(dropna=False).to_string())

    print("\nSeasons:")
    print(merged["season"].value_counts(dropna=False).to_string())

    print("\nPreview:")
    print(merged.head(10).to_string())


if __name__ == "__main__":
    main()
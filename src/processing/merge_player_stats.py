from pathlib import Path
import re
import unicodedata
import pandas as pd
from src.lib.player_standardization import standardize_player_df


BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_PLAYERS_DIR = BASE_DIR / "data" / "raw" / "players"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIFIED_OUTPUT_PATH = OUTPUT_DIR / "player_season_stats_unified.csv"


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


def ascii_normalize(text: str) -> str:
    """
    Normalize text to ASCII-safe lowercase string.

    :param text: Raw text.
    :return: ASCII normalized string.
    """
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"#\s*\d*", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_manual_team_map(raw_map: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """
    Normalize TEAM_RENAME mapping for easier matching.

    :param raw_map: Raw team mapping.
    :return: Normalized mapping.
    """
    out = {}
    for league, mapping in raw_map.items():
        out[league] = {}
        for raw_name, final_name in mapping.items():
            out[league][ascii_normalize(raw_name)] = final_name
    return out


TEAM_RENAME = build_manual_team_map(TEAM_RENAME_RAW)


def rename_team_name(team_name: str, league: str) -> tuple[str, str]:
    """
    Standardize team name using manual mapping.

    :param team_name: Original team name.
    :param league: League identifier.
    :return: Tuple (new_team_name, method).
    """
    if team_name is None:
        return None, "missing"

    normalized = ascii_normalize(team_name)
    league_map = TEAM_RENAME.get(league, {})

    if normalized in league_map:
        return league_map[normalized], "manual_override"

    return team_name, "unchanged"


def main():
    player_files = sorted(RAW_PLAYERS_DIR.glob("*.csv"))

    if not player_files:
        raise RuntimeError(f"No player CSV files found in {RAW_PLAYERS_DIR}")

    all_players = []

    for path in player_files:
        print(f"Loading player file: {path.name}")
        raw_df = pd.read_csv(path)
        df = standardize_player_df(raw_df, path.name)

        new_names = []
        for _, row in df.iterrows():
            mapped, _ = rename_team_name(row["team_name"], row["league"])
            new_names.append(mapped)

        df["team_name"] = new_names
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
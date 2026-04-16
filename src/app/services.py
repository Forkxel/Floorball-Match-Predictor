import os
import re
import sys
import unicodedata
import customtkinter as ctk
import joblib
import pandas as pd
from PIL import Image

def get_base_dir() -> str:
    """
    Resolve base directory for both normal run and PyInstaller build.

    :return: Base directory path.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

BASE_DIR = get_base_dir()

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "floorball_dataset_processed_with_roster.csv")
ROSTER_PATH = os.path.join(BASE_DIR, "data", "processed", "team_roster_strength.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "floorball_model_2.pkl")
OFFICIAL_STANDINGS_PATH = os.path.join(BASE_DIR, "data", "raw", "matches", "floorball_official_standings.csv")

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
COMPETITION_LOGOS_DIR = os.path.join(ASSETS_DIR, "competitions")
TEAM_LOGOS_DIR = os.path.join(ASSETS_DIR, "teams")


def normalize_text(value: str) -> str:
    """
    Normalize text to lowercase ASCII-like form.

    :param value: Raw text value.
    :return: Normalized text string.
    """
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_team_name(value: str) -> str:
    """
    Normalize team name and apply manual replacements.

    :param value: Raw team name.
    :return: Normalized team name.
    """
    text = normalize_text(value)

    replacements = {
        "hovslatts ik": "hovslatts ik",
        "hovslatts": "hovslatts ik",
        "hovslatts ik ": "hovslatts ik",
        "vaxjo vipers": "vaxjo ibk",
        "linkoping ibk": "linkoping ibk",
        "strangnas ibk": "strangnas ibk",
        "jonkopings ik": "jonkopings ik",
        "pixbo ibk": "pixbo wallenstam",
        "team thorengruppen sk": "thorengruppen sk",
        "nykvarns ibf": "nykvarns ibf ungdom",
        "warberg ic": "warberg ic",
    }

    return replacements.get(text, text)


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


def sanitize_id_for_filename(value: str) -> str:
    """
    Sanitize identifier for safe filename usage.

    :param value: Raw identifier.
    :return: Sanitized filename-safe string.
    """
    return str(value).replace(":", "_").replace("/", "_").replace("\\", "_")


def load_ctk_image(image_path: str, size: tuple[int, int]) -> ctk.CTkImage | None:
    """
    Load an image as a CTkImage.

    :param image_path: Path to the image file.
    :param size: Target image size.
    :return: CTkImage instance or None.
    """
    if not os.path.exists(image_path):
        return None

    try:
        pil_image = Image.open(image_path)
        return ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=size)
    except Exception:
        return None


class FloorballService:
    """
    Service layer for loading data, resolving teams, and preparing model predictions.
    """

    def __init__(self):
        self._validate_paths()

        self.df = pd.read_csv(DATA_PATH)
        self.roster_df = pd.read_csv(ROSTER_PATH)
        self.model = joblib.load(MODEL_PATH)

        if os.path.exists(OFFICIAL_STANDINGS_PATH):
            self.official_df = pd.read_csv(OFFICIAL_STANDINGS_PATH)
        else:
            self.official_df = pd.DataFrame()

        self.df["start_time"] = pd.to_datetime(self.df["start_time"], utc=True, errors="coerce")
        self.df = self.df.sort_values("start_time").reset_index(drop=True)

        self.roster_df["season_norm"] = self.roster_df["season"].apply(normalize_season)
        self.roster_df["team_name_norm"] = self.roster_df["team_name"].apply(normalize_team_name)

        self.roster_lookup = {}
        for _, row in self.roster_df.iterrows():
            self.roster_lookup[(row["league"], row["season_norm"], row["team_name_norm"])] = row["roster_strength"]

    def _validate_paths(self) -> None:
        """
        Validate that all required input files exist.

        :return: None.
        :raises FileNotFoundError: If any required file is missing.
        """
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

        if not os.path.exists(ROSTER_PATH):
            raise FileNotFoundError(f"Roster dataset not found: {ROSTER_PATH}")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    def get_competition_logo(self, competition_id: str) -> ctk.CTkImage | None:
        """
        Load the logo for a competition.

        :param competition_id: Competition identifier.
        :return: CTkImage or None.
        """
        filename = f"{sanitize_id_for_filename(competition_id)}.png"
        path = os.path.join(COMPETITION_LOGOS_DIR, filename)
        return load_ctk_image(path, (42, 42))

    def get_team_logo(self, team_id: str, size: tuple[int, int] = (42, 42)) -> ctk.CTkImage | None:
        """
        Load the logo for a team.

        :param team_id: Team identifier.
        :param size: Requested image size.
        :return: CTkImage or None.
        """
        filename = f"{sanitize_id_for_filename(team_id)}.png"

        for root, _, files in os.walk(TEAM_LOGOS_DIR):
            if filename in files:
                path = os.path.join(root, filename)
                return load_ctk_image(path, size)

        return None

    def get_competition_options(self) -> list[tuple[str, str]]:
        """
        Get all available competitions.

        :return: List of tuples in format (competition_id, competition_name).
        """
        comp_df = (
            self.df[["competition_id", "competition_name"]]
            .drop_duplicates()
            .sort_values("competition_name")
            .reset_index(drop=True)
        )
        return list(comp_df.itertuples(index=False, name=None))

    def get_latest_season_id_for_competition(self, competition_id: str) -> str:
        """
        Get the latest season identifier for a competition.

        :param competition_id: Competition identifier.
        :return: Latest season id.
        :raises ValueError: If no matches are found for the competition.
        """
        comp_df = self.df[self.df["competition_id"] == competition_id].copy()
        if comp_df.empty:
            raise ValueError("No matches found for selected competition.")

        season_order = comp_df.groupby("season_id")["start_time"].max().sort_values()
        return season_order.index[-1]

    def get_latest_season_name_for_competition(self, competition_id: str) -> str:
        """
        Get the latest season label for a competition.

        :param competition_id: Competition identifier.
        :return: Latest season name.
        :raises ValueError: If no matches are found for the competition.
        """
        comp_df = self.df[self.df["competition_id"] == competition_id].copy()
        if comp_df.empty:
            raise ValueError("No matches found for selected competition.")

        latest_idx = comp_df["start_time"].idxmax()
        return comp_df.loc[latest_idx, "season"]

    def get_teams_for_competition(self, competition_id: str) -> list[tuple[str, str]]:
        """
        Get all teams for the latest season of a competition.

        :param competition_id: Competition identifier.
        :return: List of tuples in format (team_id, team_name).
        """
        latest_season_id = self.get_latest_season_id_for_competition(competition_id)

        if not self.official_df.empty:
            standings = self.official_df[
                (self.official_df["competition_id"] == competition_id) &
                (self.official_df["season_id"] == latest_season_id)
            ].copy()

            if not standings.empty:
                standings["team_id"] = standings["team_id"].astype(str)

                teams_df = standings[["team_id", "team_name"]].drop_duplicates()
                teams_df = teams_df.sort_values("team_name").reset_index(drop=True)
                return list(teams_df.itertuples(index=False, name=None))

        comp_df = self.df[self.df["competition_id"] == competition_id].copy()
        if comp_df.empty:
            return []

        season_df = comp_df[comp_df["season_id"] == latest_season_id].copy()

        home_teams = season_df[["home_team_id", "home_team_name"]].rename(
            columns={"home_team_id": "team_id", "home_team_name": "team_name"}
        )
        away_teams = season_df[["away_team_id", "away_team_name"]].rename(
            columns={"away_team_id": "team_id", "away_team_name": "team_name"}
        )

        teams_df = (
            pd.concat([home_teams, away_teams], ignore_index=True)
            .drop_duplicates()
            .sort_values("team_name")
            .reset_index(drop=True)
        )

        teams_df["team_id"] = teams_df["team_id"].astype(str)
        return list(teams_df.itertuples(index=False, name=None))

    def compute_team_stats(self, team_id: str, matches: pd.DataFrame, home_context: bool) -> dict:
        """
        Compute aggregate team statistics from historical matches.

        :param team_id: Team identifier.
        :param matches: Match DataFrame.
        :param home_context: Whether to compute context form for home matches.
        :return: Dictionary with aggregated team stats.
        """
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

    def make_rank_map(self, matches_before: pd.DataFrame, season_id: str) -> dict:
        """
        Build ranking positions for teams within a season.

        :param matches_before: Match DataFrame containing previous matches.
        :param season_id: Season identifier.
        :return: Mapping of team_id to rank.
        """
        season_matches = matches_before[matches_before["season_id"] == season_id].copy()

        team_ids = set(season_matches["home_team_id"]).union(set(season_matches["away_team_id"]))
        table_rows = []

        for team_id in team_ids:
            stats = self.compute_team_stats(team_id, season_matches, home_context=True)

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

    def get_official_team_stats(self, competition_id: str, season_id: str, team_id: str) -> dict:
        """
        Get official standing values for a specific team.

        :param competition_id: Competition identifier.
        :param season_id: Season identifier.
        :param team_id: Team identifier.
        :return: Dictionary with official rank, points, and matches played.
        """
        if self.official_df.empty:
            return {
                "official_rank": None,
                "official_points": None,
                "official_played": None,
            }

        work = self.official_df.copy()
        work["team_id"] = work["team_id"].astype(str)

        row = work[
            (work["competition_id"] == competition_id) &
            (work["season_id"] == season_id) &
            (work["team_id"] == str(team_id))
        ]

        if row.empty:
            return {
                "official_rank": None,
                "official_points": None,
                "official_played": None,
            }

        first = row.iloc[0]
        return {
            "official_rank": first.get("official_rank"),
            "official_points": first.get("official_points"),
            "official_played": first.get("official_played"),
        }

    def get_league_season_roster_average(self, league: str, season: str) -> float | None:
        """
        Compute average roster strength for a league and season.

        :param league: League identifier.
        :param season: Normalized season label.
        :return: Average roster strength or None.
        """
        rows = self.roster_df[
            (self.roster_df["league"] == league) &
            (self.roster_df["season_norm"] == season)
        ]
        if rows.empty:
            return None

        return float(rows["roster_strength"].mean())

    def get_team_roster_strength_with_fallback(
        self,
        league: str,
        prev_season: str | None,
        current_season: str | None,
        team_name: str,
    ):
        """
        Resolve team roster strength using season and league fallbacks.

        :param league: League identifier.
        :param prev_season: Previous season label.
        :param current_season: Current season label.
        :param team_name: Team name.
        :return: Tuple of (roster_strength, source_label).
        """
        team_name_norm = normalize_team_name(team_name)

        if prev_season is not None:
            val = self.roster_lookup.get((league, prev_season, team_name_norm), None)
            if val is not None:
                return float(val), f"previous_season ({prev_season})"

        if current_season is not None:
            val = self.roster_lookup.get((league, current_season, team_name_norm), None)
            if val is not None:
                return float(val), f"current_season ({current_season})"

        if prev_season is not None:
            avg_val = self.get_league_season_roster_average(league, prev_season)
            if avg_val is not None:
                return float(avg_val), f"league_avg_previous ({prev_season})"

        if current_season is not None:
            avg_val = self.get_league_season_roster_average(league, current_season)
            if avg_val is not None:
                return float(avg_val), f"league_avg_current ({current_season})"

        return None, "missing"

    def build_prediction_input(self, competition_id: str, home_team_id: str, away_team_id: str):
        """
        Build model input features and supporting metadata for one match prediction.

        :param competition_id: Competition identifier.
        :param home_team_id: Home team identifier.
        :param away_team_id: Away team identifier.
        :return: Tuple containing model input DataFrame and auxiliary metadata.
        """
        comp_df = self.df[self.df["competition_id"] == competition_id].copy()
        if comp_df.empty:
            raise ValueError("No data found for selected competition.")

        latest_season_id = self.get_latest_season_id_for_competition(competition_id)
        latest_season_name = self.get_latest_season_name_for_competition(competition_id)

        season_df = comp_df[comp_df["season_id"] == latest_season_id].copy()
        matches_before = season_df.copy()

        league = season_df.iloc[0]["league"]

        home_row = season_df[season_df["home_team_id"] == home_team_id][["home_team_name"]].head(1)
        if home_row.empty:
            home_row = season_df[season_df["away_team_id"] == home_team_id][["away_team_name"]].rename(
                columns={"away_team_name": "home_team_name"}
            ).head(1)

        away_row = season_df[season_df["away_team_id"] == away_team_id][["away_team_name"]].head(1)
        if away_row.empty:
            away_row = season_df[season_df["home_team_id"] == away_team_id][["home_team_name"]].rename(
                columns={"home_team_name": "away_team_name"}
            ).head(1)

        if home_row.empty or away_row.empty:
            raise ValueError("Could not resolve team names.")

        home_team_name = home_row.iloc[0, 0]
        away_team_name = away_row.iloc[0, 0]

        home_stats = self.compute_team_stats(home_team_id, matches_before, home_context=True)
        away_stats = self.compute_team_stats(away_team_id, matches_before, home_context=False)

        rank_map = self.make_rank_map(matches_before, latest_season_id)

        home_rank = 0 if home_stats["played"] == 0 else rank_map.get(home_team_id, 0)
        away_rank = 0 if away_stats["played"] == 0 else rank_map.get(away_team_id, 0)

        official_home = self.get_official_team_stats(competition_id, latest_season_id, home_team_id)
        official_away = self.get_official_team_stats(competition_id, latest_season_id, away_team_id)

        season_norm = normalize_season(latest_season_name)
        prev_season = previous_season(season_norm) if season_norm else None

        home_roster_strength, home_roster_source = self.get_team_roster_strength_with_fallback(
            league=league,
            prev_season=prev_season,
            current_season=season_norm,
            team_name=home_team_name,
        )

        away_roster_strength, away_roster_source = self.get_team_roster_strength_with_fallback(
            league=league,
            prev_season=prev_season,
            current_season=season_norm,
            team_name=away_team_name,
        )

        if home_roster_strength is None or away_roster_strength is None:
            raise ValueError(
                f"Missing roster data even after fallback. "
                f"league={league}, prev_season={prev_season}, current_season={season_norm}, "
                f"home={home_team_name}, away={away_team_name}"
            )

        x_input = pd.DataFrame([{
            "competition_id": competition_id,
            "league": league,
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
            "home_roster_strength": home_roster_strength,
            "away_roster_strength": away_roster_strength,
            "roster_strength_diff": home_roster_strength - away_roster_strength,
        }])

        return (
            x_input,
            home_stats,
            away_stats,
            official_home,
            official_away,
            latest_season_name,
            home_roster_strength,
            away_roster_strength,
            home_roster_source,
            away_roster_source,
        )

    def predict_proba(self, x_input: pd.DataFrame):
        """
        Predict class probabilities for prepared model input.

        :param x_input: Model input DataFrame.
        :return: Predicted probabilities.
        """
        return self.model.predict_proba(x_input)
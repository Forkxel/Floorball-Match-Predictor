import os
import customtkinter as ctk
import pandas as pd
import joblib
from PIL import Image


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "floorball_dataset_processed.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "floorball_model.pkl")
OFFICIAL_STANDINGS_PATH = os.path.join(BASE_DIR, "data", "floorball_official_standings.csv")

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
COMPETITION_LOGOS_DIR = os.path.join(ASSETS_DIR, "competitions")
TEAM_LOGOS_DIR = os.path.join(ASSETS_DIR, "teams")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Processed dataset not found: {DATA_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

if os.path.exists(OFFICIAL_STANDINGS_PATH):
    official_df = pd.read_csv(OFFICIAL_STANDINGS_PATH)
else:
    official_df = pd.DataFrame()

df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")
df = df.sort_values("start_time").reset_index(drop=True)


def sanitize_id_for_filename(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace("\\", "_")


def load_ctk_image(image_path: str, size: tuple[int, int]) -> ctk.CTkImage | None:
    if not os.path.exists(image_path):
        return None
    try:
        pil_image = Image.open(image_path)
        return ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=size)
    except Exception:
        return None


def get_competition_logo(competition_id: str) -> ctk.CTkImage | None:
    filename = f"{sanitize_id_for_filename(competition_id)}.png"
    path = os.path.join(COMPETITION_LOGOS_DIR, filename)
    return load_ctk_image(path, (42, 42))


def get_team_logo(team_id: str) -> ctk.CTkImage | None:
    filename = f"{sanitize_id_for_filename(team_id)}.png"

    for root, _, files in os.walk(TEAM_LOGOS_DIR):
        if filename in files:
            return load_ctk_image(os.path.join(root, filename), (42, 42))

    return None


def get_competition_options() -> list[tuple[str, str]]:
    comp_df = (
        df[["competition_id", "competition_name"]]
        .drop_duplicates()
        .sort_values("competition_name")
        .reset_index(drop=True)
    )
    return list(comp_df.itertuples(index=False, name=None))


def get_latest_season_id_for_competition(competition_id: str) -> str:
    comp_df = df[df["competition_id"] == competition_id].copy()
    if comp_df.empty:
        raise ValueError("No matches found for selected competition.")

    season_order = comp_df.groupby("season_id")["start_time"].max().sort_values()
    return season_order.index[-1]


def get_teams_for_competition(competition_id: str) -> list[tuple[str, str]]:
    comp_df = df[df["competition_id"] == competition_id].copy()
    if comp_df.empty:
        return []

    latest_season_id = get_latest_season_id_for_competition(competition_id)
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

    return list(teams_df.itertuples(index=False, name=None))


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


def get_official_team_stats(competition_id: str, season_id: str, team_id: str) -> dict:
    if official_df.empty:
        return {
            "official_rank": None,
            "official_points": None,
            "official_played": None,
        }

    row = official_df[
        (official_df["competition_id"] == competition_id) &
        (official_df["season_id"] == season_id) &
        (official_df["team_id"] == team_id)
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


def build_prediction_input(competition_id: str, home_team_id: str, away_team_id: str) -> tuple[pd.DataFrame, dict, dict, dict, dict, str]:
    comp_df = df[df["competition_id"] == competition_id].copy()
    if comp_df.empty:
        raise ValueError("No data found for selected competition.")

    latest_season_id = get_latest_season_id_for_competition(competition_id)
    season_df = comp_df[comp_df["season_id"] == latest_season_id].copy()

    matches_before = season_df.copy()

    home_stats = compute_team_stats(home_team_id, matches_before, home_context=True)
    away_stats = compute_team_stats(away_team_id, matches_before, home_context=False)

    rank_map = make_rank_map(matches_before, latest_season_id)

    home_rank = 0 if home_stats["played"] == 0 else rank_map.get(home_team_id, 0)
    away_rank = 0 if away_stats["played"] == 0 else rank_map.get(away_team_id, 0)

    official_home = get_official_team_stats(competition_id, latest_season_id, home_team_id)
    official_away = get_official_team_stats(competition_id, latest_season_id, away_team_id)

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

    return X_input, home_stats, away_stats, official_home, official_away, latest_season_id


class FloorballApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Floorball Match Predictor")
        self.geometry("980x720")
        self.minsize(980, 720)

        self.competition_map: dict[str, str] = {}
        self.team_name_to_id: dict[str, str] = {}

        self.current_competition_logo = None
        self.current_home_logo = None
        self.current_away_logo = None

        self._build_ui()
        self._load_competitions()

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, corner_radius=16)
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="Floorball Match Predictor",
            font=ctk.CTkFont(size=28, weight="bold")
        )
        title.grid(row=0, column=0, padx=20, pady=(18, 6))

        subtitle = ctk.CTkLabel(
            header,
            text="Select a league, then choose a home team and an away team to predict the match outcome.",
            font=ctk.CTkFont(size=14)
        )
        subtitle.grid(row=1, column=0, padx=20, pady=(0, 18))

        body = ctk.CTkFrame(self, corner_radius=16)
        body.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)

        controls = ctk.CTkFrame(body, corner_radius=16)
        controls.grid(row=0, column=0, sticky="ns", padx=(16, 10), pady=16)

        self.result_box = ctk.CTkTextbox(body, corner_radius=16, wrap="word")
        self.result_box.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(10, 16), pady=16)
        self.result_box.insert("1.0", "Prediction results will appear here.")
        self.result_box.configure(state="disabled")

        ctk.CTkLabel(controls, text="League", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=14, pady=(14, 8))

        self.league_logo_label = ctk.CTkLabel(controls, text="", width=42, height=42)
        self.league_logo_label.pack(anchor="w", padx=14, pady=(0, 8))

        self.league_var = ctk.StringVar(value="")
        self.league_combo = ctk.CTkComboBox(
            controls,
            variable=self.league_var,
            values=[],
            command=self.on_league_change,
            width=320,
            state="readonly"
        )
        self.league_combo.pack(anchor="w", padx=14, pady=(0, 18))

        ctk.CTkLabel(controls, text="Home Team", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=14, pady=(0, 8))

        self.home_logo_label = ctk.CTkLabel(controls, text="", width=42, height=42)
        self.home_logo_label.pack(anchor="w", padx=14, pady=(0, 8))

        self.home_var = ctk.StringVar(value="")
        self.home_combo = ctk.CTkComboBox(
            controls,
            variable=self.home_var,
            values=[],
            command=self.on_home_change,
            width=320,
            state="disabled"
        )
        self.home_combo.pack(anchor="w", padx=14, pady=(0, 18))

        ctk.CTkLabel(controls, text="Away Team", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=14, pady=(0, 8))

        self.away_logo_label = ctk.CTkLabel(controls, text="", width=42, height=42)
        self.away_logo_label.pack(anchor="w", padx=14, pady=(0, 8))

        self.away_var = ctk.StringVar(value="")
        self.away_combo = ctk.CTkComboBox(
            controls,
            variable=self.away_var,
            values=[],
            command=self.on_away_change,
            width=320,
            state="disabled"
        )
        self.away_combo.pack(anchor="w", padx=14, pady=(0, 18))

        self.predict_button = ctk.CTkButton(
            controls,
            text="Predict Match",
            command=self.predict_match,
            width=320,
            height=42
        )
        self.predict_button.pack(anchor="w", padx=14, pady=(8, 10))

    def _set_result_text(self, text: str):
        self.result_box.configure(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.insert("1.0", text)
        self.result_box.configure(state="disabled")

    def _load_competitions(self):
        display_values = []
        for comp_id, comp_name in get_competition_options():
            display_values.append(comp_name)
            self.competition_map[comp_name] = comp_id

        self.league_combo.configure(values=display_values)

    def on_league_change(self, selected_name: str):
        competition_id = self.competition_map.get(selected_name)
        if not competition_id:
            return

        self.current_competition_logo = get_competition_logo(competition_id)
        if self.current_competition_logo is not None:
            self.league_logo_label.configure(image=self.current_competition_logo, text="")
        else:
            self.league_logo_label.configure(image=None, text="No logo")

        teams = get_teams_for_competition(competition_id)
        team_names = [team_name for _, team_name in teams]
        self.team_name_to_id = {team_name: team_id for team_id, team_name in teams}

        self.home_combo.configure(values=team_names, state="normal")
        self.away_combo.configure(values=team_names, state="normal")

        self.home_var.set("")
        self.away_var.set("")

        self.home_logo_label.configure(image=None, text="")
        self.away_logo_label.configure(image=None, text="")

        self._set_result_text("League selected. Now choose a home team and an away team.")

    def on_home_change(self, selected_team: str):
        team_id = self.team_name_to_id.get(selected_team)
        self.current_home_logo = get_team_logo(team_id) if team_id else None

        if self.current_home_logo is not None:
            self.home_logo_label.configure(image=self.current_home_logo, text="")
        else:
            self.home_logo_label.configure(image=None, text="No logo")

        if selected_team and selected_team == self.away_var.get():
            self.away_var.set("")
            self.away_logo_label.configure(image=None, text="")

    def on_away_change(self, selected_team: str):
        team_id = self.team_name_to_id.get(selected_team)
        self.current_away_logo = get_team_logo(team_id) if team_id else None

        if self.current_away_logo is not None:
            self.away_logo_label.configure(image=self.current_away_logo, text="")
        else:
            self.away_logo_label.configure(image=None, text="No logo")

        if selected_team and selected_team == self.home_var.get():
            self.home_var.set("")
            self.home_logo_label.configure(image=None, text="")

    def predict_match(self):
        league_name = self.league_var.get().strip()
        home_team_name = self.home_var.get().strip()
        away_team_name = self.away_var.get().strip()

        if not league_name:
            self._set_result_text("Please select a league first.")
            return

        if not home_team_name or not away_team_name:
            self._set_result_text("Please select both teams.")
            return

        if home_team_name == away_team_name:
            self._set_result_text("Home team and away team must be different.")
            return

        competition_id = self.competition_map[league_name]
        home_team_id = self.team_name_to_id.get(home_team_name)
        away_team_id = self.team_name_to_id.get(away_team_name)

        if not home_team_id or not away_team_id:
            self._set_result_text("Could not resolve selected teams.")
            return

        try:
            X_input, home_stats, away_stats, official_home, official_away, season_id = build_prediction_input(
                competition_id=competition_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id
            )

            prob = model.predict_proba(X_input)[0]
            home_win_prob = float(prob[1])
            not_home_win_prob = float(prob[0])

            predicted_label = "Home Win" if home_win_prob >= 0.5 else "Home Do Not Win"

            output = []
            output.append(f"League: {league_name}")
            output.append(f"Season: {season_id}")
            output.append(f"Home Team: {home_team_name}")
            output.append(f"Away Team: {away_team_name}")
            output.append("")
            output.append("Prediction")
            output.append(f"- Home win probability: {home_win_prob:.2%}")
            output.append(f"- Away / not-home-win probability: {not_home_win_prob:.2%}")
            output.append(f"- Predicted outcome: {predicted_label}")
            output.append("")

            output.append("Official Regular Season Stats")
            output.append(f"- Home official rank: {official_home['official_rank']}")
            output.append(f"- Home official points: {official_home['official_points']}")
            output.append(f"- Home official matches played: {official_home['official_played']}")
            output.append(f"- Away official rank: {official_away['official_rank']}")
            output.append(f"- Away official points: {official_away['official_points']}")
            output.append(f"- Away official matches played: {official_away['official_played']}")
            output.append("")

            output.append("Model Computed Stats")
            output.append("Home Team")
            output.append(f"- Computed points: {home_stats['points']}")
            output.append(f"- Computed matches played: {home_stats['played']}")
            output.append(f"- Computed goal difference: {home_stats['goal_diff']}")
            output.append(f"- Last 5 form: {home_stats['form_last5']:.2f}")
            output.append(f"- Last 5 goals scored average: {home_stats['goals_for_avg_last5']:.2f}")
            output.append(f"- Last 5 goals conceded average: {home_stats['goals_against_avg_last5']:.2f}")
            output.append(f"- Last 5 home form: {home_stats['context_form_last5']:.2f}")
            output.append("")
            output.append("Away Team")
            output.append(f"- Computed points: {away_stats['points']}")
            output.append(f"- Computed matches played: {away_stats['played']}")
            output.append(f"- Computed goal difference: {away_stats['goal_diff']}")
            output.append(f"- Last 5 form: {away_stats['form_last5']:.2f}")
            output.append(f"- Last 5 goals scored average: {away_stats['goals_for_avg_last5']:.2f}")
            output.append(f"- Last 5 goals conceded average: {away_stats['goals_against_avg_last5']:.2f}")
            output.append(f"- Last 5 away form: {away_stats['context_form_last5']:.2f}")

            self._set_result_text("\n".join(output))

        except Exception as exc:
            self._set_result_text(f"Prediction error: {exc}")


if __name__ == "__main__":
    app = FloorballApp()
    app.mainloop()
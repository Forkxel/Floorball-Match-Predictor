import customtkinter as ctk
from PIL import Image

from src.app.services import FloorballService


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FloorballApp(ctk.CTk):
    """
    Main GUI application for match prediction.
    """

    def __init__(self):
        """
        Initialize the application window, service layer, and UI state.

        :return: None.
        """
        super().__init__()

        self.service = FloorballService()

        self.title("Floorball Match Predictor")
        window_width = 980
        window_height = 760

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)

        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.resizable(False, False)

        self.competition_map = {}
        self.team_name_to_id = {}
        self.all_team_names = []

        self.current_competition_logo = None
        self.current_home_logo = None
        self.current_away_logo = None
        self.result_home_logo = None
        self.result_away_logo = None

        self.empty_logo = ctk.CTkImage(
            light_image=Image.new("RGBA", (42, 42), (0, 0, 0, 0)),
            dark_image=Image.new("RGBA", (42, 42), (0, 0, 0, 0)),
            size=(42, 42),
        )

        self.empty_logo_large = ctk.CTkImage(
            light_image=Image.new("RGBA", (110, 110), (0, 0, 0, 0)),
            dark_image=Image.new("RGBA", (110, 110), (0, 0, 0, 0)),
            size=(110, 110),
        )

        self._build_ui()
        self.after(100, self._reset_result_panel)
        self._load_competitions()

    def _build_ui(self) -> None:
        """
        Build the main application layout and widgets.

        :return: None.
        """
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, corner_radius=16)
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="Floorball Match Predictor",
            font=ctk.CTkFont(size=28, weight="bold"),
        )
        title.grid(row=0, column=0, padx=20, pady=(18, 6))

        subtitle = ctk.CTkLabel(
            header,
            text="Select a league, then choose a home team and an away team to predict the match outcome.",
            font=ctk.CTkFont(size=14),
        )
        subtitle.grid(row=1, column=0, padx=20, pady=(0, 18))

        body = ctk.CTkFrame(self, corner_radius=16)
        body.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(1, weight=1)

        controls = ctk.CTkFrame(body, corner_radius=16)
        controls.grid(row=0, column=0, sticky="ns", padx=(16, 10), pady=16)

        self.result_panel = ctk.CTkFrame(body, corner_radius=16)
        self.result_panel.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(10, 16), pady=16)
        self.result_panel.grid_columnconfigure(0, weight=1)
        self.result_panel.grid_rowconfigure(5, weight=1)

        self.result_title = ctk.CTkLabel(
            self.result_panel,
            text="",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.result_title.grid(row=0, column=0, pady=(24, 0), padx=20)

        self.teams_frame = ctk.CTkFrame(self.result_panel, fg_color="transparent")
        self.teams_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(55, 10))
        self.teams_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.result_home_logo_label = ctk.CTkLabel(
            self.teams_frame,
            text="",
            image=self.empty_logo_large,
        )
        self.result_home_logo_label.grid(row=0, column=0, pady=(0, 8))

        self.vs_label = ctk.CTkLabel(
            self.teams_frame,
            text="",
            font=ctk.CTkFont(size=28, weight="bold"),
        )
        self.vs_label.grid(row=0, column=1, pady=(0, 8))

        self.result_away_logo_label = ctk.CTkLabel(
            self.teams_frame,
            text="",
            image=self.empty_logo_large,
        )
        self.result_away_logo_label.grid(row=0, column=2, pady=(0, 8))

        self.result_home_name_label = ctk.CTkLabel(
            self.teams_frame,
            text="",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.result_home_name_label.grid(row=1, column=0, padx=10, pady=(0, 4))

        self.result_away_name_label = ctk.CTkLabel(
            self.teams_frame,
            text="",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.result_away_name_label.grid(row=1, column=2, padx=10, pady=(0, 4))

        self.result_home_pct_label = ctk.CTkLabel(
            self.teams_frame,
            text="",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.result_home_pct_label.grid(row=2, column=0, padx=10, pady=(0, 10))

        self.result_away_pct_label = ctk.CTkLabel(
            self.teams_frame,
            text="",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        self.result_away_pct_label.grid(row=2, column=2, padx=10, pady=(0, 10))

        self.bar_frame = ctk.CTkFrame(self.result_panel, corner_radius=12, fg_color="#2b2b2b")
        self.bar_frame.grid(row=2, column=0, sticky="ew", padx=30, pady=(22, 10))
        self.bar_frame.grid_columnconfigure(0, weight=1)

        self.bar_canvas = ctk.CTkCanvas(
            self.bar_frame,
            height=28,
            highlightthickness=0,
            bd=0,
            bg="#2b2b2b",
        )
        self.bar_canvas.pack(fill="x", expand=True)

        self.result_pick_label = ctk.CTkLabel(
            self.result_panel,
            text="",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        self.result_pick_label.grid(row=3, column=0, pady=(18, 6), padx=20)

        self.result_subtitle_label = ctk.CTkLabel(
            self.result_panel,
            text="",
            font=ctk.CTkFont(size=14),
        )
        self.result_subtitle_label.grid(row=4, column=0, pady=(6, 20), padx=20)

        ctk.CTkLabel(
            controls,
            text="League",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", padx=14, pady=(14, 8))

        self.league_logo_label = ctk.CTkLabel(
            controls,
            text="",
            width=42,
            height=42,
            image=self.empty_logo,
        )
        self.league_logo_label.pack(anchor="w", padx=14, pady=(0, 8))

        self.league_var = ctk.StringVar(value="")
        self.league_combo = ctk.CTkComboBox(
            controls,
            variable=self.league_var,
            values=[],
            command=self.on_league_change,
            width=320,
            state="readonly",
        )
        self.league_combo.pack(anchor="w", padx=14, pady=(0, 18))

        ctk.CTkLabel(
            controls,
            text="Home Team",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", padx=14, pady=(0, 8))

        self.home_logo_label = ctk.CTkLabel(
            controls,
            text="",
            width=42,
            height=42,
            image=self.empty_logo,
        )
        self.home_logo_label.pack(anchor="w", padx=14, pady=(0, 8))

        self.home_var = ctk.StringVar(value="")
        self.home_combo = ctk.CTkComboBox(
            controls,
            variable=self.home_var,
            values=[],
            command=self.on_home_change,
            width=320,
            state="disabled",
        )
        self.home_combo.pack(anchor="w", padx=14, pady=(0, 18))

        ctk.CTkLabel(
            controls,
            text="Away Team",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w", padx=14, pady=(0, 8))

        self.away_logo_label = ctk.CTkLabel(
            controls,
            text="",
            width=42,
            height=42,
            image=self.empty_logo,
        )
        self.away_logo_label.pack(anchor="w", padx=14, pady=(0, 8))

        self.away_var = ctk.StringVar(value="")
        self.away_combo = ctk.CTkComboBox(
            controls,
            variable=self.away_var,
            values=[],
            command=self.on_away_change,
            width=320,
            state="disabled",
        )
        self.away_combo.pack(anchor="w", padx=14, pady=(0, 18))

        self.predict_button = ctk.CTkButton(
            controls,
            text="Predict Match",
            command=self.predict_match,
            width=320,
            height=42,
        )
        self.predict_button.pack(anchor="w", padx=14, pady=(8, 10))

    def _show_probability_bar(self) -> None:
        """
        Show the probability bar widget.

        :return: None.
        """
        self.bar_frame.grid()
        self.bar_canvas.pack(fill="x", expand=True)

    def _hide_probability_bar(self) -> None:
        """
        Hide and clear the probability bar widget.

        :return: None.
        """
        self.bar_canvas.delete("all")
        self.bar_frame.grid_remove()

    def _draw_probability_bar(self, home_prob: float, away_prob: float) -> None:
        """
        Draw the win probability bar for both teams.

        :param home_prob: Predicted home win probability.
        :param away_prob: Predicted away win probability.
        :return: None.
        """
        self._show_probability_bar()
        self.bar_canvas.delete("all")
        self.bar_canvas.update_idletasks()

        width = self.bar_canvas.winfo_width()
        height = self.bar_canvas.winfo_height()

        if width <= 1:
            width = 500
        if height <= 1:
            height = 28

        home_width = int(width * home_prob)

        if home_prob >= away_prob:
            home_color = "#22c55e"
            away_color = "#ef4444"
        else:
            home_color = "#ef4444"
            away_color = "#22c55e"

        self.bar_canvas.create_rectangle(
            0, 0, home_width, height,
            fill=home_color, outline=home_color,
        )
        self.bar_canvas.create_rectangle(
            home_width, 0, width, height,
            fill=away_color, outline=away_color,
        )
        self.bar_canvas.create_line(
            home_width, 0, home_width, height,
            fill="white",
            width=2,
        )

    def _reset_result_panel(self, message: str = "") -> None:
        """
        Reset the result panel to its default empty state.

        :param message: Optional title message.
        :return: None.
        """
        self.result_title.configure(text=message)

        self.result_home_logo = None
        self.result_away_logo = None

        self.result_home_logo_label.configure(image=self.empty_logo_large, text="")
        self.result_away_logo_label.configure(image=self.empty_logo_large, text="")

        self.vs_label.configure(text="")
        self.result_home_name_label.configure(text="")
        self.result_away_name_label.configure(text="")
        self.result_home_pct_label.configure(text="")
        self.result_away_pct_label.configure(text="")
        self.result_pick_label.configure(text="")
        self.result_subtitle_label.configure(text="")

        self._hide_probability_bar()

    def _show_result_error(self, message: str) -> None:
        """
        Show an error message in the result panel.

        :param message: Error message to display.
        :return: None.
        """
        self.result_title.configure(text="")

        self.result_home_logo = None
        self.result_away_logo = None

        self.result_home_logo_label.configure(image=self.empty_logo_large, text="")
        self.result_away_logo_label.configure(image=self.empty_logo_large, text="")

        self.vs_label.configure(text="")
        self.result_home_name_label.configure(text="")
        self.result_away_name_label.configure(text="")
        self.result_home_pct_label.configure(text="")
        self.result_away_pct_label.configure(text="")
        self.result_pick_label.configure(text="")
        self.result_subtitle_label.configure(text=message)

        self._hide_probability_bar()

    def _clear_home_logo(self) -> None:
        """
        Clear the selected home team logo.

        :return: None.
        """
        self.current_home_logo = None
        self.home_logo_label.configure(image=self.empty_logo, text="")

    def _clear_away_logo(self) -> None:
        """
        Clear the selected away team logo.

        :return: None.
        """
        self.current_away_logo = None
        self.away_logo_label.configure(image=self.empty_logo, text="")

    def _clear_league_logo(self) -> None:
        """
        Clear the selected competition logo.

        :return: None.
        """
        self.current_competition_logo = None
        self.league_logo_label.configure(image=self.empty_logo, text="")

    def _refresh_team_comboboxes(self) -> None:
        """
        Refresh home and away team dropdown options.

        :return: None.
        """
        selected_home = self.home_var.get().strip()
        selected_away = self.away_var.get().strip()

        home_values = [team for team in self.all_team_names if team != selected_away]
        away_values = [team for team in self.all_team_names if team != selected_home]

        if selected_home and selected_home not in home_values:
            selected_home = ""
            self.home_var.set("")
            self._clear_home_logo()

        if selected_away and selected_away not in away_values:
            selected_away = ""
            self.away_var.set("")
            self._clear_away_logo()

        self.home_combo.configure(values=home_values)
        self.away_combo.configure(values=away_values)

        self.home_combo.set(selected_home)
        self.away_combo.set(selected_away)

    def _load_competitions(self) -> None:
        """
        Load competition options into the league dropdown.

        :return: None.
        """
        display_values = []

        for comp_id, comp_name in self.service.get_competition_options():
            display_values.append(comp_name)
            self.competition_map[comp_name] = comp_id

        self.league_combo.configure(values=display_values)

    def on_league_change(self, selected_name: str) -> None:
        """
        Handle league selection change.

        :param selected_name: Selected competition name.
        :return: None.
        """
        competition_id = self.competition_map.get(selected_name)
        if not competition_id:
            return

        self.current_competition_logo = self.service.get_competition_logo(competition_id)
        if self.current_competition_logo is not None:
            self.league_logo_label.configure(image=self.current_competition_logo, text="")
        else:
            self._clear_league_logo()

        teams = self.service.get_teams_for_competition(competition_id)
        self.all_team_names = [team_name for _, team_name in teams]
        self.team_name_to_id = {team_name: team_id for team_id, team_name in teams}

        self.home_var.set("")
        self.away_var.set("")
        self.home_combo.set("")
        self.away_combo.set("")

        self._clear_home_logo()
        self._clear_away_logo()

        self.home_combo.configure(state="normal")
        self.away_combo.configure(state="normal")

        self._refresh_team_comboboxes()
        self._reset_result_panel()

    def on_home_change(self, selected_team: str) -> None:
        """
        Handle home team selection change.

        :param selected_team: Selected home team name.
        :return: None.
        """
        if not selected_team:
            self._clear_home_logo()
            self._refresh_team_comboboxes()
            return

        if selected_team == self.away_var.get().strip():
            self._show_result_error("Home team and away team must be different.")
            self.home_var.set("")
            self.home_combo.set("")
            self._clear_home_logo()
            self._refresh_team_comboboxes()
            return

        team_id = self.team_name_to_id.get(selected_team)
        self.current_home_logo = self.service.get_team_logo(team_id, (42, 42)) if team_id else None

        if self.current_home_logo is not None:
            self.home_logo_label.configure(image=self.current_home_logo, text="")
        else:
            self._clear_home_logo()

        self._refresh_team_comboboxes()

    def on_away_change(self, selected_team: str) -> None:
        """
        Handle away team selection change.

        :param selected_team: Selected away team name.
        :return: None.
        """
        if not selected_team:
            self._clear_away_logo()
            self._refresh_team_comboboxes()
            return

        if selected_team == self.home_var.get().strip():
            self._show_result_error("Home team and away team must be different.")
            self.away_var.set("")
            self.away_combo.set("")
            self._clear_away_logo()
            self._refresh_team_comboboxes()
            return

        team_id = self.team_name_to_id.get(selected_team)
        self.current_away_logo = self.service.get_team_logo(team_id, (42, 42)) if team_id else None

        if self.current_away_logo is not None:
            self.away_logo_label.configure(image=self.current_away_logo, text="")
        else:
            self._clear_away_logo()

        self._refresh_team_comboboxes()

    def predict_match(self) -> None:
        """
        Build model input, run prediction, and update the result panel.

        :return: None.
        """
        league_name = self.league_var.get().strip()
        home_team_name = self.home_var.get().strip()
        away_team_name = self.away_var.get().strip()

        if not league_name:
            self._show_result_error("Please select a league first.")
            return

        if not home_team_name or not away_team_name:
            self._show_result_error("Please select both teams.")
            return

        if home_team_name == away_team_name:
            self._show_result_error("Home team and away team must be different.")
            return

        competition_id = self.competition_map[league_name]
        home_team_id = self.team_name_to_id.get(home_team_name)
        away_team_id = self.team_name_to_id.get(away_team_name)

        if not home_team_id or not away_team_id:
            self._show_result_error("Could not resolve selected teams.")
            return

        try:
            (
                x_input,
                home_stats,
                away_stats,
                official_home,
                official_away,
                season_name,
                home_roster_strength,
                away_roster_strength,
                home_roster_source,
                away_roster_source,
            ) = self.service.build_prediction_input(
                competition_id=competition_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
            )

            prob = self.service.predict_proba(x_input)[0]
            home_win_prob = float(prob[1])
            away_win_prob = float(prob[0])

            if home_win_prob >= 0.70:
                suggested_pick = f"Suggested bet: {home_team_name}"
            elif home_win_prob <= 0.30:
                suggested_pick = f"Suggested bet: {away_team_name}"
            else:
                suggested_pick = "Suggested bet: No recommended bet"

            self.result_title.configure(text=f"{season_name}")
            self.vs_label.configure(text="VS")

            self.result_home_name_label.configure(text=home_team_name)
            self.result_away_name_label.configure(text=away_team_name)

            self.result_home_pct_label.configure(text=f"{home_win_prob:.2%}")
            self.result_away_pct_label.configure(text=f"{away_win_prob:.2%}")

            self.result_home_logo = self.service.get_team_logo(home_team_id, (110, 110))
            self.result_away_logo = self.service.get_team_logo(away_team_id, (110, 110))

            if self.result_home_logo is not None:
                self.result_home_logo_label.configure(image=self.result_home_logo, text="")
            else:
                self.result_home_logo_label.configure(image=self.empty_logo_large, text="")

            if self.result_away_logo is not None:
                self.result_away_logo_label.configure(image=self.result_away_logo, text="")
            else:
                self.result_away_logo_label.configure(image=self.empty_logo_large, text="")

            self.result_pick_label.configure(text=suggested_pick)
            self.result_subtitle_label.configure(text="")

            self._draw_probability_bar(home_win_prob, away_win_prob)

        except Exception as exc:
            self._show_result_error(str(exc))


if __name__ == "__main__":
    app = FloorballApp()
    app.mainloop()
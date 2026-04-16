import re
from typing import Any


def clean_text(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def guess_scores_table(page: Any):
    candidates = page.locator("#points table.sortable-table")
    if candidates.count() > 0:
        return candidates.first

    for locator in [page.locator("table.sortable-table"), page.locator("table")]:
        try:
            for i in range(locator.count()):
                table = locator.nth(i)
                if table.locator("thead tr th").count() >= 5:
                    return table
        except Exception:
            pass

    raise RuntimeError("No scores table found.")


def extract_player_and_team_from_player_cell(cell: Any) -> tuple[str, str]:
    try:
        text = cell.inner_text()
    except Exception:
        text = ""

    parts = [clean_text(x) for x in text.split("\n") if clean_text(x)]
    player_name = parts[0] if parts else ""
    team_name = ""

    if len(parts) >= 2:
        team_name = re.sub(r"#\d+\b", "", parts[1]).strip()

    return player_name, team_name
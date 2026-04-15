from pathlib import Path
import re
import pandas as pd
from playwright.sync_api import sync_playwright


BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "players"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "https://fliiga.com/en/statistics/men/"
TARGET_SEASON_VALUE = "2025-2026"
OUTPUT_PATH = OUTPUT_DIR / "fliiga_players_2025_2026.csv"


def clean_text(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_numeric(value: str):
    value = clean_text(value).replace(",", ".")
    return pd.to_numeric(value, errors="coerce")


def try_accept_cookies(page) -> None:
    for text in ["Accept", "I agree", "OK", "Allow all"]:
        try:
            btn = page.get_by_text(text, exact=True)
            if btn.count() > 0:
                btn.first.click(timeout=1500)
                page.wait_for_timeout(500)
                return
        except Exception:
            pass


def click_scores_tab(page) -> None:
    candidates = [
        page.get_by_role("button", name="Scores"),
        page.get_by_text("Scores", exact=True),
    ]
    for c in candidates:
        try:
            if c.count() > 0:
                c.first.click(timeout=3000)
                page.wait_for_timeout(1500)
                return
        except Exception:
            pass
    raise RuntimeError("Couldnt click on tab Scores.")


def set_regular_season(page) -> None:
    selects = page.locator(".season-select select")
    if selects.count() == 0:
        selects = page.locator("select")

    if selects.count() == 0:
        raise RuntimeError("No season select on F-Liiga page.")

    sel = selects.first
    sel.select_option(value=TARGET_SEASON_VALUE, timeout=5000)
    page.wait_for_timeout(2000)


def guess_scores_table(page):
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

    raise RuntimeError("No F-Liiga scores table.")


def extract_player_and_team_from_player_cell(cell) -> tuple[str, str]:
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


def scrape_fliiga_players() -> pd.DataFrame:
    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1200})

        print("Opening F-Liiga page...")
        page.goto(URL, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(2500)
        try_accept_cookies(page)

        click_scores_tab(page)
        set_regular_season(page)

        table = guess_scores_table(page)

        headers = []
        ths = table.locator("thead tr th")
        for i in range(ths.count()):
            headers.append(clean_text(ths.nth(i).inner_text()))
        print("Detected headers:", headers)

        trs = table.locator("tbody tr")
        row_count = trs.count()
        print("Table rows:", row_count)

        for i in range(row_count):
            tr = trs.nth(i)
            tds = tr.locator("td")
            vals = [clean_text(tds.nth(j).inner_text()) for j in range(tds.count())]

            if len(vals) < 11:
                continue

            player_name, team_name = extract_player_and_team_from_player_cell(tds.nth(1))

            rows.append({
                "source": "fliiga",
                "league": "finland",
                "season": "2025-2026",
                "team_name": team_name if team_name else "UNKNOWN_TEAM",
                "player_name": player_name,
                "gp": parse_numeric(vals[2]),
                "goals": parse_numeric(vals[3]),
                "assists": parse_numeric(vals[4]),
                "points": parse_numeric(vals[5]),
                "shots": parse_numeric(vals[8]),
                "pim": parse_numeric(vals[7]),
                "plus_minus": parse_numeric(vals[10]),
            })

        browser.close()

    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    return df


def main():
    df = scrape_fliiga_players()
    if df.empty:
        raise RuntimeError("No data find for F-Liiga.")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
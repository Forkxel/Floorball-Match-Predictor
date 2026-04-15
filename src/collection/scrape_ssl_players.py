from pathlib import Path
import re
import time
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "players"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SSL_URL = "https://ssl.se/game-stats/players/summary?count=25"
TARGET_SEASON = "2025/2026"
TARGET_LEAGUE = "SSL Men"
TARGET_MATCH_TYPE = "Series Match"
TARGET_TEAM = "All Teams"


def clean_text(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def safe_inner_text(locator) -> str:
    try:
        return clean_text(locator.inner_text())
    except Exception:
        return ""


def try_click_cookie_buttons(page) -> None:
    candidates = [
        "button:has-text('Accept')",
        "button:has-text('I agree')",
        "button:has-text('Godkänn')",
        "button:has-text('OK')",
    ]
    for sel in candidates:
        try:
            if page.locator(sel).count() > 0:
                page.locator(sel).first.click(timeout=1500)
                page.wait_for_timeout(500)
                return
        except Exception:
            pass


def open_select_and_choose(page, expected_value: str) -> bool:
    candidates = [
        page.get_by_text(expected_value, exact=True),
        page.get_by_role("button", name=expected_value),
        page.get_by_role("link", name=expected_value),
        page.locator(f"text='{expected_value}'"),
    ]

    for candidate in candidates:
        try:
            if candidate.count() > 0:
                candidate.first.click(timeout=2000)
                page.wait_for_timeout(800)
                return True
        except Exception:
            pass
    return False


def select_visible_option(page, option_text: str) -> bool:
    candidates = [
        page.get_by_role("option", name=option_text),
        page.get_by_text(option_text, exact=True),
        page.locator(f"text='{option_text}'"),
    ]

    for candidate in candidates:
        try:
            if candidate.count() > 0:
                candidate.first.click(timeout=3000)
                page.wait_for_timeout(1500)
                return True
        except Exception:
            pass
    return False


def set_filter(page, current_or_button_text: str, target_option_text: str) -> None:
    opened = open_select_and_choose(page, current_or_button_text)
    if not opened:
        return
    select_visible_option(page, target_option_text)


def guess_table(page):
    table_candidates = [
        page.locator("table"),
        page.locator("[role='table']"),
    ]

    for group in table_candidates:
        try:
            count = group.count()
        except Exception:
            count = 0

        for i in range(count):
            table = group.nth(i)
            try:
                row_count = table.locator("tbody tr").count()
                if row_count >= 5:
                    return table
            except Exception:
                continue

    raise RuntimeError("Couldn't find table with player stats.")


def extract_headers(table) -> list[str]:
    headers = []

    ths = table.locator("thead tr th")
    if ths.count() > 0:
        for i in range(ths.count()):
            headers.append(clean_text(ths.nth(i).inner_text()))
        return headers

    first_row_cells = table.locator("tr").first.locator("th, td")
    for i in range(first_row_cells.count()):
        headers.append(clean_text(first_row_cells.nth(i).inner_text()))
    return headers


def parse_row(cells_text: list[str], season: str) -> dict | None:
    values = [clean_text(v) for v in cells_text if clean_text(v) != ""]

    if len(values) < 7:
        return None

    if re.fullmatch(r"\d+\.", values[0]) or re.fullmatch(r"\d+", values[0]):
        values = values[1:]

    if len(values) < 7:
        return None

    player_name = values[0]
    team_name = values[1]

    numeric_tail = values[2:]

    if len(numeric_tail) < 5:
        return None

    def to_num(v):
        v = v.replace(",", ".")
        return pd.to_numeric(v, errors="coerce")

    gp = to_num(numeric_tail[0])
    goals = to_num(numeric_tail[1]) if len(numeric_tail) > 1 else None
    assists = to_num(numeric_tail[2]) if len(numeric_tail) > 2 else None
    points = to_num(numeric_tail[3]) if len(numeric_tail) > 3 else None
    pim = to_num(numeric_tail[4]) if len(numeric_tail) > 4 else None

    if pd.isna(gp) and pd.isna(goals) and pd.isna(assists) and pd.isna(points):
        return None

    return {
        "source": "ssl",
        "league": "sweden",
        "season": season,
        "team_name": team_name,
        "player_name": player_name,
        "gp": gp,
        "goals": goals,
        "assists": assists,
        "points": points,
        "shots": None,
        "pim": pim,
        "plus_minus": None,
    }


def extract_rows_from_table(table, season: str) -> list[dict]:
    rows = []
    trs = table.locator("tbody tr")
    row_count = trs.count()

    for i in range(row_count):
        tr = trs.nth(i)
        cells = tr.locator("td")
        cell_texts = []

        for j in range(cells.count()):
            txt = safe_inner_text(cells.nth(j))
            cell_texts.append(txt)

        row = parse_row(cell_texts, season)
        if row:
            rows.append(row)

    return rows


def click_next_page(page) -> bool:
    candidates = [
        page.get_by_role("button", name="Next"),
        page.get_by_role("link", name="Next"),
        page.get_by_text("Next", exact=True),
        page.locator("button[aria-label*='Next']"),
        page.locator("a[aria-label*='Next']"),
        page.locator("button:has(svg)"),
    ]

    for candidate in candidates:
        try:
            if candidate.count() > 0 and candidate.first.is_enabled():
                candidate.first.click(timeout=2000)
                page.wait_for_timeout(2000)
                return True
        except Exception:
            pass

    return False


def scrape_ssl_players(season: str = TARGET_SEASON) -> pd.DataFrame:
    all_rows = []
    seen_keys = set()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1200})

        print("Opening SSL page...")
        page.goto(SSL_URL, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(3000)

        try_click_cookie_buttons(page)

        print("Setting filters...")
        set_filter(page, TARGET_SEASON, season)
        set_filter(page, TARGET_LEAGUE, TARGET_LEAGUE)
        set_filter(page, TARGET_MATCH_TYPE, TARGET_MATCH_TYPE)
        set_filter(page, TARGET_TEAM, TARGET_TEAM)

        try:
            page.goto("https://ssl.se/game-stats/players/summary?count=500", wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(3000)
            try_click_cookie_buttons(page)
            set_filter(page, season, season)
        except Exception:
            pass

        visited_pages = 0
        max_pages = 20

        while visited_pages < max_pages:
            visited_pages += 1
            print(f"Reading page {visited_pages}...")

            try:
                table = guess_table(page)
            except RuntimeError:
                break

            headers = extract_headers(table)
            if headers:
                print("Detected headers:", headers)

            rows = extract_rows_from_table(table, season)

            new_count = 0
            for row in rows:
                key = (row["season"], row["team_name"], row["player_name"], row["gp"], row["points"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_rows.append(row)
                    new_count += 1

            print(f"Added {new_count} new rows.")

            if new_count == 0:
                break

            moved = click_next_page(page)
            if not moved:
                break

        browser.close()

    df = pd.DataFrame(all_rows)

    if not df.empty:
        df = df.drop_duplicates().reset_index(drop=True)
        for col in ["gp", "goals", "assists", "points", "shots", "pim", "plus_minus"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main():
    df = scrape_ssl_players(TARGET_SEASON)

    if df.empty:
        raise RuntimeError("No data find for SSL player stats.")

    season_slug = TARGET_SEASON.replace("/", "_")
    output_path = OUTPUT_DIR / f"ssl_players_{season_slug}.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(df)}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
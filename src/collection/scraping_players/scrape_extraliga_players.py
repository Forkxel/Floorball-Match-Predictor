from pathlib import Path
import re
from urllib.parse import urlencode
import pandas as pd
from playwright.sync_api import sync_playwright


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw" / "players"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEASON_CONFIG = {
    "2025/2026": {
        "base_url": "https://www.ceskyflorbal.cz/competition/detail/statistics/8XM1?competitionFisId=4487&divisionAlias=8XM1-A",
        "part_value": "division-5397",
    },
    "2024/2025": {
        "base_url": "https://www.ceskyflorbal.cz/competition/detail/statistics/8XM1?competitionFisId=4169&divisionAlias=8XM1-A",
        "part_value": "division-5000",
    },
    "2023/2024": {
        "base_url": "https://www.ceskyflorbal.cz/competition/detail/statistics/8XM1?competitionFisId=3843&divisionAlias=8XM1-A",
        "part_value": "division-4611",
    },
}

TYPE_VALUE = "stats_all"
ITEMS_PER_PAGE = "100"


def clean_text(value: str) -> str:
    """
    Normalize text value by trimming whitespace.

    :param value: Raw text value.
    :return: Cleaned text string.
    """
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_numeric(value: str):
    """
    Convert text value to numeric.

    :param value: Raw numeric string.
    :return: Parsed numeric value or NaN.
    """
    value = clean_text(value).replace(",", ".")
    return pd.to_numeric(value, errors="coerce")


def try_accept_cookies(page) -> None:
    """
    Try to accept the cookie banner if it is visible.

    :param page: Playwright page instance.
    :return: None.
    """
    for text in ["Přijmout", "Souhlasím", "Accept", "OK"]:
        try:
            btn = page.get_by_text(text, exact=True)
            if btn.count() > 0:
                btn.first.click(timeout=2000)
                page.wait_for_timeout(800)
                return
        except Exception:
            pass


def wait_for_player_statistics_ready(page) -> None:
    """
    Wait until the player statistics page is ready.

    :param page: Playwright page instance.
    :return: None.
    """
    page.wait_for_selector("#frm-playerStatisticsFilter-team", state="attached", timeout=15000)
    page.wait_for_selector("#frm-playerStatisticsFilter-part", state="attached", timeout=15000)
    page.wait_for_selector("#frm-playerStatisticsFilter-type", state="attached", timeout=15000)
    page.wait_for_selector("#snippet-playerStatisticsGrid-playerStatisticsGrid table", state="attached", timeout=15000)
    page.wait_for_timeout(1000)


def build_filtered_url(base_url: str, part_value: str, team_value: str | None = None, page_num: int | None = None) -> str:
    """
    Build filtered statistics URL.

    :param base_url: Base competition statistics URL.
    :param part_value: Division or competition part value.
    :param team_value: Optional team filter value.
    :param page_num: Optional pagination page number.
    :return: Final filtered URL.
    """
    params = {
        "playerStatisticsFilter[part]": part_value,
        "playerStatisticsFilter[type]": TYPE_VALUE,
        "playerStatisticsGrid-itemsPerPage": ITEMS_PER_PAGE,
    }

    if team_value is not None:
        params["playerStatisticsFilter[team]"] = str(team_value)

    if page_num is not None:
        params["playerStatisticsGrid-page"] = str(page_num)

    return f"{base_url}&{urlencode(params)}"


def get_team_options(page) -> list[tuple[str, str]]:
    """
    Extract all team options from the team filter.

    :param page: Playwright page instance.
    :return: List of tuples in format (team_value, team_name).
    """
    options = page.locator("#frm-playerStatisticsFilter-team option")
    out = []

    for i in range(options.count()):
        option = options.nth(i)
        value = clean_text(option.get_attribute("value"))
        text = clean_text(option.inner_text())

        if not value or not text or text == "Všechny týmy":
            continue

        out.append((value, text))

    return out


def get_total_pages(page) -> int:
    """
    Detect the total number of pagination pages.

    :param page: Playwright page instance.
    :return: Total page count.
    """
    footer = page.locator("form#frm-playerStatisticsGrid-paginationForm h5")
    if footer.count() == 0:
        return 1

    text = clean_text(footer.first.inner_text())
    match = re.search(r"z\s+(\d+)", text)
    if not match:
        return 1

    return int(match.group(1))


def get_player_table(page):
    """
    Get the player statistics table element.

    :param page: Playwright page instance.
    :return: First matching table locator.
    :raises RuntimeError: If the statistics table is missing.
    """
    locator = page.locator("#snippet-playerStatisticsGrid-playerStatisticsGrid table")
    if locator.count() == 0:
        raise RuntimeError("No player statistics table found.")
    return locator.first


def extract_headers(table) -> list[str]:
    """
    Extract table header names.

    :param table: Table locator.
    :return: List of header labels.
    """
    headers = []
    ths = table.locator("thead tr th")

    for i in range(ths.count()):
        headers.append(clean_text(ths.nth(i).inner_text()))

    return headers


def parse_player_row(tds, season: str, team_name: str) -> dict | None:
    """
    Parse one player row from the statistics table.

    :param tds: Locator containing table cells.
    :param season: Season label.
    :param team_name: Team name.
    :return: Parsed player stats dictionary or None.
    """
    values = [clean_text(tds.nth(i).inner_text()) for i in range(tds.count())]

    if len(values) < 29:
        return None

    player_name = values[1]
    if not player_name:
        return None

    gp = parse_numeric(values[4])
    goals = parse_numeric(values[5])
    assists = parse_numeric(values[8])
    points = parse_numeric(values[9])
    plus_minus = parse_numeric(values[18])
    shots = parse_numeric(values[23])

    return {
        "source": "extraliga",
        "league": "czech",
        "season": season,
        "team_name": team_name,
        "player_name": player_name,
        "gp": gp,
        "goals": goals,
        "assists": assists,
        "points": points,
        "shots": shots,
        "pim": None,
        "plus_minus": plus_minus,
    }


def scrape_one_team(page, base_url: str, season: str, part_value: str, team_value: str, team_name: str) -> list[dict]:
    """
    Scrape all player statistics pages for one team.

    :param page: Playwright page instance.
    :param base_url: Base competition statistics URL.
    :param season: Season label.
    :param part_value: Division or competition part value.
    :param team_value: Team filter value.
    :param team_name: Team name.
    :return: List of parsed player rows.
    """
    rows = []
    seen = set()

    first_url = build_filtered_url(
        base_url=base_url,
        part_value=part_value,
        team_value=team_value,
        page_num=1,
    )

    page.goto(first_url, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(1800)
    try_accept_cookies(page)
    wait_for_player_statistics_ready(page)

    total_pages = get_total_pages(page)
    print(f"\n=== TEAM: {team_name} | SEASON: {season} | pages: {total_pages} ===")
    print(f"URL page 1: {first_url}")

    for page_num in range(1, total_pages + 1):
        page_url = build_filtered_url(
            base_url=base_url,
            part_value=part_value,
            team_value=team_value,
            page_num=page_num,
        )

        page.goto(page_url, wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(1500)

        table = get_player_table(page)

        if page_num == 1:
            headers = extract_headers(table)
            print("Detected headers:", headers)

        trs = table.locator("tbody tr")
        row_count = trs.count()
        print(f"{team_name} | {season} | page {page_num} | rows: {row_count}")

        before = len(rows)

        if page_num == 1:
            sample_players = []
            for i in range(min(5, row_count)):
                tr = trs.nth(i)
                tds = tr.locator("td")
                vals = [clean_text(tds.nth(j).inner_text()) for j in range(tds.count())]
                if len(vals) > 1:
                    sample_players.append(vals[1])
            print(f"{team_name} | sample players: {sample_players}")

        for i in range(row_count):
            tr = trs.nth(i)
            tds = tr.locator("td")
            parsed = parse_player_row(tds, season, team_name)

            if not parsed:
                continue

            key = (
                parsed["season"],
                parsed["team_name"],
                parsed["player_name"],
                parsed["gp"],
                parsed["goals"],
                parsed["assists"],
                parsed["points"],
            )
            if key in seen:
                continue

            seen.add(key)
            rows.append(parsed)

        added = len(rows) - before
        print(f"{team_name} | {season} | added rows: {added}")

        if row_count == 0:
            break

    return rows


def scrape_one_season(page, season: str, base_url: str, part_value: str) -> pd.DataFrame:
    """
    Scrape player statistics for one season.

    :param page: Playwright page instance.
    :param season: Season label.
    :param base_url: Base competition statistics URL.
    :param part_value: Division or competition part value.
    :return: DataFrame with season player statistics.
    """
    all_rows = []

    initial_url = build_filtered_url(
        base_url=base_url,
        part_value=part_value,
        team_value=None,
        page_num=1,
    )

    page.goto(initial_url, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(2500)
    try_accept_cookies(page)
    wait_for_player_statistics_ready(page)

    team_options = get_team_options(page)
    print(f"\nSeason {season} team options found:", team_options)

    for team_value, team_name in team_options:
        team_rows = scrape_one_team(
            page=page,
            base_url=base_url,
            season=season,
            part_value=part_value,
            team_value=team_value,
            team_name=team_name,
        )
        all_rows.extend(team_rows)

    return pd.DataFrame(all_rows).drop_duplicates().reset_index(drop=True)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1800, "height": 1300})

        for season, cfg in SEASON_CONFIG.items():
            print(f"SEASON: {season}")
            print(f"URL: {cfg['base_url']}")
            print(f"PART: {cfg['part_value']}")

            df = scrape_one_season(page=page, season=season, base_url=cfg["base_url"], part_value=cfg["part_value"])

            if df.empty:
                print(f"[WARN] No data for season {season}")
                continue

            season_slug = season.replace("/", "_")
            output_path = OUTPUT_DIR / f"extraliga_players_{season_slug}.csv"
            df.to_csv(output_path, index=False)

            print(f"Saved: {output_path}")
            print(f"Rows: {len(df)}")
            print(df.head(5).to_string())

        browser.close()


if __name__ == "__main__":
    main()
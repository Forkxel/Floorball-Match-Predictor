from pathlib import Path
import re
import pandas as pd
from playwright.sync_api import sync_playwright


BASE_DIR = Path(__file__).resolve().parent.parent.parent
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
        "part_value": "division-4603",
    },
}

TYPE_VALUE = "stats_all"


def clean_text(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def parse_numeric(value: str):
    value = clean_text(value).replace(",", ".")
    return pd.to_numeric(value, errors="coerce")


def try_accept_cookies(page) -> None:
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
    page.wait_for_selector("#frm-playerStatisticsFilter-team", state="attached", timeout=15000)
    page.wait_for_selector("#frm-playerStatisticsFilter-part", state="attached", timeout=15000)
    page.wait_for_selector("#frm-playerStatisticsFilter-type", state="attached", timeout=15000)
    page.wait_for_selector("#snippet-playerStatisticsGrid-playerStatisticsGrid table", state="attached", timeout=15000)
    page.wait_for_timeout(1000)


def set_hidden_select_value(page, selector: str, value: str) -> None:
    page.eval_on_selector(
        selector,
        """(el, value) => {
            el.value = value;
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }""",
        value,
    )
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1200)


def set_player_filters(page, part_value: str, team_value: str | None = None) -> None:
    set_hidden_select_value(page, "#frm-playerStatisticsFilter-part", part_value)
    set_hidden_select_value(page, "#frm-playerStatisticsFilter-type", TYPE_VALUE)

    if team_value is not None:
        set_hidden_select_value(page, "#frm-playerStatisticsFilter-team", team_value)


def set_page_size(page, page_size: str = "100") -> None:
    candidates = [
        "#frm-playerStatisticsGrid-paginationForm-itemsPerPage",
        "select[name='itemsPerPage']",
    ]

    for selector in candidates:
        try:
            if page.locator(selector).count() > 0:
                page.eval_on_selector(
                    selector,
                    """(el, value) => {
                        el.value = value;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }""",
                    page_size,
                )
                page.wait_for_load_state("networkidle")
                page.wait_for_timeout(1200)
                return
        except Exception:
            pass


def get_team_options(page) -> list[tuple[str, str]]:
    options = page.locator("#frm-playerStatisticsFilter-team option")
    out = []

    for i in range(options.count()):
        opt = options.nth(i)
        value = clean_text(opt.get_attribute("value"))
        text = clean_text(opt.inner_text())

        if not value or not text or text == "Všechny týmy":
            continue

        out.append((value, text))

    return out


def get_total_pages(page) -> int:
    footer = page.locator("form#frm-playerStatisticsGrid-paginationForm h5")
    if footer.count() == 0:
        return 1

    text = clean_text(footer.first.inner_text())
    m = re.search(r"z\s+(\d+)", text)
    if not m:
        return 1
    return int(m.group(1))


def set_page_number(page, page_num: int) -> None:
    inp = page.locator("#frm-playerStatisticsGrid-paginationForm-page")
    if inp.count() == 0:
        return

    page.eval_on_selector(
        "#frm-playerStatisticsGrid-paginationForm-page",
        """(el, value) => {
            el.value = String(value);
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        }""",
        page_num,
    )
    inp.first.press("Enter")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(1200)


def get_player_table(page):
    locator = page.locator("#snippet-playerStatisticsGrid-playerStatisticsGrid table")
    if locator.count() == 0:
        raise RuntimeError("No player statistics table.")
    return locator.first


def extract_headers(table) -> list[str]:
    headers = []
    ths = table.locator("thead tr th")
    for i in range(ths.count()):
        headers.append(clean_text(ths.nth(i).inner_text()))
    return headers


def parse_player_row(tds, season: str, team_name: str) -> dict | None:
    values = [clean_text(tds.nth(j).inner_text()) for j in range(tds.count())]

    if len(values) < 29:
        return None

    player_name = values[1] if len(values) > 1 else ""
    if not player_name:
        return None

    gp = parse_numeric(values[4])
    goals = parse_numeric(values[5])
    assists = parse_numeric(values[8])
    points = parse_numeric(values[9])
    plus_minus = parse_numeric(values[18])
    shots = parse_numeric(values[23])

    return {
        "source": "ceskyflorbal",
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
    rows = []
    seen = set()

    page.goto(base_url, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(2000)
    try_accept_cookies(page)
    wait_for_player_statistics_ready(page)

    set_player_filters(page, part_value=part_value, team_value=team_value)
    set_page_size(page, "100")

    total_pages = get_total_pages(page)
    print(f"\n=== TEAM: {team_name} | SEASON: {season} | pages: {total_pages} ===")

    for page_num in range(1, total_pages + 1):
        if page_num > 1:
            set_page_number(page, page_num)

        table = get_player_table(page)

        if page_num == 1:
            headers = extract_headers(table)
            print("Detected headers:", headers)

        trs = table.locator("tbody tr")
        row_count = trs.count()
        print(f"{team_name} | {season} | page {page_num} | rows: {row_count}")

        before = len(rows)

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
    all_rows = []

    page.goto(base_url, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(2500)
    try_accept_cookies(page)
    wait_for_player_statistics_ready(page)

    set_player_filters(page, part_value=part_value)

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

    df = pd.DataFrame(all_rows).drop_duplicates().reset_index(drop=True)
    return df


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1800, "height": 1300})

        for season, cfg in SEASON_CONFIG.items():
            print(f"\n==============================")
            print(f"SEASON: {season}")
            print(f"URL: {cfg['base_url']}")
            print(f"PART: {cfg['part_value']}")
            print(f"==============================")

            df = scrape_one_season(
                page=page,
                season=season,
                base_url=cfg["base_url"],
                part_value=cfg["part_value"],
            )

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
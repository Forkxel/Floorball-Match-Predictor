from typing import Any, Dict, List, Optional


def collect_nodes_with_standings(obj: Any, nodes: List[Dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        if "standings" in obj and isinstance(obj["standings"], list):
            nodes.append(obj)
        for value in obj.values():
            collect_nodes_with_standings(value, nodes)
    elif isinstance(obj, list):
        for item in obj:
            collect_nodes_with_standings(item, nodes)


def get_phase_text(node: Dict[str, Any]) -> str:
    phase_candidates = [
        node.get("phase"),
        node.get("type"),
        node.get("name"),
        node.get("description"),
    ]

    parent_stage = node.get("stage", {})
    if isinstance(parent_stage, dict):
        phase_candidates.extend([
            parent_stage.get("phase"),
            parent_stage.get("type"),
            parent_stage.get("name"),
            parent_stage.get("description"),
        ])

    return " ".join(str(x).lower() for x in phase_candidates if x)


def is_regular_phase(phase_text: str) -> bool:
    return "regular" in phase_text or "reg" in phase_text


def extract_regular_standings_records(
    standings_data: Dict[str, Any],
    competition_id: str,
    season_id: str,
    season_name: Optional[str],
    normalized_season: Optional[str],
) -> List[Dict[str, Any]]:

    records: List[Dict[str, Any]] = []
    nodes: List[Dict[str, Any]] = []

    collect_nodes_with_standings(standings_data, nodes)

    for node in nodes:
        phase_text = get_phase_text(node)
        if not is_regular_phase(phase_text):
            continue

        for standing_row in node.get("standings", []):
            competitor = standing_row.get("competitor", {})
            competitor_id = competitor.get("id")
            competitor_name = competitor.get("name")

            if not competitor_id:
                continue

            records.append({
                "competition_id": competition_id,
                "season_id": season_id,
                "season_name": season_name,
                "season": normalized_season,
                "team_id": competitor_id,
                "team_name": competitor_name,
                "official_rank": standing_row.get("rank"),
                "official_points": standing_row.get("points"),
                "official_played": standing_row.get("played"),
                "official_wins": standing_row.get("wins"),
                "official_losses": standing_row.get("losses"),
                "source_phase_text": phase_text,
            })

    return records
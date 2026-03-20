from pathlib import Path
import uuid

from src.simulation.audit import PredictionAuditor
from src.utils.database import DatabaseManager


def test_grade_pick_against_line_excludes_no_bet_rows():
    outcome, bet_won, direction_correct = PredictionAuditor._grade_pick_against_line(
        pick_direction="NO_BET",
        sportsbook_line=24.5,
        target_actual=27,
    )

    assert outcome is None
    assert bet_won is None
    assert direction_correct is None


def test_grade_pick_against_line_scores_over_pick_from_reported_line():
    outcome, bet_won, direction_correct = PredictionAuditor._grade_pick_against_line(
        pick_direction="OVER",
        sportsbook_line=24.5,
        target_actual=26,
    )

    assert outcome == "OVER"
    assert bet_won == 1
    assert direction_correct == 1


def test_grade_pick_against_line_marks_push_without_right_wrong_credit():
    outcome, bet_won, direction_correct = PredictionAuditor._grade_pick_against_line(
        pick_direction="UNDER",
        sportsbook_line=24.0,
        target_actual=24.0,
    )

    assert outcome == "PUSH"
    assert bet_won is None
    assert direction_correct is None


def test_get_injuries_for_date_returns_ranked_snapshots():
    db_path = Path(".") / f"test_injuries_{uuid.uuid4().hex}.db"
    db = DatabaseManager(db_path)
    report_date = "2026-03-19"

    try:
        db.insert_injury_snapshot({
            "player_id": 7,
            "player_name": "Test Player",
            "team_abbreviation": "LAL",
            "status": "QUESTIONABLE",
            "reason": "Hip",
            "source_name": "ROTOWIRE",
            "fetched_at": "2026-03-19T09:00:00",
            "report_date": report_date,
            "p_play": 0.5,
        })
        db.insert_injury_snapshot({
            "player_id": 7,
            "player_name": "Test Player",
            "team_abbreviation": "LAL",
            "status": "OUT",
            "reason": "Hip",
            "source_name": "ESPN",
            "fetched_at": "2026-03-19T10:00:00",
            "report_date": report_date,
            "p_play": 0.0,
        })

        injuries = db.get_injuries_for_date(report_date)

        assert len(injuries) == 1
        assert injuries[0]["player_name"] == "Test Player"
        assert injuries[0]["source_name"] == "ESPN"
        assert injuries[0]["status"] == "OUT"
    finally:
        if db_path.exists():
            db_path.unlink()

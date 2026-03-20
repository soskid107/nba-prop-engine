"""
Create a sanitized starter SQLite database for fresh clones.

The starter DB keeps the schema and core basketball history needed to run the
engine, while stripping local operating history, API usage, cached live data,
and private audit state.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DB = PROJECT_ROOT / "data" / "nba_props.db"
STARTER_DB = PROJECT_ROOT / "data" / "nba_props_starter.db"


TABLES_TO_CLEAR = [
    "api_usage",
    "bias_tracker",
    "calibration_buckets",
    "edge_performance",
    "edge_tracking",
    "http_cache",
    "injury_snapshots",
    "learning_insights",
    "model_performance",
    "odds_snapshots",
    "player_prop_odds",
    "prediction_log",
    "prediction_rejections",
    "predictions_archive",
    "predictions_archive_v1",
    "prior_updates",
    "regime_log",
]


def main() -> None:
    if not SOURCE_DB.exists():
        raise FileNotFoundError(f"Source DB not found: {SOURCE_DB}")

    STARTER_DB.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_DB, STARTER_DB)

    conn = sqlite3.connect(STARTER_DB)
    cur = conn.cursor()

    for table in TABLES_TO_CLEAR:
        cur.execute(f"DELETE FROM {table}")

    conn.commit()
    cur.execute("VACUUM")
    conn.close()

    size_mb = STARTER_DB.stat().st_size / (1024 * 1024)
    print(f"Created starter DB: {STARTER_DB} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

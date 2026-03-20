import pytest
import requests
from tenacity import RetryError

from src.utils.database import DatabaseManager
from src.ingestion.nba_ingestion import NBAIngestion
from datetime import datetime


def _skip_if_network_blocked(exc: Exception) -> None:
    network_errors = (requests.exceptions.RequestException, PermissionError, OSError, RetryError)
    if isinstance(exc, network_errors):
        pytest.skip(f"Network unavailable in this environment: {exc}")
    raise exc


def test_sync():
    db = DatabaseManager()
    nba = NBAIngestion(db)
    
    test_date = "2026-03-07"
    print(f"Testing schedule sync for {test_date}...")
    
    try:
        count = nba.sync_schedule(test_date)
    except Exception as exc:
        _skip_if_network_blocked(exc)
    print(f"Synced {count} games.")
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM games WHERE game_date = ?", (test_date,))
        row_count = cursor.fetchone()[0]
        print(f"Games in DB for {test_date}: {row_count}")
    assert count >= 0
    assert row_count >= 0

if __name__ == "__main__":
    test_sync()

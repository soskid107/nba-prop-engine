import pytest
import requests
from tenacity import RetryError

from src.ingestion.nba_ingestion import NBAIngestion
from src.utils.database import DatabaseManager


def _skip_if_network_blocked(exc: Exception) -> None:
    network_errors = (requests.exceptions.RequestException, PermissionError, OSError, RetryError)
    if isinstance(exc, network_errors):
        pytest.skip(f"Network unavailable in this environment: {exc}")
    raise exc


def test_bdl_roster():
    print("Testing balldontlie.io roster fallback...")
    db = DatabaseManager()
    nba = NBAIngestion(db)
    
    # Test a team (Lakers ID: 1610612747)
    # Case 1: Force fallback by using an invalid NBA team ID or just calling the logic
    # But first, let's just test if the http client works
    
    print("\n1. Testing HTTP Client for balldontlie...")
    try:
        res = nba.http.get_balldontlie_api("/teams")
    except Exception as exc:
        _skip_if_network_blocked(exc)
    if res and 'data' in res:
        print(f"   [OK] Fetched {len(res['data'])} teams from balldontlie")
    else:
        pytest.fail("Could not fetch teams from balldontlie")

    print("\n2. Testing get_team_roster fallback...")
    # We can't easily force nba_api to fail without mocking, but we can call the fallback logic directly 
    # if we modify the code slightly, or just trust the warn/info logs in a real failure.
    
    # Let's just try to get Lakers roster (should work via NBA API first)
    try:
        roster = nba.get_team_roster(1610612747)
    except Exception as exc:
        _skip_if_network_blocked(exc)
    print(f"   [OK] Fetched {len(roster)} players for Lakers")
    if roster:
        print(f"   Sample player: {roster[0]['full_name']}")

    print("\n3. Testing manual fallback path (Logic Check)...")
    team_abbr = "LAL"
    try:
        teams_data = nba.http.get_balldontlie_api("/teams")
    except Exception as exc:
        _skip_if_network_blocked(exc)
    bdl_team_id = None
    for t in teams_data['data']:
        if t['abbreviation'] == team_abbr:
            bdl_team_id = t['id']
            break
    
    if bdl_team_id:
        try:
            players_data = nba.http.get_balldontlie_api("/players", params={'team_ids[]': bdl_team_id})
        except Exception as exc:
            _skip_if_network_blocked(exc)
        if players_data and 'data' in players_data:
            print(f"   [OK] Manually fetched {len(players_data['data'])} players for {team_abbr} via balldontlie")
            print(f"   Sample: {players_data['data'][0]['first_name']} {players_data['data'][0]['last_name']}")
    else:
        pytest.fail("Could not find LAL in balldontlie teams")

if __name__ == "__main__":
    test_bdl_roster()

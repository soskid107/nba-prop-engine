
import sys
from src.utils.database import DatabaseManager
from src.ingestion.nba_ingestion import NBAIngestion

def fix_team(team_abbr: str):
    db = DatabaseManager()
    ingestion = NBAIngestion(db)
    
    print(f"Fixing logs for {team_abbr}...")
    
    # 1. Identify players on this team (from players table)
    # Note: players table has team_abbreviation (hopefully)
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT player_id, full_name FROM players WHERE team_abbreviation = ?", (team_abbr,))
        players = cursor.fetchall()
        
    if not players:
        print(f"No players found for {team_abbr} in 'players' table.")
        # Try finding by team_id? 
        # But 'players' table was populated by CommonAllPlayers which usually has team info.
        return

    player_ids = [p['player_id'] for p in players]
    print(f"Found {len(player_ids)} players for {team_abbr}")
    
    # 2. Delete existing logs for these players (to force refresh)
    # Only delete current season logs to be safe/fast
    with db.get_connection() as conn:
        cursor = conn.cursor()
        placeholders = ','.join(['?'] * len(player_ids))
        cursor.execute(f"DELETE FROM player_logs WHERE player_id IN ({placeholders}) AND season = '2025-26'", player_ids)
        print(f"Deleted existing 2025-26 logs for {len(player_ids)} players")
        
    # 3. Refetch
    ingestion.backfill_player_logs(player_ids, seasons=['2025-26'])
    print("Backfill complete.")

if __name__ == "__main__":
    fix_team('ATL')
    fix_team('LAL') # Fix LAL too for potential additional tests

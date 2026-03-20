
import sys
import time
from src.utils.database import DatabaseManager
from src.ingestion.nba_ingestion import NBAIngestion

def run_full_backfill():
    print("=== STARTING FULL LEAGUE BACKFILL ===")
    
    db = DatabaseManager()
    ingestion = NBAIngestion(db)
    
    # 1. Get all active players
    # We want players who have played in 2025-26
    # Or just get all players mapped to a team
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT player_id, full_name, team_abbreviation FROM players WHERE is_active = 1")
        players = cursor.fetchall()
        
    print(f"Found {len(players)} active players.")
    
    player_ids = [p['player_id'] for p in players]
    
    # 2. Define Target Seasons (Current Season Only)
    # Historical seasons (2021-2025) are verified healthy.
    seasons = ['2025-26']
    
    # Filter list
    players_to_process = player_ids
    print(f"Total active players to process: {len(players_to_process)}")

    # 4. Batch Process Remainder
    batch_size = 10 
    total_batches = (len(players_to_process) + batch_size - 1) // batch_size
    
    print(f"\n[INGESTION] Processing {len(players_to_process)} players in {total_batches} batches...")
    
    total_games_fetched = 0
    
    for i in range(0, len(players_to_process), batch_size):
        batch = players_to_process[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"\n--- Batch {batch_num}/{total_batches} ({len(batch)} players) ---")
        
        try:
            # Clean up these specific players before fetching (Upsert is cleaner, but Delete ensures no ghost rows)
            # We assume upsert in ingestion handles it, but let's be safe and clear old rows for these specific players/seasons
            with db.get_connection() as conn:
                 placeholders = ','.join(['?'] * len(batch))
                 season_placeholders = ','.join(['?'] * len(seasons))
                 # Construct valid SQL for IN clauses
                 # Delete relevant logs for this batch
                 conn.execute(
                     f"DELETE FROM player_logs WHERE player_id IN ({placeholders}) AND season IN ({season_placeholders})",
                     batch + seasons
                 )
            
            stats = ingestion.backfill_player_logs(batch, seasons=seasons)
            total_games_fetched += stats['games']
            
            # Forced sleep between batches to be nice to API
            time.sleep(3) 
            
        except Exception as e:
            print(f"Error in batch {batch_num}: {e}")
            # Continue to next batch instead of crashing
            time.sleep(5)
            
    print(f"\n=== BACKFILL COMPLETE ===")
    print(f"Total new games logged: {total_games_fetched}")

if __name__ == "__main__":
    run_full_backfill()

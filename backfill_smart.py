"""
Smart Backfill via nba_api
- Only targets players who have appeared in recent predictions/odds
- Uses short timeouts (15s) with fast retries
- Processes in small batches with progress tracking
- Skips players who are already up-to-date
"""
import sys
import time
from datetime import datetime
from src.utils.database import DatabaseManager
from src.ingestion.nba_ingestion import NBAIngestion

TARGET_DATE = "2026-03-05"  # We want logs up to this date

def get_priority_players(db):
    """Get players who matter: those with recent odds/predictions."""
    player_ids = set()
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Players from recent predictions CSVs (they have odds lines)
        cursor.execute("""
            SELECT DISTINCT player_id FROM player_logs 
            WHERE season = '2025-26' AND player_id IS NOT NULL
        """)
        for row in cursor.fetchall():
            player_ids.add(row['player_id'])
        
        # 2. Active players with team assignments
        cursor.execute("""
            SELECT player_id FROM players 
            WHERE is_active = 1 AND team_abbreviation NOT IN ('RET', 'UNK', '')
        """)
        for row in cursor.fetchall():
            player_ids.add(row['player_id'])
    
    return list(player_ids)

def get_stale_players(db, player_ids, cutoff_date):
    """Filter to only players whose latest log is before cutoff."""
    stale = []
    with db.get_connection() as conn:
        cursor = conn.cursor()
        for pid in player_ids:
            cursor.execute(
                "SELECT MAX(game_date) as latest FROM player_logs WHERE player_id = ?", 
                (pid,)
            )
            row = cursor.fetchone()
            latest = row['latest'] if row and row['latest'] else '2000-01-01'
            if latest < cutoff_date:
                stale.append((pid, latest))
    
    # Sort by staleness (most stale first)
    stale.sort(key=lambda x: x[1])
    return stale

def run_smart_backfill():
    db = DatabaseManager()
    ingestion = NBAIngestion(db)
    
    print("=== SMART BACKFILL ===")
    print(f"Target: Bring all active player logs up to {TARGET_DATE}\n")
    
    # Step 1: Get priority players
    all_players = get_priority_players(db)
    print(f"[1/3] Found {len(all_players)} priority players")
    
    # Step 2: Filter to stale ones only
    stale_players = get_stale_players(db, all_players, TARGET_DATE)
    print(f"[2/3] {len(stale_players)} players need updating")
    
    if not stale_players:
        print("All players are up to date!")
        return
    
    # Show a sample
    print(f"  Most stale: player_id={stale_players[0][0]}, last_log={stale_players[0][1]}")
    print(f"  Least stale: player_id={stale_players[-1][0]}, last_log={stale_players[-1][1]}")
    
    # Step 3: Process in batches
    batch_size = 5
    total_new_games = 0
    failed = []
    
    print(f"\n[3/3] Processing {len(stale_players)} players in batches of {batch_size}...")
    
    for i in range(0, len(stale_players), batch_size):
        batch = stale_players[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(stale_players) + batch_size - 1) // batch_size
        
        print(f"\n--- Batch {batch_num}/{total_batches} ---")
        
        for pid, last_date in batch:
            try:
                games = ingestion.fetch_player_game_logs(pid, '2025-26', incremental=True)
                total_new_games += games
                if games > 0:
                    print(f"  ✓ Player {pid}: +{games} new games")
            except Exception as e:
                print(f"  ✗ Player {pid}: {e}")
                failed.append(pid)
        
        # Brief pause between batches
        if i + batch_size < len(stale_players):
            time.sleep(2)
    
    print(f"\n=== BACKFILL COMPLETE ===")
    print(f"New games added: {total_new_games}")
    print(f"Failed players: {len(failed)}")
    
    # Verify
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(game_date) FROM player_logs")
        print(f"Latest game date in DB: {cursor.fetchone()[0]}")

if __name__ == "__main__":
    run_smart_backfill()

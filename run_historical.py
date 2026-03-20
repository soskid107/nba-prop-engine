"""
Historical Data Backfill Script
------------------------------
Fetches NBA game logs from 2000-01 to Present.
This allows for training deep regression models on 20+ years of data.

Usage:
    python run_historical.py [--start_year 2000] [--end_year 2025]
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Set output encoding
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.utils.database import DatabaseManager
from src.ingestion.nba_ingestion import NBAIngestion

def generate_season_strings(start_year: int, end_year: int) -> list:
    """Generate season strings like ['2000-01', '2001-02', ...]"""
    seasons = []
    for year in range(start_year, end_year + 1):
        next_year = (year + 1) % 100
        season_str = f"{year}-{next_year:02d}"
        seasons.append(season_str)
    return seasons

def run_backfill(start_year=2000, end_year=2025):
    print(f"============================================================")
    print(f" [NBA] HISTORICAL BACKFILL ENGINE (2000-2025)")
    print(f" Target Range: {start_year} to {end_year}")
    print(f"============================================================")
    
    db = DatabaseManager()
    ingestion = NBAIngestion(db)
    
    # 1. Load Master Player Index
    # We need to know who existed in 2000 to fetch their logs
    print("\n[1/3] Updating Master Player Index...")
    ingestion.load_historical_players_index()
    
    # 2. Identify Players for each Season
    # We query the DB players table where overlap exists
    # We stored FROM_YEAR in 'first_name' and TO_YEAR in 'last_name' (hacky but works)
    
    seasons = generate_season_strings(start_year, end_year)
    
    print(f"\n[2/3] Starting Season-by-Season Ingestion ({len(seasons)} seasons)...")
    
    total_games_fetched = 0
    
    for season in seasons:
        season_start_year = int(season.split('-')[0])
        print(f"\n>>> PROCESSING SEASON {season} <<<")
        
        # Find eligible players
        # Players whose career overlaps with this season
        # FROM_YEAR <= season_start_year AND TO_YEAR >= season_start_year
        with db.get_connection() as conn:
            cursor = conn.cursor()
            # Note: We cast the hacky columns back to int for comparison
            cursor.execute("""
                SELECT player_id, full_name, first_name, last_name 
                FROM players 
                WHERE CAST(first_name AS INTEGER) <= ? 
                AND CAST(last_name AS INTEGER) >= ?
            """, (season_start_year, season_start_year))
            
            eligible_players = cursor.fetchall()
            
        print(f"    Found {len(eligible_players)} eligible players for {season}")
        
        # Batch Fetch
        # We process in chunks to show progress
        batch_size = 10
        count = 0
        
        for i in range(0, len(eligible_players), batch_size):
            batch = eligible_players[i:i+batch_size]
            for p in batch:
                pid = p['player_id']
                # Fetch logs
                # We disable incremental check here to enforce checking for this specific season
                # (Output shows it fetches 0 if existing, but API call is made)
                # Actually, fetch_player_game_logs has an incremental flag. 
                # If we want to force fill specific season, we should rely on it returning 0 if DB has them.
                # But to save API calls, we could check DB first.
                
                # Check directly if we have logs for this player + season
                with db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) as cnt FROM player_logs WHERE player_id = ? AND season = ?", (pid, season))
                    if cursor.fetchone()['cnt'] > 10:
                        # Assume data exists (min 10 games)
                        sys.stdout.write(".")
                        sys.stdout.flush()
                        continue

                # If missing, fetch
                added = ingestion.fetch_player_game_logs(pid, season, incremental=False)
                if added > 0:
                    sys.stdout.write("+")
                    total_games_fetched += added
                else:
                    sys.stdout.write("x")
                sys.stdout.flush()
                
            count += len(batch)
            print(f" {count}/{len(eligible_players)}")
            
            # Nap to be nice to API
            # fetch_player_game_logs already handles rate limit, but we can add extra
            # time.sleep(0.5) 
            
    print(f"\n\n[COMPLETE] Backfill finished.")
    print(f"Total new game logs ingested: {total_games_fetched}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2000)
    parser.add_argument("--end_year", type=int, default=2025)
    args = parser.parse_args()
    
    run_backfill(args.start_year, args.end_year)

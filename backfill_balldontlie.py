import sys
import time
from datetime import datetime, timedelta
from src.utils.database import DatabaseManager
from src.utils.http_client import SmartHttpClient

def run_date_based_backfill(start_date_str: str, end_date_str: str):
    print(f"=== STARTING DATE-BASED BACKFILL ({start_date_str} to {end_date_str}) ===")
    
    db = DatabaseManager()
    http = SmartHttpClient(db)
    
    # Pre-load NBA API player map (name -> nba_player_id) to match balldontlie stats
    print("Loading player map from DB...")
    player_map = {}
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT player_id, full_name, team_abbreviation FROM players WHERE is_active = 1")
        for row in cursor.fetchall():
            # Standardize names for matching (lower, no punctuation)
            clean_name = row['full_name'].lower().replace("'", "").replace(".", "").replace("-", " ")
            player_map[clean_name] = {
                'id': row['player_id'],
                'team': row['team_abbreviation']
            }
    
    print(f"Loaded {len(player_map)} active players into map.")
    
    # Parse dates
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    current_dt = start_dt
    total_games_inserted = 0
    
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        print(f"\n--- Fetching stats for {date_str} ---")
        
        # balldontlie allows pagination, usually 1 day of NBA games is < 100 players playing? 
        # Wait, up to 15 games * 20 players = 300 stats. So we need pagination.
        
        page = 1
        has_more = True
        inserted_today = 0
        
        while has_more:
            res = http.get_balldontlie_api("/stats", params={
                'dates[]': date_str,
                'per_page': 100,
                'cursor': page if page > 1 else None # Actually balldontlie v1 uses cursor pagination now
            })
            
            if not res or 'data' not in res:
                print(f"  [ERROR] Failed to fetch data for {date_str}")
                break
                
            data = res['data']
            if not data:
                break
                
            for stat in data:
                # Only insert if player played (min > 0)
                min_str = stat.get('min')
                if not min_str or min_str == '0' or min_str == '00':
                    continue
                    
                p_info = stat.get('player', {})
                fname = p_info.get('first_name', '')
                lname = p_info.get('last_name', '')
                full_name = f"{fname} {lname}".lower().replace("'", "").replace(".", "").replace("-", " ")
                
                # Match player to our DB
                matched_nba_id = None
                if full_name in player_map:
                    matched_nba_id = player_map[full_name]['id']
                else:
                    # Try partial match or just skip (many end of bench guys we don't care about)
                    continue
                
                # Determine matchup (we need opponent abbreviation)
                game_info = stat.get('game', {})
                team_info = stat.get('team', {})
                
                home_team_id = game_info.get('home_team_id')
                home_team_score = game_info.get('home_team_score')
                visitor_team_score = game_info.get('visitor_team_score')
                
                is_home = (team_info.get('id') == home_team_id)
                
                # Unfortunately balldontlie v1 doesn't easily give opponent abbreviation in the stat object itself cleanly 
                # without extra /games fetches, but let's just make it 'UNK' or fetch teams
                # We can map balldontlie team ID to abbreviation
                # We can do this next, but for now we skip opp_abbr
                
                # To be precise, let's just assume we can process it:
                try:
                    minutes = 0.0
                    if ':' in str(min_str):
                        parts = str(min_str).split(':')
                        minutes = float(parts[0]) + float(parts[1])/60
                    else:
                        minutes = float(min_str)
                    
                    pts = stat.get('pts', 0)
                    ppm = pts / minutes if minutes > 0 else 0
                    
                    log_data = {
                        'player_id': matched_nba_id,
                        'game_id': str(game_info.get('id', '')),
                        'game_date': date_str,
                        'season': '2025-26', # hardcoded for this backfill
                        'team_abbreviation': player_map[full_name]['team'],
                        'opponent_abbreviation': 'UNK', # Can be updated later or ignored by model
                        'is_home': int(is_home),
                        'is_starter': 0,
                        'minutes': minutes,
                        'points': pts,
                        'rebounds': stat.get('reb', 0),
                        'assists': stat.get('ast', 0),
                        'steals': stat.get('stl', 0),
                        'blocks': stat.get('blk', 0),
                        'turnovers': stat.get('turnover', 0),
                        'fgm': stat.get('fgm', 0),
                        'fga': stat.get('fga', 0),
                        'fg3m': stat.get('fg3m', 0),
                        'fg3a': stat.get('fg3a', 0),
                        'ftm': stat.get('ftm', 0),
                        'fta': stat.get('fta', 0),
                        'plus_minus': 0, # balldontlie doesn't always have this
                        'ppm': ppm
                    }
                    db.insert_player_log(log_data)
                    inserted_today += 1
                except Exception as e:
                    print(f"Error parsing stat for {full_name}: {e}")
            
            # Check for next page
            meta = res.get('meta', {})
            next_cursor = meta.get('next_cursor')
            if next_cursor:
                page = next_cursor
            else:
                has_more = False
                
        print(f"  [OK] Saved {inserted_today} player logs for {date_str}")
        total_games_inserted += inserted_today
        current_dt += timedelta(days=1)
        
    print(f"\n=== BACKFILL COMPLETE ===")
    print(f"Total new games logged via balldontlie: {total_games_inserted}")

if __name__ == "__main__":
    run_date_based_backfill("2026-02-12", "2026-03-05")


import pandas as pd
from datetime import datetime
from nba_api.stats.endpoints import scoreboardv2, boxscoretraditionalv3
from src.utils.database import DatabaseManager
from src.ingestion.nba_ingestion import NBAIngestion

def fetch_live_boxscores(game_date):
    print(f"\n[Live] Fetching boxscores for {game_date}...")
    
    # 1. Get Games
    try:
        board = scoreboardv2.ScoreboardV2(game_date=game_date)
        games = board.get_data_frames()[0]
    except Exception as e:
        print(f"Error fetching scoreboard: {e}")
        return

    if games.empty:
        print("No games found.")
        return

    print(f"Found {len(games)} games.")
    
    db = DatabaseManager()
    
    # 2. Get Boxscores for each game
    processed_count = 0
    for _, game in games.iterrows():
        game_id = game['GAME_ID']
        matchup = f"{game['HOME_TEAM_ID']} vs {game['VISITOR_TEAM_ID']}"
        status = game['GAME_STATUS_TEXT']
        print(f"  - Fetching {game_id} ({status})...")
        
        try:
            box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
            frames = box.get_data_frames()
            if not frames:
                 print(f"    -> No data frames returned (Game likely not started).")
                 continue
            player_stats = frames[0]
            
            if player_stats.empty:
                print(f"    -> No stats available yet.")
                continue
                
            # 3. Save to DB (mimicking the player_log structure)
            with db.get_connection() as conn:
                for _, row in player_stats.iterrows():
                    # Map V3 columns (often personId instead of PLAYER_ID)
                    pid = row.get('personId')
                    pts = row.get('points', 0)
                    min_str = row.get('minutes', '')
                    
                    if not pid or min_str == '': # DNP check
                        continue
                        
                    # Parse minutes
                    minutes = 0.0
                    if min_str:
                        try:
                            # Usually format is "PT32M45.00S" or similar in V3, or just MM:SS
                            # Let's handle standard string first
                            s = str(min_str)
                            if 'PT' in s: # ISO format sometimes
                                pass # complex parsing, but usually it's simple string in DF
                            
                            if ':' in s:
                                m, sec = s.split(':')
                                minutes = float(m) + float(sec)/60
                            else:
                                minutes = float(s)
                        except:
                            pass
                            
                    # Upsert into player_logs
                    conn.execute("""
                        INSERT OR REPLACE INTO player_logs (
                            player_id, game_id, game_date, season, 
                            team_id, team_abbreviation, 
                            minutes, points, rebounds, assists, 
                            steals, blocks, turnovers, 
                            fgm, fga, fg3m, fg3a, ftm, fta, 
                            plus_minus, ppm, is_home
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    """, (
                        pid, game_id, game_date, '2025-26',
                        row.get('teamId'), row.get('teamTricode'),
                        minutes, pts, row.get('reboundsTotal', 0), row.get('assists', 0),
                        row.get('steals', 0), row.get('blocks', 0), row.get('turnovers', 0),
                        row.get('fieldGoalsMade', 0), row.get('fieldGoalsAttempted', 0), 
                        row.get('threePointersMade', 0), row.get('threePointersAttempted', 0), 
                        row.get('freeThrowsMade', 0), row.get('freeThrowsAttempted', 0),
                        row.get('plusMinusPoints', 0), (pts/minutes if minutes > 0 else 0),
                        1 if row.get('teamId') == game['HOME_TEAM_ID'] else 0
                    ))
            
            processed_count += len(player_stats)
            print(f"    -> Updated {len(player_stats)} players.")
            
        except Exception as e:
            print(f"    -> Error fetching boxscore: {e}")

    print(f"\n[Done] Updated {processed_count} live player records.")

if __name__ == "__main__":
    fetch_live_boxscores('2026-01-25')

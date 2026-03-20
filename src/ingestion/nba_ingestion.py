"""
NBA Data Ingestion

Fetches player game logs, team data, and schedule from the NBA API.
Uses incremental loading to only fetch new data.
"""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from nba_api.stats.endpoints import (
    playergamelog,
    commonplayerinfo,
    commonteamroster,
    leaguegamefinder,
    scoreboardv2,
    commonallplayers
)
from nba_api.stats.static import players, teams

from ..utils.config import get_config
from ..utils.database import DatabaseManager
from ..utils.http_client import SmartHttpClient


class NBAIngestion:
    """Handles all NBA data ingestion with rate limiting."""
    
    # Team abbreviation mappings (NBA API to standard)
    TEAM_ABBR_MAP = {
        'PHX': 'PHO',  # Phoenix sometimes uses PHX
        'BKN': 'BRK',  # Brooklyn
        'CHA': 'CHO',  # Charlotte (older)
        'NOP': 'NOP',  # New Orleans
    }
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """Initialize NBA ingestion.
        
        Args:
            db: Optional database manager instance
        """
        self.config = get_config()
        self.db = db or DatabaseManager()
        self.http = SmartHttpClient(self.db)
        self._last_api_call = 0.0
    
    def _rate_limit(self) -> None:
        """Apply rate limiting between NBA API calls."""
        elapsed = time.time() - self._last_api_call
        min_delay = self.config.nba_api_delay + (self.config.nba_api_jitter * 0.5)
        
        if elapsed < min_delay:
            sleep_time = min_delay - elapsed
            print(f"  [Rate Limit] Sleeping {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        
        self._last_api_call = time.time()
    
    def _parse_minutes(self, min_str: str) -> float:
        """Parse minutes string (MM:SS) to float.
        
        Args:
            min_str: Minutes string like "32:45" or None
            
        Returns:
            Minutes as float (32.75 for 32:45)
        """
        if not min_str or min_str == '' or min_str is None:
            return 0.0
        
        try:
            if ':' in str(min_str):
                parts = str(min_str).split(':')
                return float(parts[0]) + float(parts[1]) / 60
            return float(min_str)
        except (ValueError, IndexError):
            return 0.0
    
    # =====================
    # Static Data Loading
    # =====================
    def load_all_teams(self) -> int:
        """Load all NBA teams into database.
        
        Returns:
            Number of teams loaded
        """
        print("\n[NBA] Loading all teams...")
        nba_teams = teams.get_teams()
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            for team in nba_teams:
                cursor.execute("""
                    INSERT OR REPLACE INTO teams 
                    (team_id, abbreviation, full_name, city, nickname)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    team['id'],
                    team['abbreviation'],
                    team['full_name'],
                    team['city'],
                    team['nickname']
                ))
        
        print(f"  [OK] Loaded {len(nba_teams)} teams")
        return len(nba_teams)
    
    def load_all_players(self) -> int:
        """Load all active NBA players into database with Team ID.
        
        Uses CommonAllPlayers endpoint to get live team affiliation.
        
        Returns:
            Number of players loaded
        """
        print("\n[NBA] Loading all active players (via API)...")
        self._rate_limit()
        
        try:
            # Fetch active players only
            resp = commonallplayers.CommonAllPlayers(is_only_current_season=1)
            df = resp.get_data_frames()[0]
        except Exception as e:
            print(f"  [WARN] Failed to fetch active players: {e}")
            return 0
        
        count = 0
        with self.db.get_connection() as conn:
            # We use a transaction to ensure integrity
            for _, row in df.iterrows():
                try:
                    # Parse name (Last, First)
                    full_name = row.get('DISPLAY_FIRST_LAST')
                    # If unavailable, try parsing DISPLAY_LAST_COMMA_FIRST
                    if not full_name:
                        parts = row.get('DISPLAY_LAST_COMMA_FIRST', '').split(', ')
                        if len(parts) == 2:
                            full_name = f"{parts[1]} {parts[0]}"
                        else:
                            full_name = row.get('DISPLAY_LAST_COMMA_FIRST')
                            
                    self.db.upsert_player({
                        'player_id': row.get('PERSON_ID'),
                        'full_name': full_name,
                        'first_name': row.get('ROSTERSTATUS'), # Not truly first name but we keep schema
                        'last_name': row.get('FROM_YEAR'), # Schema compat - Storing From Year temporarily
                        'team_id': row.get('TEAM_ID'),
                        'team_abbreviation': row.get('TEAM_ABBREVIATION'),
                        'is_active': 1
                    })
                    count += 1
                except Exception as ex:
                    print(f"  [WARN] Skipping player {row.get('PERSON_ID')}: {ex}")
        reconciled = self.db.reconcile_player_teams_from_logs()
        if reconciled > 0:
            print(f"  [OK] Reconciled {reconciled} player teams from recent logs")

        print(f"  [OK] Loaded {count} active players")
        return count

    def load_historical_players_index(self) -> int:
        """Load ALL NBA players (past and present) into database.
        
        Required for historical backfill (2000-2025).
        
        Returns:
            Number of players loaded
        """
        print("\n[NBA] Loading MASTER PLAYER INDEX (1946-Present)...")
        self._rate_limit()
        
        try:
            # Fetch ALL players
            resp = commonallplayers.CommonAllPlayers(is_only_current_season=0)
            df = resp.get_data_frames()[0]
        except Exception as e:
            print(f"  [WARN] Failed to fetch master player index: {e}")
            return 0
            
        count = 0
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                try:
                    pid = row.get('PERSON_ID')
                    full_name = row.get('DISPLAY_FIRST_LAST')
                    start_year = row.get('FROM_YEAR')
                    end_year = row.get('TO_YEAR')
                    
                    # Only care about "Modern Era" (roughly 2000+) for this request
                    # But we store provided range.
                    # We reuse upsert_player but need to be careful about 'is_active'
                    
                    # Check if player exists to avoid overwriting current 'is_active' status 
                    # with '0' if they are actually active but this endpoint says otherwise? 
                    # Actually CommonAllPlayers(0) has RosterStatus too.
                    
                    is_active = 1 if row.get('ROSTERSTATUS') == 1 else 0
                    
                    # Fix for current season: If active, force TO_YEAR to be at least 2025
                    try:
                         if is_active:
                             end_year = max(int(end_year), 2025)
                    except (ValueError, TypeError):
                        pass

                    self.db.upsert_player({
                        'player_id': pid,
                        'full_name': full_name,
                        'first_name': str(start_year), # Storing metadata in unused fields for now
                        'last_name': str(end_year),
                        'team_id': row.get('TEAM_ID'),
                        'team_abbreviation': row.get('TEAM_ABBREVIATION') or 'RET',
                        'is_active': is_active
                    })
                    count += 1
                except Exception:
                    continue
        
        print(f"  [OK] Loaded {count} historical players into index")
        return count
    
    # =====================
    # Game Log Ingestion
    # =====================
    def fetch_player_game_logs(self, player_id: int, season: str,
                               incremental: bool = True) -> int:
        """Fetch game logs for a player.
        
        Args:
            player_id: NBA player ID
            season: Season string (e.g., "2024-25")
            incremental: If True, only fetch games after last known game
            
        Returns:
            Number of new games loaded
        """
        # Check last known game if incremental
        last_date = None
        if incremental:
            last_date = self.db.get_latest_game_date(player_id)
        
        self._rate_limit()
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                log = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season,
                    season_type_all_star='Regular Season',
                    timeout=15  # Fast timeout for backfill throughput
                )
                df = log.get_data_frames()[0]
                break # Success
            except Exception as e:
                if attempt < max_retries - 1:
                    sleep_time = (attempt + 1) * 2 # 2s, 4s
                    print(f"  [Retry {attempt+1}] Connection issue ({e}). Sleeping {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"  [WARN] Failed to fetch logs for player {player_id} after retries: {e}")
                    return 0
        
        if df.empty:
            return 0
        
        new_games = 0
        for _, row in df.iterrows():
            game_date_str = row.get('GAME_DATE', '')
            try:
                # Parse "OCT 30, 2024" to ISO "2024-10-30"
                if ',' in str(game_date_str):
                     dt = datetime.strptime(str(game_date_str), '%b %d, %Y')
                     game_date = dt.strftime('%Y-%m-%d')
                else:
                     game_date = str(game_date_str)
            except ValueError:
                game_date = str(game_date_str)

            
            # Skip if we already have this game (incremental)
            if last_date and game_date <= last_date:
                continue
            
            # Parse matchup for opponent and team
            matchup = row.get('MATCHUP', '')
            is_home = '@' not in matchup
            
            parts = matchup.split(' ') if matchup else []
            team_abbr = parts[0] if len(parts) > 0 else None
            opp_abbr = parts[-1] if len(parts) > 0 else None
            
            # Matchup check
            if not team_abbr:
                team_abbr = row.get('TEAM_ABBREVIATION') # Fallback if present
            
            # Calculate PPM
            minutes = self._parse_minutes(row.get('MIN'))
            points = int(row.get('PTS', 0) or 0)
            ppm = points / minutes if minutes > 0 else 0.0
            
            log_data = {
                'player_id': player_id,
                'game_id': row.get('Game_ID', ''),
                'game_date': game_date,
                'season': season,
                'team_abbreviation': team_abbr,
                'opponent_abbreviation': opp_abbr,
                'is_home': int(is_home),
                'is_starter': 0,  # Will update later
                'minutes': minutes,
                'points': points,
                'rebounds': int(row.get('REB', 0) or 0),
                'assists': int(row.get('AST', 0) or 0),
                'steals': int(row.get('STL', 0) or 0),
                'blocks': int(row.get('BLK', 0) or 0),
                'turnovers': int(row.get('TOV', 0) or 0),
                'fgm': int(row.get('FGM', 0) or 0),
                'fga': int(row.get('FGA', 0) or 0),
                'fg3m': int(row.get('FG3M', 0) or 0),
                'fg3a': int(row.get('FG3A', 0) or 0),
                'ftm': int(row.get('FTM', 0) or 0),
                'fta': int(row.get('FTA', 0) or 0),
                'plus_minus': int(row.get('PLUS_MINUS', 0) or 0),
                'ppm': ppm
            }
            
            self.db.insert_player_log(log_data)
            new_games += 1
        
        return new_games
    
    def backfill_player_logs(self, player_ids: List[int], 
                             seasons: List[str] = None,
                             progress_callback=None) -> Dict[str, int]:
        """Backfill game logs for multiple players.
        
        Args:
            player_ids: List of player IDs to fetch
            seasons: List of seasons (default: current season only)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with counts: {'players': N, 'games': M}
        """
        if seasons is None:
            seasons = [self.config.current_season]
        
        total_games = 0
        players_processed = 0
        
        print(f"\n[NBA] Backfilling logs for {len(player_ids)} players...")
        
        for i, player_id in enumerate(player_ids):
            for season in seasons:
                games = self.fetch_player_game_logs(player_id, season, incremental=True)
                total_games += games
            
            players_processed += 1
            
            if progress_callback:
                progress_callback(i + 1, len(player_ids))
            elif (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(player_ids)} players")
        
        print(f"   Processed {players_processed} players, {total_games} new games")
        return {'players': players_processed, 'games': total_games}
    
    # =====================
    # Schedule / Today's Games
    # =====================
    def get_todays_games(self, game_date: str = None) -> List[Dict[str, Any]]:
        """Fetch today's NBA games and sync to database.
        
        Args:
            game_date: ISO date string (YYYY-MM-DD), defaults to today
            
        Returns:
            List of game dicts with home/away teams
        """
        game_date = game_date or datetime.now().strftime('%Y-%m-%d')
        print(f"\n[NBA] Fetching games for {game_date}...")
        
        # 1. Sync to database first
        self.sync_schedule(game_date)
        
        # 2. Return from API directly for immediate use
        self._rate_limit()
        try:
            # ScoreboardV2 uses local date
            scoreboard = scoreboardv2.ScoreboardV2(game_date=game_date)
            games_df = scoreboard.get_data_frames()[0]
        except Exception as e:
            print(f"  [WARN] Failed to fetch games from API: {e}")
            # Fallback to DB
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM games WHERE game_date = ?", (game_date,))
                return [dict(row) for row in cursor.fetchall()]
        
        games = []
        for _, row in games_df.iterrows():
            game = {
                'game_id': row.get('GAME_ID'),
                'game_date': game_date,
                'home_team_id': row.get('HOME_TEAM_ID'),
                'away_team_id': row.get('VISITOR_TEAM_ID'),
                'status': row.get('GAME_STATUS_TEXT', 'Scheduled')
            }
            games.append(game)
        
        print(f"  [OK] Found {len(games)} games for {game_date}")
        return games

    def sync_schedule(self, game_date: str) -> int:
        """Fetch games for a date and persist to the games table.
        
        Args:
            game_date: ISO date string (YYYY-MM-DD)
            
        Returns:
            Number of games synced
        """
        self._rate_limit()
        try:
            # Always use ScoreboardV2 for specific date sync
            scoreboard = scoreboardv2.ScoreboardV2(game_date=game_date)
            games_df = scoreboard.get_data_frames()[0]
            header_df = scoreboard.get_data_frames()[1] # Line Score usually has team abbrs
        except Exception as e:
            print(f"  [WARN] Scoreboard sync failed for {game_date}: {e}")
            return 0
            
        if games_df.empty:
            return 0
            
        # Build team mapping for abbreviations
        team_abbrs = {}
        if not header_df.empty:
            for _, row in header_df.iterrows():
                tid = row.get('TEAM_ID')
                abbr = row.get('TEAM_ABBREVIATION')
                if tid and abbr:
                    team_abbrs[tid] = abbr

        count = 0
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            for _, row in games_df.iterrows():
                gid = row.get('GAME_ID')
                h_id = row.get('HOME_TEAM_ID')
                a_id = row.get('VISITOR_TEAM_ID')
                
                # Insert/Update game record
                cursor.execute("""
                    INSERT OR REPLACE INTO games 
                    (game_id, game_date, season, home_team_id, away_team_id, 
                     home_team_abbr, away_team_abbr, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    gid,
                    game_date,
                    self.config.current_season,
                    h_id,
                    a_id,
                    team_abbrs.get(h_id),
                    team_abbrs.get(a_id),
                    row.get('GAME_STATUS_TEXT', 'Scheduled')
                ))
                count += 1
            conn.commit()
            
        print(f"  [OK] Synced {count} games into DB for {game_date}")
        return count
    
    def get_team_roster(self, team_id: int, season: str = None) -> List[Dict[str, Any]]:
        """Fetch team roster.
        
        Args:
            team_id: NBA team ID
            season: Season string (default: current)
            
        Returns:
            List of player dicts
        """
        season = season or self.config.current_season
        self._rate_limit()
        
        # 1. Try NBA API (nba_api library)
        try:
            roster = commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season
            )
            df = roster.get_data_frames()[0]
            
            players_list = []
            for _, row in df.iterrows():
                players_list.append({
                    'player_id': row.get('PLAYER_ID'),
                    'full_name': row.get('PLAYER'),
                    'position': row.get('POSITION'),
                    'team_id': team_id
                })
            return players_list

        except Exception as e:
            print(f"  [WARN] stats.nba.com roster fetch failed for team {team_id}: {e}")
            
        # 2. Fallback to balldontlie.io
        print(f"  [INFO] Attempting balldontlie.io fallback for team {team_id}...")
        try:
            # Need to map NBA Team ID to balldontlie Team ID if they differ? 
            # They usually match or we can fetch by abbreviation.
            # balldontlie /players endpoint allows filtering by team_ids[]
            
            # Map NBA team_id to abbreviation first?
            team_abbr = None
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT abbreviation FROM teams WHERE team_id = ?", (team_id,))
                row = cursor.fetchone()
                if row:
                    team_abbr = row['abbreviation']

            if team_abbr:
                # balldontlie v1 uses ?team_ids[]
                # First find the balldontlie team_id
                teams_data = self.http.get_balldontlie_api("/teams")
                bdl_team_id = None
                if teams_data and 'data' in teams_data:
                    for t in teams_data['data']:
                        if t['abbreviation'] == team_abbr or t['id'] == (team_id % 100): # heuristic
                            bdl_team_id = t['id']
                            break
                
                if bdl_team_id:
                    # Fetch players for this team
                    players_data = self.http.get_balldontlie_api("/players", params={'team_ids[]': bdl_team_id})
                    if players_data and 'data' in players_data:
                        players_list = []
                        for p in players_data['data']:
                            players_list.append({
                                'player_id': p.get('id'), # Note: IDs might differ! 
                                'full_name': f"{p.get('first_name')} {p.get('last_name')}",
                                'position': p.get('position'),
                                'team_id': team_id # Keep the original NBA team_id for consistency
                            })
                        print(f"  [OK] Successfully recovered {len(players_list)} players via balldontlie")
                        return players_list
        except Exception as ex:
            print(f"  [ERROR] balldontlie fallback failed: {ex}")

        return []
    
    def get_players_for_todays_games(self) -> List[int]:
        """Get all player IDs for teams playing today.
        
        Returns:
            List of player IDs
        """
        games = self.get_todays_games()
        
        team_ids = set()
        for game in games:
            team_ids.add(game['home_team_id'])
            team_ids.add(game['away_team_id'])
        
        player_ids = []
        for team_id in team_ids:
            roster = self.get_team_roster(team_id)
            for player in roster:
                if player['player_id']:
                    player_ids.append(player['player_id'])
        
        return player_ids


# Convenience function
def get_nba_ingestion() -> NBAIngestion:
    """Get NBA ingestion instance."""
    return NBAIngestion()

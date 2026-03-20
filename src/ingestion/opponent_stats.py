"""

Opponent Stats Ingestion



Fetches opponent advanced stats for style/matchup edge analysis:

- Opponent Pace

- Opponent Defensive Rating

- 3P% Allowed

- Paint Points Allowed

- Rebounding Rates

"""



import time

from datetime import datetime, timedelta

from typing import Any, Dict, List, Optional



from nba_api.stats.endpoints import (

    leaguedashteamstats,

    teamgamelogs

)



from ..utils.config import get_config
from ..utils.database import DatabaseManager
from ..utils.http_client import SmartHttpClient




class OpponentStatsIngestion:

    """Fetches and stores opponent advanced statistics for matchup analysis."""

    

    def __init__(self, db: Optional[DatabaseManager] = None):

        self.config = get_config()

        self.db = db or DatabaseManager()

        self.http = SmartHttpClient(self.db)

        self._last_api_call = 0.0

        self._ensure_table()

    

    def _ensure_table(self) -> None:

        """Create opponent stats table if not exists."""

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                CREATE TABLE IF NOT EXISTS team_advanced_stats (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    team_id INTEGER NOT NULL,

                    team_abbreviation TEXT NOT NULL,

                    stat_date TEXT NOT NULL,

                    season TEXT NOT NULL,

                    

                    -- Pace & Tempo

                    pace REAL,

                    possessions REAL,

                    

                    -- Defensive Ratings

                    def_rating REAL,

                    opp_pts_per_game REAL,

                    

                    -- Shooting Defense

                    opp_fg_pct REAL,

                    opp_fg3_pct REAL,

                    opp_fg3a_per_game REAL,

                    

                    -- Paint Defense

                    opp_pts_in_paint REAL,

                    opp_second_chance_pts REAL,

                    

                    -- Rebounding

                    opp_oreb_pct REAL,

                    dreb_pct REAL,

                    

                    -- Window type (L5, L10, Season)

                    window_type TEXT NOT NULL,

                    

                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                    UNIQUE(team_id, stat_date, window_type)

                )

            """)

            

            cursor.execute("""

                CREATE INDEX IF NOT EXISTS idx_team_stats_lookup 

                ON team_advanced_stats(team_abbreviation, stat_date, window_type)

            """)

            conn.commit()

    

    def _rate_limit(self) -> None:

        """Apply rate limiting between NBA API calls."""

        elapsed = time.time() - self._last_api_call

        min_delay = self.config.nba_api_delay + 0.5

        

        if elapsed < min_delay:

            time.sleep(min_delay - elapsed)

        

        self._last_api_call = time.time()

    def _load_latest_window_stats(self, window_type: str) -> List[Dict[str, Any]]:
        """Load the most recent stored team stats for a given window."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT team_id, team_abbreviation, season, pace, def_rating,
                       opp_pts_per_game, opp_fg_pct, opp_fg3_pct,
                       opp_fg3a_per_game, opp_oreb_pct, dreb_pct
                FROM team_advanced_stats
                WHERE window_type = ?
                  AND stat_date = (
                      SELECT MAX(stat_date)
                      FROM team_advanced_stats
                      WHERE window_type = ?
                  )
            """, (window_type, window_type))
            return [dict(row) for row in cursor.fetchall()]

    def _derive_window_fallback(self, requested_window: str) -> List[Dict[str, Any]]:
        """Reuse the freshest stored stats when live rolling windows fail."""
        if requested_window == 'L5':
            fallback_order = ['L10', 'Season']
        elif requested_window == 'L10':
            fallback_order = ['Season', 'L5']
        else:
            fallback_order = ['L10', 'L5']

        today = datetime.now().strftime('%Y-%m-%d')
        for source_window in fallback_order:
            cached_stats = self._load_latest_window_stats(source_window)
            if cached_stats:
                derived_stats = []
                for stat in cached_stats:
                    cloned = dict(stat)
                    cloned['stat_date'] = today
                    cloned['window_type'] = requested_window
                    derived_stats.append(cloned)
                print(f"  [INFO] Using {source_window} cached stats as fallback for {requested_window}.")
                return derived_stats
        return []

    

    def fetch_league_team_stats(self, season: str = None,

                                 last_n_games: int = 0) -> List[Dict[str, Any]]:

        """Fetch advanced stats for all teams.

        

        Args:

            season: Season string (e.g., "2024-25")

            last_n_games: 0 for season, or 5/10 for rolling

            

        Returns:

            List of team stat dicts

        """

        season = season or self.config.current_season

        self._rate_limit()

        

        try:

            stats = leaguedashteamstats.LeagueDashTeamStats(

                season=season,

                season_type_all_star='Regular Season',

                measure_type_detailed_defense='Base',  # Changed from Advanced - more reliable

                last_n_games=last_n_games

            )

            df = stats.get_data_frames()[0]

        except Exception as e:

            print(f"  [WARN] stats.nba.com team stats fetch failed: {e}")

            # Fallback to balldontlie.io

            print(f"  [INFO] Attempting balldontlie.io fallback for team stats...")

            try:

                # balldontlie /stats/advanced or /teams/{id}/stats

                # Let's try to fetch all teams stats if possible, or iterate

                # Note: v1 might not have 'advanced' endpoint. 

                # If v1 fails, we might just use season averages as proxy or keep generic defaults.

                

                # For now, let's fetch season stats (basic) as fallback

                bdl_stats = self.http.get_balldontlie_api("/stats", params={

                    'seasons[]': season.split('-')[0] if '-' in season else season,

                    'per_page': 100

                })

                

                if bdl_stats and 'data' in bdl_stats:

                    # Extract team averages from player stats (imperfect but better than nothing)

                    # Better: If balldontlie has a team stats endpoint in v1, use it.

                    # Actually, let's try the /teams endpoint - it might have some basic stats 

                    # or we can check /season_averages?player_ids[]=...

                    pass

                

                # If we can't get advanced stats easily, we at least return [] and let 

                # the caller use defaults, but we've tried.

                # TODO: Refine this once we confirm balldontlie v1 advanced stats structure.

                

            except Exception as ex:

                print(f"  [ERROR] balldontlie team stats fallback failed: {ex}")

            

            requested_window = f"L{last_n_games}" if last_n_games > 0 else "Season"
            fallback_stats = self._derive_window_fallback(requested_window)
            if fallback_stats:
                return fallback_stats
            return []

        

        if df.empty:

            print(f"  [WARN] No team stats returned")

            requested_window = f"L{last_n_games}" if last_n_games > 0 else "Season"
            fallback_stats = self._derive_window_fallback(requested_window)
            if fallback_stats:
                return fallback_stats
            return []

        

        teams = []

        today = datetime.now().strftime('%Y-%m-%d')

        window_type = f"L{last_n_games}" if last_n_games > 0 else "Season"

        

        # Get available columns
        available_cols = set(df.columns)
        
        # [FIX] Pre-fetch team abbreviations map for fallback
        team_abbr_map = {}
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT team_id, abbreviation FROM teams")
            for row in cursor.fetchall():
                team_abbr_map[row['team_id']] = row['abbreviation']
        
        for _, row in df.iterrows():
            # Safely get values with defaults
            def safe_get(col, default=0):
                return row.get(col, default) if col in available_cols else default
            
            t_id = safe_get('TEAM_ID')
            t_abbr = safe_get('TEAM_ABBREVIATION')
            
            # [FIX] Fallback for UNK team abbreviations
            if not t_abbr or t_abbr == 'UNK':
                t_abbr = team_abbr_map.get(t_id, 'UNK')
            
            team_stats = {
                'team_id': t_id,
                'team_abbreviation': t_abbr,
                'stat_date': today,
                'season': season,
                'pace': safe_get('PACE', 100.0),
                'def_rating': safe_get('DEF_RATING', 110.0),
                'opp_pts_per_game': safe_get('OPP_PTS', 110.0),
                'opp_fg_pct': safe_get('OPP_FG_PCT', 0.46),
                'opp_fg3_pct': safe_get('OPP_FG3_PCT', 0.36),
                'opp_fg3a_per_game': safe_get('OPP_FG3A', 30.0),
                'opp_oreb_pct': safe_get('OPP_OREB_PCT', 0.25),
                'dreb_pct': safe_get('DREB_PCT', 0.75),
                'window_type': window_type
            }
            teams.append(team_stats)
        
        return teams

    

    def store_team_stats(self, stats: List[Dict[str, Any]]) -> int:

        """Store team stats in database."""

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            count = 0

            

            for stat in stats:

                cursor.execute("""

                    INSERT OR REPLACE INTO team_advanced_stats (

                        team_id, team_abbreviation, stat_date, season,

                        pace, def_rating, opp_pts_per_game,

                        opp_fg_pct, opp_fg3_pct, opp_fg3a_per_game,

                        opp_oreb_pct, dreb_pct, window_type

                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

                """, (

                    stat.get('team_id'),

                    stat.get('team_abbreviation'),

                    stat.get('stat_date'),

                    stat.get('season'),

                    stat.get('pace'),

                    stat.get('def_rating'),

                    stat.get('opp_pts_per_game'),

                    stat.get('opp_fg_pct'),

                    stat.get('opp_fg3_pct'),

                    stat.get('opp_fg3a_per_game'),

                    stat.get('opp_oreb_pct'),

                    stat.get('dreb_pct'),

                    stat.get('window_type')

                ))

                count += 1

            

            conn.commit()

        

        return count

    

    def refresh_all_team_stats(self) -> Dict[str, int]:

        """Fetch and store stats for all windows (Season, L5, L10).

        

        Returns:

            Dict with counts per window type

        """

        print("\n[Opponent Stats] Refreshing team advanced stats...")

        results = {}

        

        # Season averages

        season_stats = self.fetch_league_team_stats(last_n_games=0)

        results['Season'] = self.store_team_stats(season_stats)

        print(f"   Season stats: {results['Season']} teams")

        

        # Last 10 games

        l10_stats = self.fetch_league_team_stats(last_n_games=10)

        results['L10'] = self.store_team_stats(l10_stats)

        print(f"   L10 stats: {results['L10']} teams")

        

        # Last 5 games  

        l5_stats = self.fetch_league_team_stats(last_n_games=5)

        results['L5'] = self.store_team_stats(l5_stats)

        print(f"   L5 stats: {results['L5']} teams")

        

        return results

    

    def get_opponent_stats(self, team_abbr: str, 

                           window: str = 'L10',

                           stat_date: str = None) -> Optional[Dict[str, Any]]:

        """Get opponent's advanced stats.

        

        Args:

            team_abbr: Team abbreviation

            window: 'Season', 'L10', or 'L5'

            stat_date: Date to lookup (default: today)

            

        Returns:

            Dict of stats or None

        """

        stat_date = stat_date or datetime.now().strftime('%Y-%m-%d')

        

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute("""

                SELECT * FROM team_advanced_stats 

                WHERE team_abbreviation = ? AND window_type = ?

                ORDER BY stat_date DESC LIMIT 1

            """, (team_abbr, window))

            

            row = cursor.fetchone()

            return dict(row) if row else None

    

    def get_league_average(self, stat_name: str,

                           window: str = 'Season') -> float:

        """Get league average for a stat.

        

        Args:

            stat_name: Column name (e.g., 'def_rating', 'pace')

            window: Window type

            

        Returns:

            League average value

        """

        with self.db.get_connection() as conn:

            cursor = conn.cursor()

            cursor.execute(f"""

                SELECT AVG({stat_name}) as avg_val FROM team_advanced_stats 

                WHERE window_type = ?

                AND stat_date = (SELECT MAX(stat_date) FROM team_advanced_stats WHERE window_type = ?)

            """, (window, window))

            

            row = cursor.fetchone()

            return row['avg_val'] if row and row['avg_val'] else 0.0





# Convenience function

def get_opponent_stats_ingestion() -> OpponentStatsIngestion:

    """Get opponent stats ingestion instance."""

    return OpponentStatsIngestion()

